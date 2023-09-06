use camino::Utf8Path;
use inkwell::support::LLVMString;
use inkwell::targets::{InitializationConfig, Target, TargetData};
use inkwell::types::{BasicTypeEnum, FunctionType, StringRadix, StructType};
use std::collections::HashMap;
use std::ffi::OsString;
use std::sync::LazyLock;

use inkwell::values::BasicValue;
use inkwell::AddressSpace;
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::BasicType,
    values::{BasicValueEnum, FunctionValue, PointerValue},
};

use crate::tyck::{Type, TypeKind, VarId};
use crate::{
    parser::ast::{DeclInfo, Defn, DefnKind, Expr, ExprKind, Stmt, StmtKind},
    tyck::StructInfo,
};

pub static STDLIB_PATH: LazyLock<OsString> = LazyLock::new(|| "lib/lib.c".into());

pub struct Emitter<'a, 'ctx, 'alloc> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub opt_fn: Option<FunctionValue<'ctx>>,

    target_data: &'a TargetData,
    struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    variables: HashMap<VarId, PointerValue<'ctx>>,
}

impl<'ctx, 'alloc> Emitter<'_, 'ctx, 'alloc> {
    #[inline]
    pub fn curr_fn(&self) -> FunctionValue<'ctx> {
        self.opt_fn.unwrap()
    }

    /// Unit value `{}`.
    #[inline]
    fn unit_value(&self) -> BasicValueEnum<'ctx> {
        self.context.struct_type(&[], false).get_undef().into()
    }

    /// Opaque pointer type `ptr`.
    #[inline]
    fn opaque_ptr_type(&self) -> BasicTypeEnum<'ctx> {
        self.unit_ty().ptr_type(AddressSpace::default()).into()
    }

    /// Emits LLVM IR for the given program as a `.ll` file.
    pub fn emit_ir(
        module_name: &str,
        output_path: &Utf8Path,
        program: &[Defn],
        struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    ) -> Result<(), LLVMString> {
        let context = &Context::create();
        let builder = &context.create_builder();
        let module = &context.create_module(module_name);

        Target::initialize_native(&InitializationConfig::default())
            .expect("failed to initialize target");

        let engine = module
            .create_execution_engine()
            .expect("failed to create execution engine");

        let mut emitter = Emitter {
            context,
            builder,
            module,
            opt_fn: None,
            variables: HashMap::new(),
            struct_infos,
            target_data: engine.get_target_data(),
        };

        // Emit declarations first so that all types are present
        emitter.add_struct_decls();
        emitter.add_fn_decls(program);

        for defn in program {
            emitter.emit_defn(defn);
        }

        module.print_to_file(&output_path)?;

        Ok(())
    }

    fn add_struct_decls(&self) {
        // Forward declare all structs to handle self/cyclical references.
        for (name, _) in &self.struct_infos {
            self.context.opaque_struct_type(name);
        }

        // Fill in struct fields for the types.
        for (name, info) in &self.struct_infos {
            let ty = self.module.get_struct_type(name).unwrap();
            let field_types: Vec<_> = info
                .members
                .iter()
                .map(|decl| self.ty_to_ll_ty(decl.ty.get().unwrap()))
                .collect();

            ty.set_body(&field_types, false);
        }
    }

    /// Unit type `()`.
    fn unit_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(&[], false)
    }

    fn generic_slice_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[self.opaque_ptr_type(), self.context.i32_type().into()],
            false,
        )
    }

    fn ty_to_ll_ty(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        use TypeKind::*;
        match ty.kind {
            Integral(_) => {
                unreachable!("all integral types should have been resolved to a sized integer type")
            }
            Bool => self.context.bool_type().into(),
            Byte | Char => self.context.i8_type().into(),
            Short | UShort => self.context.i16_type().into(),
            Int | UInt => self.context.i32_type().into(),
            Long | ULong => self.context.i64_type().into(),
            USize | ISize => self
                .context
                .ptr_sized_int_type(self.target_data, None)
                .into(),
            Float => self.context.f32_type().into(),
            Double => self.context.f64_type().into(),
            Str => unreachable!("bare str cannot be repr'd as an LLVM type"),
            Slice(_) => unreachable!("bare slice cannot be repr'd as an LLVM type"),
            Struct(name) => self.context.get_struct_type(name).unwrap().into(),
            Pointer(
                _,
                Type {
                    kind: Str | Slice(_),
                    ..
                },
            ) => self.generic_slice_ty().into(),
            Pointer(_, ty) => self
                .ty_to_ll_ty(*ty)
                .ptr_type(AddressSpace::default())
                .into(),
            Array(ty, size) => self.ty_to_ll_ty(*ty).array_type(size as u32).into(),
            Tuple(tys) => {
                let ll_tys: Vec<_> = tys.iter().map(|ty| self.ty_to_ll_ty(*ty)).collect();
                self.context.struct_type(&ll_tys, false).into()
            }
            Fn(..) => self
                .get_function_ty(ty)
                .ptr_type(AddressSpace::default())
                .into(),
        }
    }

    fn get_function_ty(&self, ty: Type<'_>) -> FunctionType<'ctx> {
        match ty.kind {
            TypeKind::Fn(param_tys, return_ty) => {
                let ll_param_tys: Vec<_> = param_tys
                    .iter()
                    .map(|ty| self.ty_to_ll_ty(*ty).into())
                    .collect();
                let ll_return_ty = self.ty_to_ll_ty(*return_ty);
                ll_return_ty.fn_type(&ll_param_tys, false)
            }
            _ => panic!("uh oh not a function"),
        }
    }

    fn add_fn_decls(&mut self, defns: &[Defn]) {
        for defn in defns {
            match &defn.kind {
                DefnKind::ExternFn {
                    decl,
                    params,
                    return_ty,
                    ..
                }
                | DefnKind::Fn {
                    decl,
                    params,
                    return_ty,
                    ..
                } => {
                    let return_ty = return_ty.get().unwrap();
                    let param_types: Vec<_> = params
                        .iter()
                        .map(|param| {
                            let param_ty = param.ty.get().unwrap();
                            self.ty_to_ll_ty(param_ty).into()
                        })
                        .collect();
                    self.module.add_function(
                        decl.name.item(),
                        self.ty_to_ll_ty(return_ty).fn_type(&param_types, false),
                        None,
                    );
                }
                _ => {}
            }
        }
    }

    pub fn emit_defn(&mut self, defn: &Defn) {
        match &defn.kind {
            DefnKind::Fn {
                decl, params, body, ..
            } => {
                self.emit_fn(decl, params, body);
            }
            DefnKind::Static { .. } => todo!("emit static variable definition"),
            _ => {}
        }
    }

    /// Emits a function definition.
    fn emit_fn(&mut self, decl: &DeclInfo, params: &[DeclInfo], body: &Expr) {
        let function = self
            .module
            .get_function(decl.name.item())
            .expect("function is declared");
        self.opt_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        for (param, decl) in function.get_param_iter().zip(params) {
            let slot = self.alloc_stack_slot(param.get_type());

            self.builder.build_store(slot, param);

            self.variables.insert(decl.id.get().expect("id"), slot);
        }

        let body = self.emit_expr(body);
        self.builder.build_return(Some(&body));
    }

    fn alloc_stack_slot(&self, ll_ty: BasicTypeEnum<'ctx>) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let curr_fn = self.opt_fn.unwrap();

        let entry = curr_fn.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(instr) => builder.position_at(entry, &instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(ll_ty, "")
    }

    /// Builds a `*str` fat pointer with the given address (`ptr`) and byte length (`size`).
    fn build_str_slice(&self, ptr: PointerValue<'ctx>, size: usize) -> BasicValueEnum<'ctx> {
        let mut slice = self.generic_slice_ty().get_undef();
        slice = self
            .builder
            .build_insert_value(slice, ptr, 0, "")
            .unwrap()
            .into_struct_value();
        slice = self
            .builder
            .build_insert_value(
                slice,
                self.context.i32_type().const_int(size as u64, false),
                1,
                "",
            )
            .unwrap()
            .into_struct_value();
        slice.as_basic_value_enum()
    }

    fn emit_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        match &expr.kind {
            ExprKind::Bool(v) => self.context.bool_type().const_int(*v as u64, false).into(),
            ExprKind::Int(v) => self
                .context
                .i32_type()
                .const_int_from_string(v, StringRadix::Decimal)
                .unwrap()
                .into(),
            ExprKind::Float(v) => self.context.f32_type().const_float_from_string(v).into(),
            ExprKind::Str(s) => {
                let global = self.builder.build_global_string_ptr(s, "");
                self.build_str_slice(global.as_pointer_value(), s.as_bytes().len())
            }
            ExprKind::Char(v) => self
                .context
                .i8_type()
                .const_int(v.bytes().nth(0).unwrap() as u64, false)
                .into(),
            ExprKind::Tuple(exprs) => {
                let mut struct_value = self
                    .ty_to_ll_ty(expr.ty.get().unwrap())
                    .into_struct_type()
                    .get_undef();

                // Evaluate each expression in order from left to right, assigning the result to each struct member.
                for (i, expr) in exprs.iter().enumerate() {
                    let value = self.emit_expr(expr);
                    struct_value = self
                        .builder
                        .build_insert_value(struct_value, value, i as u32, "")
                        .unwrap()
                        .into_struct_value();
                }
                struct_value.into()
            }
            ExprKind::Array(exprs, size) => {
                let ty = exprs[0].ty.get().unwrap();
                let ll_ty = self.ty_to_ll_ty(ty);
                let size = size.map(|s| *s.item()).unwrap_or_else(|| exprs.len());
                let arr_ty = ll_ty.array_type(size as u32);

                // Evaluate all exprs in order from left to right.
                let values: Vec<_> = exprs.iter().map(|expr| self.emit_expr(expr)).collect();

                // If the array size is longer than the number of exprs, repeat the values to fill in the gap.
                let mut arr = arr_ty.get_undef();
                for (i, value) in values.iter().cycle().take(size).enumerate() {
                    arr = self
                        .builder
                        .build_insert_value(arr, *value, i as u32, "")
                        .unwrap()
                        .into_array_value();
                }

                arr.as_basic_value_enum()
            }
            ExprKind::Id(name, var_id, is_func) => {
                if is_func.get() {
                    let func = self.module.get_function(name).unwrap();
                    func.as_global_value().as_pointer_value().into()
                } else {
                    let expr_ty = self.ty_to_ll_ty(expr.ty.get().unwrap());
                    let ptr = self.variables.get(&var_id.get().unwrap()).unwrap();
                    self.builder.build_load(expr_ty, *ptr, "")
                }
            }
            ExprKind::Block(stmts) => self.emit_block(stmts),
            ExprKind::Group(expr) => self.emit_expr(expr),
            ExprKind::Struct(name, init) => {
                let info = *self.struct_infos.get(name.item()).unwrap();
                let mut agg = self
                    .context
                    .get_struct_type(name.item())
                    .unwrap()
                    .get_undef()
                    .into();
                for (field, expr) in *init {
                    let idx = info.member_index(field.item());
                    let value = self.emit_expr(expr);
                    agg = self
                        .builder
                        .build_insert_value(agg, value, idx as u32, "")
                        .unwrap();
                }
                agg.as_basic_value_enum()
            }
            ExprKind::Range(start, end) => {
                let start = start.map(|e| self.emit_expr(e));
                let end = end.map(|e| self.emit_expr(e));

                match (start, end) {
                    (Some(start), Some(end)) => {
                        let mut agg = self
                            .context
                            .get_struct_type("Range")
                            .unwrap()
                            .get_undef()
                            .into();
                        agg = self.builder.build_insert_value(agg, start, 0, "").unwrap();
                        agg = self.builder.build_insert_value(agg, end, 1, "").unwrap();
                        agg.as_basic_value_enum()
                    }
                    (None, Some(end)) => {
                        let mut agg = self
                            .context
                            .get_struct_type("RangeTo")
                            .unwrap()
                            .get_undef()
                            .into();
                        agg = self.builder.build_insert_value(agg, end, 0, "").unwrap();
                        agg.as_basic_value_enum()
                    }
                    (Some(start), None) => {
                        let mut agg = self
                            .context
                            .get_struct_type("RangeFrom")
                            .unwrap()
                            .get_undef()
                            .into();
                        agg = self.builder.build_insert_value(agg, start, 0, "").unwrap();
                        agg.as_basic_value_enum()
                    }
                    (None, None) => self
                        .context
                        .get_struct_type("RangeFull")
                        .unwrap()
                        .get_undef()
                        .as_basic_value_enum(),
                }
            }
            ExprKind::Call(callable, args) => {
                let callee = self.emit_expr(callable);
                let args: Vec<_> = args.iter().map(|arg| self.emit_expr(arg).into()).collect();
                self.builder
                    .build_indirect_call(
                        self.get_function_ty(callable.ty.get().unwrap()),
                        callee.into_pointer_value(),
                        &args[..],
                        "",
                    )
                    .try_as_basic_value()
                    .left_or(self.unit_value())
            }
            // ExprKind::Cast(expr, _) => {
            //     let value = self.emit_expr(expr);
            //     let expr_ll_ty = self.ty_to_ll_ty(expr.ty.get().unwrap());

            //     match (value.get_type(), expr.ty.get()) {
            //         (ty, _) if ty == expr_ll_ty => value,
            //         (
            //             BasicTypeEnum::PointerType(_),
            //             Some(Type {
            //                 kind: TypeKind::Pointer(_, to_ty),
            //                 ..
            //             }),
            //         ) => self
            //             .builder
            //             .build_pointer_cast(
            //                 value.into_pointer_value(),
            //                 self.ty_to_ll_ty(*to_ty)
            //                     .ptr_type(AddressSpace::default())
            //                     .into(),
            //                 "",
            //             )
            //             .into(),
            //         _ => unreachable!("only thin pointer casts supported"),
            //     }
            // }
            _ => self.unit_value(),
            // ExprKind::PrefixOp(_, _) => todo!(),
            // ExprKind::BinOp(_, _, _) => todo!(),
            // ExprKind::Field(_, _, _, _) => todo!(),
            // ExprKind::Index(_, _, _, _) => todo!(),
            // ExprKind::If(_, _) => todo!(),
        }
    }

    fn emit_block(&mut self, stmts: &[Stmt]) -> BasicValueEnum<'ctx> {
        let mut value = self.unit_value();
        for stmt in stmts {
            value = self.emit_stmt(stmt);
        }
        value
    }

    fn emit_stmt(&mut self, stmt: &Stmt) -> BasicValueEnum<'ctx> {
        match &stmt.kind {
            StmtKind::Let(decl, expr) => {
                let value = self.emit_expr(expr);
                let ty = self.ty_to_ll_ty(decl.ty.get().unwrap());
                let slot = self.alloc_stack_slot(ty);
                self.builder.build_store(slot, value);
                self.variables.insert(decl.id.get().unwrap(), slot);
                self.unit_value()
            }
            StmtKind::Expr(expr) => self.emit_expr(expr),
            StmtKind::Semi(expr) => {
                self.emit_expr(expr);
                self.unit_value()
            }
        }
    }
}
