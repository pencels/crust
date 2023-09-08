use camino::Utf8Path;
use inkwell::support::LLVMString;
use inkwell::targets::{InitializationConfig, Target, TargetData};
use inkwell::types::{BasicTypeEnum, FunctionType, IntType, PointerType, StringRadix, StructType};
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

use crate::parser::ast::{BinOpKind, Operator};
use crate::tyck::{Type, TypeKind, VarId};
use crate::util;
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
    /// Unit value `{}`.
    #[inline]
    fn unit_value(&self) -> BasicValueEnum<'ctx> {
        self.context.struct_type(&[], false).get_undef().into()
    }

    /// Opaque pointer type `ptr`.
    #[inline]
    fn opaque_ptr_type(&self) -> PointerType<'ctx> {
        self.unit_type().ptr_type(AddressSpace::default())
    }

    /// Returns the `usize` type equivalent.
    #[inline]
    fn size_type(&self) -> IntType<'ctx> {
        self.context.ptr_sized_int_type(self.target_data, None)
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

    fn extract_char_int_value(&self, c: &str) -> u64 {
        let unescaped = util::escape::unescape(c).unwrap();
        let c = unescaped.chars().nth(0).unwrap();

        if unescaped.len() == 1 && c.is_ascii() {
            c as u64
        } else {
            panic!("'{}' is not ascii", c);
        }
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
                .map(|decl| self.ty_to_ll_type(decl.ty.get().unwrap()))
                .collect();

            ty.set_body(&field_types, false);
        }
    }

    /// Unit type `()`.
    fn unit_type(&self) -> StructType<'ctx> {
        self.context.struct_type(&[], false)
    }

    fn generic_slice_type(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.opaque_ptr_type().into(),
                self.context.i32_type().into(),
            ],
            false,
        )
    }

    fn ty_to_ll_type(&self, ty: Type) -> BasicTypeEnum<'ctx> {
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
            ) => self.generic_slice_type().into(),
            Pointer(_, ty) => self
                .ty_to_ll_type(*ty)
                .ptr_type(AddressSpace::default())
                .into(),
            Array(ty, size) => self.ty_to_ll_type(*ty).array_type(size as u32).into(),
            Tuple(tys) => {
                let ll_tys: Vec<_> = tys.iter().map(|ty| self.ty_to_ll_type(*ty)).collect();
                self.context.struct_type(&ll_tys, false).into()
            }
            Fn(..) => self
                .get_function_type(ty)
                .ptr_type(AddressSpace::default())
                .into(),
        }
    }

    fn add_crust_function(&mut self, name: &str, ty: FunctionType<'ctx>) -> FunctionValue<'ctx> {
        self.module
            .add_function(&("__crust__".to_owned() + name), ty, None)
    }

    fn get_crust_function(&self, name: &str) -> Option<FunctionValue<'ctx>> {
        self.module.get_function(&("__crust__".to_owned() + name))
    }

    fn get_function_type(&self, ty: Type<'_>) -> FunctionType<'ctx> {
        match ty.kind {
            TypeKind::Fn(param_tys, return_ty) => {
                let ll_param_tys: Vec<_> = param_tys
                    .iter()
                    .map(|ty| self.ty_to_ll_type(*ty).into())
                    .collect();
                let ll_return_ty = self.ty_to_ll_type(*return_ty);
                ll_return_ty.fn_type(&ll_param_tys, false)
            }
            _ => panic!("uh oh not a function"),
        }
    }

    fn add_fn_decls(&mut self, defns: &[Defn]) {
        for defn in defns {
            let extern_c = match &defn.kind {
                DefnKind::ExternFn { c, .. } => *c,
                DefnKind::Fn { .. } => false,
                _ => false,
            };

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
                    let param_types: Vec<_> = params
                        .iter()
                        .map(|param| {
                            let param_ty = param.ty.get().unwrap();
                            self.ty_to_ll_type(param_ty).into()
                        })
                        .collect();
                    let return_ty = return_ty.get().unwrap();
                    let return_ty = if decl.name.item() == &"main" && return_ty.is_unit() {
                        Type::INT
                    } else {
                        return_ty
                    };

                    let ll_fn_ty = self.ty_to_ll_type(return_ty).fn_type(&param_types, false);

                    if extern_c {
                        self.module.add_function(decl.name.item(), ll_fn_ty, None);
                    } else {
                        self.add_crust_function(decl.name.item(), ll_fn_ty);
                    }
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
                let ret_ty = match decl.ty.get().unwrap().kind {
                    TypeKind::Fn(_, ret_ty) => ret_ty,
                    _ => panic!("umm should be a fn type"),
                };
                self.emit_fn(decl, params, ret_ty, body);
            }
            DefnKind::Static { .. } => todo!("emit static variable definition"),
            _ => {}
        }
    }

    /// Emits a function definition.
    fn emit_fn(&mut self, decl: &DeclInfo, params: &[DeclInfo], ret_ty: &Type, body: &Expr) {
        let function = self
            .get_crust_function(decl.name.item())
            .expect("ICE: crust internal function not found");
        self.opt_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        for (param, decl) in function.get_param_iter().zip(params) {
            let slot = self.alloc_stack_slot(param.get_type());

            self.builder.build_store(slot, param);

            self.variables.insert(decl.id.get().expect("id"), slot);
        }

        let body = self.emit_expr(body);

        if decl.name.item() == &"main" && ret_ty == &Type::UNIT {
            self.builder
                .build_return(Some(&self.context.i32_type().const_zero()));
        } else {
            self.builder.build_return(Some(&body));
        }
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
        let mut slice = self.generic_slice_type().get_undef();
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

    fn int_type(&self, ty: &Type) -> IntType<'ctx> {
        match ty.kind {
            TypeKind::Integral(_) => unreachable!("{}", "{integer} should be resolved by now"),
            TypeKind::Byte | TypeKind::Char => self.context.i8_type(),
            TypeKind::Short | TypeKind::UShort => self.context.i16_type(),
            TypeKind::Int | TypeKind::UInt => self.context.i32_type(),
            TypeKind::Long | TypeKind::ULong => self.context.i32_type(),
            TypeKind::USize | TypeKind::ISize => {
                self.context.ptr_sized_int_type(self.target_data, None)
            }
            _ => panic!("int_type called on non-int type {:?}", ty.kind),
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        let value = match &expr.kind {
            ExprKind::Bool(v) => self.context.bool_type().const_int(*v as u64, false).into(),
            ExprKind::Int(v) => {
                let ty = expr.effective_ty().unwrap();
                if ty.is_int() {
                    self.int_type(&ty)
                        .const_int_from_string(v, StringRadix::Decimal)
                        .unwrap()
                        .into()
                } else if ty == Type::BOOL {
                    let val = self
                        .size_type()
                        .const_int_from_string(v, StringRadix::Decimal)
                        .unwrap();
                    self.builder
                        .build_int_compare(
                            inkwell::IntPredicate::NE,
                            val,
                            self.size_type().const_zero(),
                            "",
                        )
                        .as_basic_value_enum()
                } else if ty.is_sized_ptr() {
                    let val = self
                        .context
                        .ptr_sized_int_type(self.target_data, None)
                        .const_int_from_string(v, StringRadix::Decimal)
                        .unwrap();
                    self.builder
                        .build_int_to_ptr(val, self.opaque_ptr_type(), "")
                        .as_basic_value_enum()
                } else if ty.is_float() {
                    let val = self
                        .size_type()
                        .const_int_from_string(v, StringRadix::Decimal)
                        .unwrap();
                    self.builder
                        .build_signed_int_to_float(val, self.context.f64_type(), "")
                        .as_basic_value_enum()
                } else {
                    unreachable!("ICE: int literal coerced to invalid type")
                }
            }
            ExprKind::Float(v) => self.context.f64_type().const_float_from_string(v).into(),
            ExprKind::Str(s) => {
                let s = util::escape::unescape(s).unwrap();
                let global = self.builder.build_global_string_ptr(&s, "");
                self.build_str_slice(global.as_pointer_value(), s.as_bytes().len())
            }
            ExprKind::Char(v) => self
                .context
                .i8_type()
                .const_int(self.extract_char_int_value(v), false)
                .into(),
            ExprKind::Tuple(exprs) => {
                let mut struct_value = self
                    .ty_to_ll_type(expr.ty.get().unwrap())
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
                let ll_ty = self.ty_to_ll_type(ty);
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
            ExprKind::Id(name, var_id, decl) => {
                let decl = decl.get().unwrap();
                if decl.is_fn {
                    let func = match decl.extern_c {
                        Some(true) => self.module.get_function(name),
                        Some(false) => self.get_crust_function(name),
                        None => unreachable!(),
                    };
                    func.unwrap().as_global_value().as_pointer_value().into()
                } else {
                    let expr_ty = self.ty_to_ll_type(expr.ty.get().unwrap());
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
                        self.get_function_type(callable.ty.get().unwrap()),
                        callee.into_pointer_value(),
                        &args[..],
                        "",
                    )
                    .try_as_basic_value()
                    .left_or(self.unit_value())
            }
            ExprKind::Cast(inner, _) => {
                let value = self.emit_expr(inner);
                let inner_ty = inner.ty.get().unwrap();
                let as_ty = expr.ty.get().unwrap();

                match (inner_ty.kind, as_ty.kind) {
                    // Drop fatness (tyck ensures that the to_ty can only be a *())
                    (TypeKind::Pointer(_, from), TypeKind::Pointer(_, to))
                        if !from.is_sized() && to.is_sized() =>
                    {
                        self.builder
                            .build_extract_value(value.into_struct_value(), 0, "")
                            .unwrap()
                            .into()
                    }
                    _ => self.build_conversion(value, inner_ty, as_ty),
                }
            }
            ExprKind::If(thens, last) => {
                let func = self.opt_fn.unwrap();
                let if_ty = expr.ty.get().unwrap();

                let dest = self.context.append_basic_block(func, "");
                let dest_builder = self.context.create_builder();
                dest_builder.position_at_end(dest);
                let phi = dest_builder.build_phi(self.ty_to_ll_type(if_ty), "");

                for (cond, then) in thens.iter() {
                    // Emit cond instrs and coerce to bool
                    let cond_value = self.emit_expr(cond);

                    // Branch to either then or cond/else blocks
                    let then_block = self.context.append_basic_block(func, "");
                    let cond_or_else_block = self.context.append_basic_block(func, "");
                    self.builder.build_conditional_branch(
                        cond_value.into_int_value(),
                        then_block,
                        cond_or_else_block,
                    );

                    // Emit instrs into "then" block and branch to destination phi instr
                    self.builder.position_at_end(then_block);
                    let then_value = self.emit_expr(then);
                    self.builder.build_unconditional_branch(dest);
                    phi.add_incoming(&[(&then_value as &dyn BasicValue, then_block)]);

                    // Position builder for next iteration
                    self.builder.position_at_end(cond_or_else_block);
                }

                // Last block either has some instrs or results in `()`
                let else_block = self.builder.get_insert_block().unwrap();
                let last_value = if let Some(last) = last {
                    self.emit_expr(last)
                } else {
                    self.unit_value()
                };
                self.builder.build_unconditional_branch(dest);
                phi.add_incoming(&[(&last_value as &dyn BasicValue, else_block)]);

                // Instructions continue at the destination block
                self.builder.position_at_end(dest);
                phi.as_basic_value()
            }
            ExprKind::BinOp(op, lhs, rhs) => {
                let ty = expr.ty.get().unwrap();

                // These operators evaluate the exprs conditionally
                if let Operator::Simple(op @ (BinOpKind::AmpAmp | BinOpKind::PipePipe)) = op.item()
                {
                    let lhs_block = self.builder.get_insert_block().unwrap();
                    let rhs_block = self.context.append_basic_block(self.opt_fn.unwrap(), "");
                    let dest_block = self.context.append_basic_block(self.opt_fn.unwrap(), "");

                    let lhs_value = self.emit_expr(lhs);
                    if op == &BinOpKind::AmpAmp {
                        self.builder.build_conditional_branch(
                            lhs_value.into_int_value(),
                            rhs_block,
                            dest_block,
                        );
                    } else {
                        self.builder.build_conditional_branch(
                            lhs_value.into_int_value(),
                            dest_block,
                            rhs_block,
                        );
                    }

                    self.builder.position_at_end(rhs_block);
                    let rhs_value = self.emit_expr(rhs);
                    self.builder.build_unconditional_branch(dest_block);

                    self.builder.position_at_end(dest_block);
                    let phi = self.builder.build_phi(self.ty_to_ll_type(ty), "");
                    phi.add_incoming(&[(&lhs_value, lhs_block), (&rhs_value, rhs_block)]);
                    return phi.as_basic_value();
                };

                let lhs_value = self.emit_expr(lhs);
                let rhs_value = self.emit_expr(rhs);
                match op.item() {
                    Operator::Simple(op) => match op {
                        BinOpKind::Plus => {
                            if ty.is_int() {
                                self.builder
                                    .build_int_add(
                                        lhs_value.into_int_value(),
                                        rhs_value.into_int_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_float() {
                                self.builder
                                    .build_float_add(
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_sized_ptr() {
                                let (ptr, index) = if lhs.ty.get().unwrap().is_sized_ptr() {
                                    (lhs_value, rhs_value)
                                } else {
                                    (rhs_value, lhs_value)
                                };
                                unsafe {
                                    self.builder
                                        .build_gep(
                                            self.ty_to_ll_type(*ty.pointee_ty().unwrap()),
                                            ptr.into_pointer_value(),
                                            &[index.into_int_value()],
                                            "",
                                        )
                                        .as_basic_value_enum()
                                }
                            } else {
                                unreachable!()
                            }
                        }
                        BinOpKind::Minus => {
                            if ty == Type::ISIZE {
                                self.builder
                                    .build_int_sub(
                                        lhs_value.into_int_value(),
                                        rhs_value.into_int_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_int() {
                                self.builder
                                    .build_int_sub(
                                        lhs_value.into_int_value(),
                                        rhs_value.into_int_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_float() {
                                self.builder
                                    .build_float_sub(
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_sized_ptr() {
                                let rhs_value =
                                    self.builder.build_int_neg(rhs_value.into_int_value(), "");
                                unsafe {
                                    self.builder
                                        .build_gep(
                                            self.ty_to_ll_type(*ty.pointee_ty().unwrap()),
                                            lhs_value.into_pointer_value(),
                                            &[rhs_value],
                                            "",
                                        )
                                        .as_basic_value_enum()
                                }
                            } else {
                                unreachable!()
                            }
                        }

                        BinOpKind::Star => {
                            if ty.is_int() {
                                self.builder
                                    .build_int_mul(
                                        lhs_value.into_int_value(),
                                        rhs_value.into_int_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_float() {
                                self.builder
                                    .build_float_mul(
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else {
                                unreachable!()
                            }
                        }
                        BinOpKind::Slash => {
                            if ty.is_int() {
                                if ty.is_signed().unwrap() {
                                    self.builder
                                        .build_int_signed_div(
                                            lhs_value.into_int_value(),
                                            rhs_value.into_int_value(),
                                            "",
                                        )
                                        .as_basic_value_enum()
                                } else {
                                    self.builder
                                        .build_int_unsigned_div(
                                            lhs_value.into_int_value(),
                                            rhs_value.into_int_value(),
                                            "",
                                        )
                                        .as_basic_value_enum()
                                }
                            } else if ty.is_float() {
                                self.builder
                                    .build_float_div(
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else {
                                unreachable!()
                            }
                        }
                        BinOpKind::Percent => {
                            if ty.is_int() {
                                if ty.is_signed().unwrap() {
                                    self.builder
                                        .build_int_signed_rem(
                                            lhs_value.into_int_value(),
                                            rhs_value.into_int_value(),
                                            "",
                                        )
                                        .as_basic_value_enum()
                                } else {
                                    self.builder
                                        .build_int_unsigned_rem(
                                            lhs_value.into_int_value(),
                                            rhs_value.into_int_value(),
                                            "",
                                        )
                                        .as_basic_value_enum()
                                }
                            } else if ty.is_float() {
                                self.builder
                                    .build_float_rem(
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else {
                                unreachable!()
                            }
                        }
                        BinOpKind::Caret => self
                            .builder
                            .build_xor(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                            .as_basic_value_enum(),
                        BinOpKind::Amp => self
                            .builder
                            .build_and(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                            .as_basic_value_enum(),
                        BinOpKind::Pipe => self
                            .builder
                            .build_or(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                            .as_basic_value_enum(),
                        BinOpKind::LtLt => self
                            .builder
                            .build_left_shift(
                                lhs_value.into_int_value(),
                                rhs_value.into_int_value(),
                                "",
                            )
                            .as_basic_value_enum(),
                        BinOpKind::GtGt => self
                            .builder
                            .build_right_shift(
                                lhs_value.into_int_value(),
                                rhs_value.into_int_value(),
                                ty.is_signed().unwrap(),
                                "",
                            )
                            .as_basic_value_enum(),
                        BinOpKind::Lt
                        | BinOpKind::Le
                        | BinOpKind::Gt
                        | BinOpKind::Ge
                        | BinOpKind::EqEq
                        | BinOpKind::Ne => {
                            let ty = lhs.effective_ty().unwrap();
                            if ty.is_float() {
                                self.builder
                                    .build_float_compare(
                                        self.float_cmp_op(op).unwrap(),
                                        lhs_value.into_float_value(),
                                        rhs_value.into_float_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else if ty.is_int() {
                                self.builder
                                    .build_int_compare(
                                        self.int_cmp_op(op, ty.is_signed().unwrap()).unwrap(),
                                        lhs_value.into_int_value(),
                                        rhs_value.into_int_value(),
                                        "",
                                    )
                                    .as_basic_value_enum()
                            } else {
                                todo!()
                            }
                        }
                        BinOpKind::AmpAmp | BinOpKind::PipePipe => unreachable!(
                            "ICE: lazy boolean operators should have been handled already"
                        ),
                    },
                    Operator::Assign(_) => todo!(),
                }
            }
            _ => todo!(),
            // ExprKind::PrefixOp(_, _) => todo!(),
            // ExprKind::Field(_, _, _, _) => todo!(),
            // ExprKind::Index(_, _, _, _) => todo!(),
        };

        // If there's a coercion, apply type conversion after the main expression has been emitted.
        match (expr.ty.get(), expr.coerced_ty.get()) {
            (Some(from_ty), Some(to_ty)) => self.build_conversion(value, from_ty, to_ty),
            _ => value,
        }
    }

    fn int_cmp_op(&self, op: &BinOpKind, signed: bool) -> Option<inkwell::IntPredicate> {
        Some(match op {
            BinOpKind::Lt => {
                if signed {
                    inkwell::IntPredicate::SLT
                } else {
                    inkwell::IntPredicate::ULT
                }
            }
            BinOpKind::Le => {
                if signed {
                    inkwell::IntPredicate::SLE
                } else {
                    inkwell::IntPredicate::ULE
                }
            }
            BinOpKind::Gt => {
                if signed {
                    inkwell::IntPredicate::SGT
                } else {
                    inkwell::IntPredicate::UGT
                }
            }
            BinOpKind::Ge => {
                if signed {
                    inkwell::IntPredicate::SGE
                } else {
                    inkwell::IntPredicate::UGE
                }
            }
            BinOpKind::EqEq => inkwell::IntPredicate::EQ,
            BinOpKind::Ne => inkwell::IntPredicate::NE,
            _ => return None,
        })
    }

    fn float_cmp_op(&self, op: &BinOpKind) -> Option<inkwell::FloatPredicate> {
        Some(match op {
            BinOpKind::Lt => inkwell::FloatPredicate::OLT,
            BinOpKind::Le => inkwell::FloatPredicate::OLE,
            BinOpKind::Gt => inkwell::FloatPredicate::OGT,
            BinOpKind::Ge => inkwell::FloatPredicate::OGE,
            BinOpKind::EqEq => inkwell::FloatPredicate::OEQ,
            BinOpKind::Ne => inkwell::FloatPredicate::ONE,
            _ => return None,
        })
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
                let ty = self.ty_to_ll_type(decl.ty.get().unwrap());
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

    fn build_conversion(
        &mut self,
        value: BasicValueEnum<'ctx>,
        from_ty: Type,
        to_ty: Type,
    ) -> BasicValueEnum<'ctx> {
        match value {
            BasicValueEnum::IntValue(int_value) => match to_ty.kind {
                TypeKind::Bool => self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::NE,
                        int_value,
                        int_value.get_type().const_int(0, false),
                        "",
                    )
                    .as_basic_value_enum(),
                TypeKind::Integral(_) => {
                    unreachable!("{}", "ICE: {integer} types should have been resolved")
                }
                TypeKind::Byte
                | TypeKind::Char
                | TypeKind::Short
                | TypeKind::UShort
                | TypeKind::Int
                | TypeKind::UInt
                | TypeKind::Long
                | TypeKind::ULong
                | TypeKind::USize
                | TypeKind::ISize => {
                    let to_int_type = self.ty_to_ll_type(to_ty).into_int_type();
                    if int_value.get_type().get_bit_width() > to_int_type.get_bit_width() {
                        self.builder
                            .build_int_truncate(int_value, to_int_type, "")
                            .as_basic_value_enum()
                    } else {
                        match from_ty.is_signed() {
                            Some(true) => self
                                .builder
                                .build_int_s_extend(int_value, to_int_type, "")
                                .as_basic_value_enum(),
                            Some(false) | None => self
                                .builder
                                .build_int_z_extend(int_value, to_int_type, "")
                                .as_basic_value_enum(),
                        }
                    }
                }
                TypeKind::Float | TypeKind::Double => {
                    let to_float_type = self.ty_to_ll_type(to_ty).into_float_type();
                    match from_ty.is_signed() {
                        Some(true) => self
                            .builder
                            .build_signed_int_to_float(int_value, to_float_type, "")
                            .as_basic_value_enum(),
                        Some(false) | None => self
                            .builder
                            .build_unsigned_int_to_float(int_value, to_float_type, "")
                            .as_basic_value_enum(),
                    }
                }
                TypeKind::Pointer(_, ty) if ty.is_sized() => self
                    .builder
                    .build_int_to_ptr(int_value, self.opaque_ptr_type(), "")
                    .as_basic_value_enum(),
                TypeKind::Fn(_, _) => self
                    .builder
                    .build_int_to_ptr(
                        int_value,
                        self.ty_to_ll_type(to_ty).ptr_type(AddressSpace::default()),
                        "",
                    )
                    .as_basic_value_enum(),
                _ => unreachable!("ICE: conversion not supported"),
            },
            BasicValueEnum::FloatValue(float_value) => match to_ty.kind {
                TypeKind::Bool => self
                    .builder
                    .build_float_compare(
                        inkwell::FloatPredicate::UNE,
                        float_value,
                        float_value.get_type().const_zero(),
                        "",
                    )
                    .as_basic_value_enum(),
                TypeKind::Integral(_) => {
                    unreachable!("{}", "ICE: {integer} types should have been resolved")
                }
                TypeKind::Byte
                | TypeKind::Char
                | TypeKind::Short
                | TypeKind::UShort
                | TypeKind::Int
                | TypeKind::UInt
                | TypeKind::Long
                | TypeKind::ULong
                | TypeKind::USize
                | TypeKind::ISize => {
                    let int_type = self.ty_to_ll_type(to_ty).into_int_type();
                    match to_ty.is_signed() {
                        Some(true) => self
                            .builder
                            .build_float_to_signed_int(float_value, int_type, "")
                            .as_basic_value_enum(),
                        Some(false) => self
                            .builder
                            .build_float_to_unsigned_int(float_value, int_type, "")
                            .as_basic_value_enum(),
                        None => unreachable!(),
                    }
                }
                TypeKind::Float | TypeKind::Double => {
                    let to_float_type = self.ty_to_ll_type(to_ty).into_float_type();
                    if from_ty.bit_width() < to_ty.bit_width() {
                        self.builder
                            .build_float_ext(float_value, to_float_type, "")
                            .as_basic_value_enum()
                    } else {
                        self.builder
                            .build_float_trunc(float_value, to_float_type, "")
                            .as_basic_value_enum()
                    }
                }
                _ => unreachable!("ICE: conversion not supported"),
            },
            BasicValueEnum::PointerValue(ptr_value) => match to_ty.kind {
                TypeKind::Bool => self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::NE,
                        ptr_value,
                        self.opaque_ptr_type().const_null(),
                        "",
                    )
                    .as_basic_value_enum(),
                TypeKind::Integral(_) => {
                    unreachable!("{}", "ICE: {integer} types should have been resolved")
                }
                TypeKind::Byte
                | TypeKind::Char
                | TypeKind::Short
                | TypeKind::UShort
                | TypeKind::Int
                | TypeKind::UInt
                | TypeKind::Long
                | TypeKind::ULong
                | TypeKind::USize
                | TypeKind::ISize => {
                    let to_int_type = self.ty_to_ll_type(to_ty).into_int_type();
                    self.builder
                        .build_ptr_to_int(ptr_value, to_int_type, "")
                        .as_basic_value_enum()
                }
                TypeKind::Pointer(_, t) if t.is_sized() => ptr_value.as_basic_value_enum(),
                TypeKind::Fn(..) => ptr_value.as_basic_value_enum(),
                _ => unreachable!("ICE: conversion not supported"),
            },
            _ => value,
        }
    }
}
