use inkwell::types::{BasicTypeEnum, StringRadix, StructType};
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
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

use crate::tyck::{
    Type, VarId, RANGE_FROM_TY_NAME, RANGE_FULL_TY_NAME, RANGE_TO_TY_NAME, RANGE_TY_NAME,
};
use crate::{
    parser::ast::{DeclInfo, Defn, DefnKind, Expr, ExprKind, Stmt, StmtKind},
    tyck::StructInfo,
};
use tempfile::tempdir;

pub static STDLIB_PATH: LazyLock<OsString> = LazyLock::new(|| "lib/lib.c".into());

pub struct Emitter<'a, 'ctx, 'alloc> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub opt_fn: Option<FunctionValue<'ctx>>,

    struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    variables: HashMap<VarId, PointerValue<'ctx>>,
}

impl<'a, 'ctx, 'alloc> Emitter<'a, 'ctx, 'alloc> {
    #[inline]
    pub fn curr_fn(&self) -> FunctionValue<'ctx> {
        self.opt_fn.unwrap()
    }

    #[inline]
    fn unit_value(&self) -> BasicValueEnum<'ctx> {
        self.context.struct_type(&[], false).get_undef().into()
    }

    /// Emits LLVM IR for the given program as a `.ll` file.
    pub fn emit_ir(
        module_name: &str,
        dir_path: &Path,
        output_filename: Option<&Path>,
        program: &[Defn],
        struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    ) -> PathBuf {
        let context = &Context::create();
        let builder = &context.create_builder();
        let module = &context.create_module(module_name);

        let mut emitter = Emitter {
            context,
            builder,
            module,
            opt_fn: None,
            variables: HashMap::new(),
            struct_infos,
        };

        // Emit declarations first so that all types are present
        emitter.emit_struct_decls();
        emitter.emit_fn_decls(program);

        for defn in program {
            emitter.emit_defn(defn);
        }

        let ll_file = dir_path.join(&format!("{}.ll", module_name));
        module.print_to_file(OsStr::new(&ll_file)).unwrap();
        module.print_to_file(OsStr::new("a.ll")).unwrap();

        ll_file
    }

    fn emit_struct_decls(&self) {
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

    fn unit_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(&[], false)
    }

    fn generic_slice_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.unit_ty().ptr_type(AddressSpace::default()).into(),
                self.context.i32_type().into(),
            ],
            false,
        )
    }

    fn ty_to_ll_ty(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Bool => self.context.bool_type().into(),
            Type::Int => self.context.i32_type().into(),
            Type::Float => self.context.f32_type().into(),
            Type::Char => self.context.i8_type().into(),
            Type::Str => unreachable!("bare str cannot be repr'd as an LLVM type"),
            Type::Slice(_) => unreachable!("bare slice cannot be repr'd as an LLVM type"),
            Type::Struct(name) => self.context.get_struct_type(name).unwrap().into(),
            Type::Pointer(_, Type::Str | Type::Slice(_)) => self.generic_slice_ty().into(),
            Type::Pointer(_, ty) => self
                .ty_to_ll_ty(*ty)
                .ptr_type(AddressSpace::default())
                .into(),
            Type::Array(ty, size) => self.ty_to_ll_ty(*ty).array_type(size as u32).into(),
            Type::Tuple(tys) => {
                let ll_tys: Vec<_> = tys.iter().map(|ty| self.ty_to_ll_ty(*ty)).collect();
                self.context.struct_type(&ll_tys, false).into()
            }
            Type::Fn(param_tys, return_ty) => {
                let ll_param_tys: Vec<_> = param_tys
                    .iter()
                    .map(|ty| self.ty_to_ll_ty(*ty).into())
                    .collect();
                let ll_return_ty = self.ty_to_ll_ty(*return_ty);
                ll_return_ty
                    .fn_type(&ll_param_tys, false)
                    .ptr_type(AddressSpace::default())
                    .into()
            }
        }
    }

    fn emit_fn_decls(&mut self, defns: &[Defn]) {
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
            DefnKind::Struct { .. } => {}
            DefnKind::Fn {
                decl, params, body, ..
            } => {
                self.emit_fn(decl, params, body);
            }
            DefnKind::Static { decl, expr } => todo!(),
            DefnKind::ExternFn { .. } => {
                // fn already declared, no body to emit
            }
        }
    }

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

            self.builder.build_store(slot, param).expect("die");

            self.variables.insert(decl.id.get().expect("id"), slot);
        }

        let body = self.emit_expr(body);
        self.builder.build_return(Some(&body)).expect("die");
    }

    fn alloc_stack_slot(&self, ll_ty: BasicTypeEnum<'ctx>) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let curr_fn = self.opt_fn.unwrap();

        let entry = curr_fn.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(instr) => builder.position_at(entry, &instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(ll_ty, "").expect("die")
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
                let global = unsafe { self.builder.build_global_string(s, "") }.expect("die");
                let ptr = self
                    .builder
                    .build_pointer_cast(
                        global.as_pointer_value(),
                        self.unit_ty().ptr_type(AddressSpace::default()),
                        "",
                    )
                    .expect("die");
                self.build_str_slice(ptr, s.as_bytes().len())
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
                    self.builder.build_load(expr_ty, *ptr, "").expect("die")
                }
            }
            ExprKind::Block(stmts) => self.emit_block(stmts),
            _ => self.unit_value(),
            // ExprKind::PrefixOp(_, _) => todo!(),
            // ExprKind::BinOp(_, _, _) => todo!(),
            // ExprKind::Cast(_, _) => todo!(),
            // ExprKind::Group(_) => todo!(),
            // ExprKind::Field(_, _, _, _) => todo!(),
            // ExprKind::Call(_, _) => todo!(),
            // ExprKind::Index(_, _, _, _) => todo!(),
            // ExprKind::Range(_, _) => todo!(),
            // ExprKind::Struct(_, _) => todo!(),
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
                self.builder.build_store(slot, value).expect("die");
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
