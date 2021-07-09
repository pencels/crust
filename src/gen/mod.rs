use std::{collections::HashMap, path::PathBuf};

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicType, StringRadix, StructType},
    values::{BasicValueEnum, FunctionValue, PointerValue},
};

use crate::{
    parser::ast::{DeclInfo, Defn, DefnKind, Expr, ExprKind, Stmt, StmtKind, Type},
    tyck::{self, StructInfo},
};

pub struct Emitter<'a, 'ctx, 'alloc> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub opt_fn: Option<FunctionValue<'ctx>>,

    struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    variables: HashMap<String, PointerValue<'ctx>>,
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

    pub fn emit_program(
        module_name: &str,
        program: &[Defn],
        struct_infos: HashMap<&'alloc str, StructInfo<'alloc>>,
    ) {
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

        emitter.emit_struct_decls();

        for defn in program {
            emitter.emit_defn(defn);
        }

        let mut out_path = PathBuf::from(module_name);
        out_path.set_extension("ll");
        module.print_to_file(out_path).unwrap();
    }

    fn emit_struct_decls(&self) {
        for (name, _) in &self.struct_infos {
            self.context.opaque_struct_type(name);
        }
        for (name, info) in &self.struct_infos {
            let ty = self.module.get_struct_type(name).expect("die");
            let field_types: Vec<_> = info
                .members
                .iter()
                .map(|decl| self.ty_to_ll_ty(decl.ty))
                .collect();

            ty.set_body(field_types.as_slice(), false);
        }
    }

    pub fn emit_defn(&mut self, defn: &Defn) {
        match &defn.kind {
            DefnKind::Struct { .. } => {}
            DefnKind::Fn {
                decl,
                params,
                return_ty_ast,
                return_ty,
                body,
            } => {
                let return_ty = return_ty
                    .get()
                    .expect("ICE: return ty should be set in tyck phase");
                self.emit_fn(decl, params, return_ty, body);
            }
            DefnKind::Static { decl, expr } => todo!(),
        }
    }

    fn emit_fn(
        &mut self,
        decl: &DeclInfo,
        params: &[DeclInfo],
        return_ty: tyck::Type,
        body: &Expr,
    ) {
        let mut param_types: Vec<_> = params
            .iter()
            .map(|param| {
                let param_ty = param
                    .ty
                    .get()
                    .expect("ICE: type not resolved in tyck phase");
                self.ty_to_ll_ty(param_ty)
            })
            .collect();
        let function = self.module.add_function(
            decl.name.item(),
            self.ty_to_ll_ty(return_ty).fn_type(&mut param_types, false),
            None,
        );

        self.opt_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        for (i, param) in function.get_param_iter().enumerate() {
            let name = params[i].name.item();
            let slot = self.alloc_stack_slot(name, param.get_type());

            self.builder.build_store(slot, param);

            self.variables.insert(name.to_string(), slot);
        }

        let body = self.emit_expr(body);
        self.builder.build_return(Some(&body));
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
                let ty = self.ty_to_ll_ty(decl.ty.get().expect("ICE: decl doesn't have ty"));
                let name = decl.name.item();
                let slot = self.alloc_stack_slot(name, ty);
                self.builder.build_store(slot, value);
                self.variables.insert(name.to_string(), slot);
                self.unit_value()
            }
            StmtKind::Expr(expr) => self.emit_expr(expr),
            StmtKind::Semi(expr) => {
                self.emit_expr(expr);
                self.unit_value()
            }
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        match &expr.kind {
            ExprKind::Bool(b) => self.context.bool_type().const_int(*b as u64, false).into(),
            ExprKind::Int(i) => self
                .context
                .i32_type()
                .const_int_from_string(i, StringRadix::Decimal)
                .unwrap()
                .into(),
            ExprKind::Float(f) => self.context.f32_type().const_float_from_string(f).into(),
            ExprKind::Str(_) => todo!(),
            ExprKind::Char(c) => todo!(),
            ExprKind::Tuple(exprs) => {
                let mut struct_value = self
                    .ty_to_ll_ty(
                        expr.ty
                            .get()
                            .expect("ICE: expr ty should be filled by now..."),
                    )
                    .into_struct_type()
                    .get_undef();
                for (i, expr) in exprs.iter().enumerate() {
                    let value = self.emit_expr(expr);
                    struct_value = self
                        .builder
                        .build_insert_value(struct_value, value, i as u32, "")
                        .expect("ICE: please work")
                        .into_struct_value();
                }
                struct_value.into()
            }
            ExprKind::Array(_, _) => todo!(),
            ExprKind::Id(name, id) => {
                let ptr = self.variables.get(*name).expect("ICE: undefined variable");
                self.builder.build_load(*ptr, "")
            }
            ExprKind::PrefixOp(_, _) => todo!(),
            ExprKind::BinOp(_, _, _) => todo!(),
            ExprKind::Cast(expr, _) => todo!("conversion operation"),
            ExprKind::Group(inner) => self.emit_expr(inner),
            ExprKind::Field(_, _, _, _) => todo!(),
            ExprKind::Call(_, _) => todo!(),
            ExprKind::Index(_, _, _) => todo!(),
            ExprKind::Range(_, _) => todo!(),
            ExprKind::Block(stmts) => self.emit_block(stmts),
            ExprKind::Struct(name, inits) => {
                let info = *self.struct_infos.get(name.item()).expect("die!!!");
                let mut struct_value = self
                    .module
                    .get_struct_type(name.item())
                    .expect("ICE: struct ll ty should be present")
                    .get_undef();
                for (init_name, expr) in inits.iter() {
                    let value = self.emit_expr(expr);
                    let i = info.member_index(init_name.item());
                    struct_value = self
                        .builder
                        .build_insert_value(struct_value, value, i as u32, "")
                        .expect("struct index oob")
                        .into_struct_value();
                }
                struct_value.into()
            }
            ExprKind::If(_, _) => todo!(),
        }
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn alloc_stack_slot(
        &self,
        name: &str,
        ty: inkwell::types::BasicTypeEnum<'ctx>,
    ) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();

        let entry = self.curr_fn().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(ty, "")
    }

    fn ty_to_ll_ty(&self, ty: tyck::Type) -> inkwell::types::BasicTypeEnum<'ctx> {
        let context = self.context;
        match ty {
            tyck::Type::Bool => context.bool_type().into(),
            tyck::Type::Int => context.i32_type().into(),
            tyck::Type::Float => context.f32_type().into(),
            tyck::Type::Char => context.i8_type().into(),
            tyck::Type::Str => todo!(),
            tyck::Type::Struct(name) => self
                .module
                .get_struct_type(name)
                .expect("ICE: struct ll ty should be present")
                .as_basic_type_enum(),
            tyck::Type::Pointer(_, _) => todo!(),
            tyck::Type::Slice(_) => todo!(),
            tyck::Type::Array(_, _) => todo!(),
            tyck::Type::Tuple(tys) => {
                let tys: Vec<_> = tys.iter().map(|ty| self.ty_to_ll_ty(*ty)).collect();
                context.struct_type(tys.as_slice(), false).into()
            }
            tyck::Type::Fn(_, _) => todo!(),
        }
    }
}
