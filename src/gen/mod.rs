use std::{collections::HashMap, path::PathBuf};

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicType, StringRadix},
    values::{BasicValue, BasicValueEnum, FunctionValue, PointerValue},
};

use crate::{
    parser::ast::{DeclInfo, Defn, DefnKind, Expr, ExprKind, Stmt, StmtKind, Type},
    tyck,
};

pub struct Emitter<'a, 'ctx> {
    pub context: &'ctx Context,
    pub builder: &'a Builder<'ctx>,
    pub module: &'a Module<'ctx>,
    pub opt_fn: Option<FunctionValue<'ctx>>,

    variables: HashMap<String, PointerValue<'ctx>>,
}

impl<'a, 'ctx> Emitter<'a, 'ctx> {
    #[inline]
    pub fn curr_fn(&self) -> FunctionValue<'ctx> {
        self.opt_fn.unwrap()
    }

    #[inline]
    fn unit_value(&self) -> BasicValueEnum<'ctx> {
        self.context.bool_type().const_zero().into()
    }

    pub fn emit_program(module_name: &str, program: &[Defn]) {
        let context = &Context::create();
        let builder = &context.create_builder();
        let module = &context.create_module(module_name);

        let mut emitter = Emitter {
            context,
            builder,
            module,
            opt_fn: None,
            variables: HashMap::new(),
        };

        for defn in program {
            emitter.emit_defn(defn);
        }

        let mut out_path = PathBuf::from(module_name);
        out_path.set_extension("ll");
        module.print_to_file(out_path).unwrap();
    }

    pub fn emit_defn(&mut self, defn: &Defn) {
        match &defn.kind {
            DefnKind::Struct { name, members } => {}
            DefnKind::Fn {
                decl,
                params,
                return_type,
                body,
            } => {
                self.emit_fn(decl, params, return_type, body);
            }
            DefnKind::Static { decl, expr } => todo!(),
        }
    }

    fn emit_fn(
        &mut self,
        decl: &DeclInfo,
        params: &[DeclInfo],
        return_type: &Option<Type>,
        body: &Expr,
    ) {
        let mut param_types: Vec<_> = params
            .iter()
            .map(|param| {
                let param_ty = param
                    .ty
                    .get()
                    .expect("ICE: type not resolved in tyck phase");
                ty_to_ll_ty(self.context, param_ty)
            })
            .collect();
        let function = self.module.add_function(
            decl.name.item(),
            ty_to_ll_ty(self.context, tyck::Type::Int).fn_type(&mut param_types, false),
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
                let ty = ty_to_ll_ty(
                    self.context,
                    decl.ty.get().expect("ICE: decl doesn't have ty"),
                );
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
            ExprKind::Bool(b) => self
                .context
                .bool_type()
                .const_int(if *b { 1 } else { 0 }, false)
                .into(),
            ExprKind::Int(i) => self
                .context
                .i32_type()
                .const_int_from_string(i, StringRadix::Decimal)
                .unwrap()
                .into(),
            ExprKind::Float(f) => self.context.f32_type().const_float_from_string(f).into(),
            ExprKind::Str(_) => todo!(),
            ExprKind::Char(c) => todo!(),
            ExprKind::Tuple(_) => todo!(),
            ExprKind::Array(_, _) => todo!(),
            ExprKind::Id(name, id) => {
                let ptr = self.variables.get(*name).expect("ICE: undefined variable");
                self.builder.build_load(*ptr, name)
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
            ExprKind::Struct(_, _) => todo!(),
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

        builder.build_alloca(ty, name)
    }
}

fn ty_to_ll_ty<'ctx>(
    context: &'ctx Context,
    ty: tyck::Type,
) -> inkwell::types::BasicTypeEnum<'ctx> {
    match ty {
        tyck::Type::Bool => context.bool_type().into(),
        tyck::Type::Int => context.i32_type().into(),
        tyck::Type::Float => context.f32_type().into(),
        tyck::Type::Char => context.i8_type().into(),
        tyck::Type::Str => todo!(),
        tyck::Type::Struct(_) => todo!(),
        tyck::Type::Pointer(_, _) => todo!(),
        tyck::Type::Slice(_) => todo!(),
        tyck::Type::Array(_, _) => todo!(),
        tyck::Type::Tuple(_) => todo!(),
        tyck::Type::Fn(_, _) => todo!(),
    }
}
