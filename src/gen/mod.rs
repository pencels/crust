use inkwell::types::{BasicTypeEnum, StructType};
use lazy_static::lazy_static;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

use inkwell::values::{BasicValue, CallableValue, IntValue};
use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    types::{BasicType, StringRadix},
    values::{BasicValueEnum, FunctionValue, PointerValue},
};
use inkwell::{AddressSpace, IntPredicate};

use crate::parser::ast::{BinOpKind, Operator, PrefixOpKind, Spanned};
use crate::tyck::{
    Type, VarId, RANGE_FROM_TY_NAME, RANGE_FULL_TY_NAME, RANGE_TO_TY_NAME, RANGE_TY_NAME,
};
use crate::{
    parser::ast::{DeclInfo, Defn, DefnKind, Expr, ExprKind, Stmt, StmtKind},
    tyck::StructInfo,
};
use tempfile::tempdir;

lazy_static! {
    static ref STDLIB_PATH: &'static OsStr = OsStr::new("lib/lib.c");
}

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

    pub fn emit_program(
        module_name: &str,
        output_filename: Option<&Path>,
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
        emitter.emit_fn_decls(program);

        for defn in program {
            emitter.emit_defn(defn);
        }

        let dir = tempdir().unwrap();

        let ll_file = dir
            .path()
            .join(&format!("{}.ll", module_name))
            .into_os_string();
        module.print_to_file(&ll_file).unwrap();
        module.print_to_file("a.ll").unwrap();

        let ll_obj_file = dir
            .path()
            .join(&format!("{}.o", module_name))
            .into_os_string();
        let mut compile_ll_cmd = Command::new("llc");
        compile_ll_cmd.args([
            &ll_file,
            OsStr::new("-filetype=obj"),
            OsStr::new("-relocation-model=pic"),
            OsStr::new("-o"),
            &ll_obj_file,
        ]);
        let status = compile_ll_cmd.status().expect("llc command failed to run");
        if !status.success() {
            panic!("aaaa llc returned non-zero exit status");
        }

        let mut compile_cmd = Command::new("clang");
        compile_cmd.args([
            *STDLIB_PATH,
            &ll_obj_file,
            OsStr::new("-o"),
            OsStr::new(module_name),
        ]);
        compile_cmd.status().unwrap();
        let status = compile_cmd.status().expect("clang command failed to run");
        if !status.success() {
            panic!("aaaa clang returned non-zero exit status");
        }
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
                .map(|decl| self.ty_to_ll_ty(decl.ty.get().expect("type")))
                .collect();

            ty.set_body(field_types.as_slice(), false);
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
                    let return_ty = return_ty
                        .get()
                        .expect("ICE: return ty should be set in tyck phase");
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
                    self.module.add_function(
                        decl.name.item(),
                        self.ty_to_ll_ty(return_ty).fn_type(&mut param_types, false),
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
            DefnKind::ExternFn { .. } => {
                // fn already declared, no body to emit
            }
        }
    }

    fn emit_fn(&mut self, decl: &DeclInfo, params: &[DeclInfo], return_ty: Type, body: &Expr) {
        let function = self
            .module
            .get_function(decl.name.item())
            .expect("function is declared");
        self.opt_fn = Some(function);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        for (i, param) in function.get_param_iter().enumerate() {
            let slot = self.alloc_stack_slot(param.get_type());

            self.builder.build_store(slot, param);

            self.variables.insert(params[i].id.get().expect("id"), slot);
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
                let slot = self.alloc_stack_slot(ty);
                self.builder.build_store(slot, value);
                self.variables.insert(decl.id.get().expect("id"), slot);
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
            ExprKind::Str(s) => {
                let global = unsafe { self.builder.build_global_string(s, "") };
                let ptr = self.builder.build_pointer_cast(
                    global.as_pointer_value(),
                    self.unit_ty().ptr_type(AddressSpace::Generic),
                    "",
                );
                self.build_str_slice(ptr, s.as_bytes().len())
            }
            ExprKind::Char(c) => self
                .context
                .i8_type()
                .const_int(c.bytes().nth(0).unwrap() as u64, false)
                .into(),
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
            ExprKind::Array(exprs, size) => {
                let ty = exprs[0].ty.get().expect("expr doesn't have ty");
                let ll_ty = self.ty_to_ll_ty(ty);
                let size = size.map(|s| *s.item()).unwrap_or_else(|| exprs.len());
                let arr_ty = ll_ty.array_type(size as u32);

                let values: Vec<_> = exprs.iter().map(|expr| self.emit_expr(expr)).collect();

                let mut arr = arr_ty.get_undef();
                for (i, value) in values.iter().cycle().take(size).enumerate() {
                    arr = self
                        .builder
                        .build_insert_value(arr, *value, i as u32, "")
                        .expect("index oob")
                        .into_array_value();
                }

                arr.as_basic_value_enum()
            }
            ExprKind::Id(name, id, is_func) => {
                if is_func.get() {
                    self.module
                        .get_function(name)
                        .expect("functions are forward declared")
                        .as_global_value()
                        .as_basic_value_enum()
                } else {
                    let ptr = self
                        .variables
                        .get(&id.get().expect("id"))
                        .expect("ICE: undefined variable");
                    self.builder.build_load(*ptr, "")
                }
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Amp | PrefixOpKind::AmpMut), expr) => {
                match &expr.kind {
                    ExprKind::Index(lhs, rhs, num_derefs, autoderef_ty) => {
                        let (arr_ptr, _) = self.emit_indexable_ty_parts(
                            lhs,
                            num_derefs.get(),
                            autoderef_ty.get().unwrap(),
                        );
                        let rhs_value = self.emit_expr(rhs);
                        let elem_ptr = unsafe {
                            self.builder
                                .build_gep(arr_ptr, &[rhs_value.into_int_value()], "")
                        };
                        elem_ptr.as_basic_value_enum()
                    }
                    _ => self.emit_place_expr(expr).into(),
                }
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Star), expr) => {
                let ptr = self.emit_expr(expr).into_pointer_value();
                self.builder.build_load(ptr, "")
            }
            ExprKind::PrefixOp(_, _) => todo!(),
            ExprKind::BinOp(Spanned(_, Operator::Simple(op)), lhs, rhs) => {
                self.emit_binop_expr(*op, lhs, rhs)
            }
            ExprKind::BinOp(Spanned(_, Operator::Assign(op)), lhs, rhs) => {
                self.emit_assign_expr(*op, lhs, rhs)
            }
            ExprKind::Cast(castee, _) => {
                let value = self.emit_expr(castee);
                match (
                    castee.ty.get().expect("castee ty not resolved"),
                    expr.ty.get().expect("expr ty not resolved"),
                ) {
                    (Type::Pointer(_, Type::Slice(_)), Type::Pointer(_, Type::Slice(_))) => {
                        // Slices have the same layout, can use the value itself
                        value
                    }
                    (Type::Pointer(_, Type::Slice(_)), Type::Pointer(_, ptr_ty)) => {
                        let ptr = self
                            .builder
                            .build_extract_value(value.into_struct_value(), 0, "")
                            .expect("index oob")
                            .into_pointer_value();
                        self.builder
                            .build_pointer_cast(
                                ptr,
                                self.ty_to_ll_ty(*ptr_ty).ptr_type(AddressSpace::Generic),
                                "",
                            )
                            .as_basic_value_enum()
                    }
                    (Type::Pointer(_, _), Type::Pointer(_, ptr_ty)) => self
                        .builder
                        .build_pointer_cast(
                            value.into_pointer_value(),
                            self.ty_to_ll_ty(*ptr_ty).ptr_type(AddressSpace::Generic),
                            "",
                        )
                        .as_basic_value_enum(),
                    (Type::Int, Type::Float) => self
                        .builder
                        .build_signed_int_to_float(
                            value.into_int_value(),
                            self.context.f32_type(),
                            "",
                        )
                        .as_basic_value_enum(),
                    _ => todo!("perform conversion"),
                }
            }
            ExprKind::Group(inner) => self.emit_expr(inner),
            ExprKind::Field(lhs, _, field, num_derefs) => {
                let mut value = self.emit_expr(lhs);
                for _ in 0..num_derefs.get() {
                    value = self.builder.build_load(value.into_pointer_value(), "");
                }
                let i = field.item().get_index();
                self.builder
                    .build_extract_value(value.into_struct_value(), i as u32, "")
                    .expect("index oob")
            }
            ExprKind::Call(callee, args) => {
                let callee_value = self.emit_expr(callee);
                let args: Vec<_> = args.iter().map(|arg| self.emit_expr(arg)).collect();
                self.builder
                    .build_call(
                        CallableValue::try_from(callee_value.into_pointer_value()).unwrap(),
                        &args,
                        "",
                    )
                    .try_as_basic_value()
                    .unwrap_left()
            }
            ExprKind::Index(lhs, rhs, num_derefs, autoderef_ty) => {
                let (arr_ptr, _) = self.emit_indexable_ty_parts(
                    lhs,
                    num_derefs.get(),
                    autoderef_ty.get().unwrap(),
                );
                let rhs_value = self.emit_expr(rhs);
                let elem_ptr = unsafe {
                    self.builder
                        .build_gep(arr_ptr, &[rhs_value.into_int_value()], "")
                };
                self.builder.build_load(elem_ptr, "").as_basic_value_enum()
            }
            ExprKind::Range(start, end) => match (start, end) {
                (Some(start), Some(end)) => {
                    self.build_struct(RANGE_TY_NAME, &[("start", start), ("end", end)])
                }
                (None, Some(end)) => self.build_struct(RANGE_TO_TY_NAME, &[("end", end)]),
                (Some(start), None) => self.build_struct(RANGE_FROM_TY_NAME, &[("start", start)]),
                _ => self.build_struct(RANGE_FULL_TY_NAME, &[]),
            },
            ExprKind::Block(stmts) => self.emit_block(stmts),
            ExprKind::Struct(name, inits) => {
                let name = name.item();
                let inits: Vec<_> = inits
                    .iter()
                    .map(|(field, expr)| (*field.item(), expr))
                    .collect();
                self.build_struct(name, inits.as_slice())
            }
            ExprKind::If(thens, els) => {
                let mut branches = Vec::new();
                for (cond, then) in *thens {
                    let cond_value = self.emit_expr(cond);
                    let then_block = self.context.append_basic_block(self.curr_fn(), "");
                    let else_block = self.context.append_basic_block(self.curr_fn(), "");
                    self.builder.build_conditional_branch(
                        cond_value.into_int_value(),
                        then_block,
                        else_block,
                    );

                    self.builder.position_at_end(then_block);
                    let then_value = self.emit_expr(then);
                    branches.push((then_block, then_value));

                    self.builder.position_at_end(else_block);
                }

                let else_value = if let Some(expr) = els {
                    self.emit_expr(expr)
                } else {
                    self.unit_value()
                };

                let else_block = self
                    .builder
                    .get_insert_block()
                    .expect("should be pointing at a block");
                branches.push((else_block, else_value));

                let after_block = self.context.append_basic_block(self.curr_fn(), "");
                self.builder.position_at_end(after_block);
                let phi = self
                    .builder
                    .build_phi(self.ty_to_ll_ty(expr.ty.get().expect("type")), "");

                for (block, value) in &branches {
                    self.builder.position_at_end(*block);
                    self.builder.build_unconditional_branch(after_block);
                    phi.add_incoming(&[(value as &dyn BasicValue, *block)]);
                }

                self.builder.position_at_end(after_block);

                phi.as_basic_value()
            }
        }
    }

    /// Emits the pointer or slice pointer to the given expr. Essentially, emits the `&(expr)`.
    fn emit_place_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        match &expr.kind {
            ExprKind::Id(_, id, _) => {
                let id = id.get().expect("variable not given id");
                self.variables
                    .get(&id)
                    .expect("variable id not registered")
                    .as_basic_value_enum()
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Star), expr) => self.emit_expr(expr),
            ExprKind::Group(expr) => self.emit_place_expr(expr),
            ExprKind::Field(lhs, _, Spanned(_, field), num_derefs) => {
                let num_derefs = num_derefs.get();
                let mut lhs_ptr = self.emit_place_expr(lhs).into_pointer_value();
                for _ in 0..num_derefs {
                    lhs_ptr = self.builder.build_load(lhs_ptr, "").into_pointer_value();
                }
                self.builder
                    .build_struct_gep(lhs_ptr, field.get_index() as u32, "")
                    .expect("index oob")
                    .as_basic_value_enum()
            }
            ExprKind::Index(lhs, rhs, num_derefs, autoderef_ty) => {
                let (mut ptr, mut size) = self.emit_indexable_ty_parts(
                    lhs,
                    num_derefs.get(),
                    autoderef_ty.get().expect("autoderef ty not resolved"),
                );
                let rhs_value = self.emit_expr(rhs);

                match rhs.ty.get().expect("expr ty not resolved") {
                    Type::Int => unsafe {
                        self.builder
                            .build_gep(ptr, &[rhs_value.into_int_value()], "")
                            .as_basic_value_enum()
                    },
                    Type::Struct(struct_name) => {
                        match struct_name {
                            RANGE_FROM_TY_NAME => {
                                let index = self
                                    .builder
                                    .build_extract_value(rhs_value.into_struct_value(), 0, "")
                                    .expect("index oob")
                                    .into_int_value();
                                ptr = unsafe { self.builder.build_gep(ptr, &[index], "") };
                                size = self.builder.build_int_sub(size, index, "");
                            }
                            RANGE_TO_TY_NAME => {
                                let index = self
                                    .builder
                                    .build_extract_value(rhs_value.into_struct_value(), 0, "")
                                    .expect("index oob")
                                    .into_int_value();
                                size = index;
                            }
                            RANGE_TY_NAME => {
                                let rhs_value = rhs_value.into_struct_value();
                                let start_index = self
                                    .builder
                                    .build_extract_value(rhs_value, 0, "")
                                    .expect("index oob")
                                    .into_int_value();
                                let end_index = self
                                    .builder
                                    .build_extract_value(rhs_value, 1, "")
                                    .expect("index oob")
                                    .into_int_value();
                                ptr = unsafe { self.builder.build_gep(ptr, &[start_index], "") };
                                size = self.builder.build_int_sub(end_index, start_index, "");
                            }
                            RANGE_FULL_TY_NAME => {}
                            _ => unreachable!("tyck should filter out non-range struct indices"),
                        }

                        let ptr = self.builder.build_pointer_cast(
                            ptr,
                            self.unit_ty().ptr_type(AddressSpace::Generic),
                            "",
                        );

                        let mut slice = self.generic_slice_ty().get_undef();
                        slice = self
                            .builder
                            .build_insert_value(slice, ptr, 0, "")
                            .expect("index oob")
                            .into_struct_value();
                        slice = self
                            .builder
                            .build_insert_value(slice, size, 1, "")
                            .expect("index oob")
                            .into_struct_value();
                        slice.as_basic_value_enum()
                    }
                    _ => panic!("rhs type should be index"),
                }
            }
            _ => panic!("temporary promotions are not supported"),
        }
    }

    /// Emits the ref of an expr that can occur on the lhs of an index lhs[rhs].
    fn emit_indexable_ref_expr(&mut self, expr: &Expr) -> BasicValueEnum<'ctx> {
        match &expr.kind {
            ExprKind::Id(_, id, _) => {
                let id = id.get().expect("variable not given id");
                self.variables
                    .get(&id)
                    .expect("variable id not registered")
                    .as_basic_value_enum()
            }
            ExprKind::Group(expr) => self.emit_indexable_ref_expr(expr),
            ExprKind::Cast(expr, _) => self.emit_indexable_ref_expr(expr),
            _ => self.emit_expr(expr),
        }
    }

    /// Emits a typed pointer to array/slice data and the length of that data.
    fn emit_indexable_ty_parts(
        &mut self,
        lhs: &Expr,
        num_derefs: usize,
        autoderef_ty: Type,
    ) -> (PointerValue<'ctx>, IntValue<'ctx>) {
        // TODO: revise so that lhs_value can either be a simple *T or a fat pointer.
        // Logic below will make the appropriate casts depending on the autoderef_ty
        let lhs_value = if num_derefs > 0 {
            let mut ptr = self.emit_expr(lhs).into_pointer_value();
            for _ in 0..num_derefs {
                ptr = self.builder.build_load(ptr, "").into_pointer_value();
            }
            ptr.as_basic_value_enum()
        } else {
            self.emit_expr(lhs)
        };

        println!("{:#?}", lhs);
        println!("{:#?}", lhs_value);
        println!("{:?}", autoderef_ty);

        match autoderef_ty {
            Type::Array(ty, size) => (
                self.builder.build_pointer_cast(
                    lhs_value.into_pointer_value(),
                    self.ty_to_ll_ty(*ty).ptr_type(AddressSpace::Generic),
                    "",
                ),
                self.context.i32_type().const_int(size as u64, false).into(),
            ),
            Type::Pointer(_, &Type::Array(ty, size)) => {
                let ptr = self
                    .builder
                    .build_load(lhs_value.into_pointer_value(), "")
                    .into_pointer_value();
                (
                    self.builder.build_pointer_cast(
                        ptr,
                        self.ty_to_ll_ty(*ty).ptr_type(AddressSpace::Generic),
                        "",
                    ),
                    self.context.i32_type().const_int(size as u64, false).into(),
                )
            }
            Type::Pointer(_, &Type::Slice(ty)) => (
                self.builder.build_pointer_cast(
                    self.builder
                        .build_extract_value(lhs_value.into_struct_value(), 0, "")
                        .expect("index oob")
                        .into_pointer_value(),
                    self.ty_to_ll_ty(*ty).ptr_type(AddressSpace::Generic),
                    "",
                ),
                self.builder
                    .build_extract_value(lhs_value.into_struct_value(), 1, "")
                    .expect("index oob")
                    .into_int_value(),
            ),
            Type::Pointer(_, Type::Str) => (
                self.builder.build_pointer_cast(
                    self.builder
                        .build_extract_value(lhs_value.into_struct_value(), 0, "")
                        .expect("index oob")
                        .into_pointer_value(),
                    self.ty_to_ll_ty(Type::Char).ptr_type(AddressSpace::Generic),
                    "",
                ),
                self.builder
                    .build_extract_value(lhs_value.into_struct_value(), 1, "")
                    .expect("index oob")
                    .into_int_value(),
            ),
            _ => panic!("lhs type should be indexable"),
        }
    }

    fn emit_binop_expr(&mut self, op: BinOpKind, lhs: &Expr, rhs: &Expr) -> BasicValueEnum<'ctx> {
        match op {
            BinOpKind::AmpAmp | BinOpKind::PipePipe => {
                let lhs_block = self
                    .builder
                    .get_insert_block()
                    .expect("builder was not pointing to a block");

                let lhs_value = self.emit_expr(lhs);
                let rhs_block = self.context.append_basic_block(self.curr_fn(), "rhs");
                let end_block = self.context.append_basic_block(self.curr_fn(), "end");
                match op {
                    BinOpKind::AmpAmp => {
                        self.builder.build_conditional_branch(
                            lhs_value.into_int_value(),
                            rhs_block,
                            end_block,
                        );
                    }
                    BinOpKind::PipePipe => {
                        self.builder.build_conditional_branch(
                            lhs_value.into_int_value(),
                            end_block,
                            rhs_block,
                        );
                    }
                    _ => unreachable!(),
                }

                self.builder.position_at_end(rhs_block);
                let rhs_value = self.emit_expr(rhs);
                self.builder.build_unconditional_branch(end_block);
                let final_rhs_block = self
                    .builder
                    .get_insert_block()
                    .expect("builder was not pointing to a block");

                self.builder.position_at_end(end_block);
                let phi_value = self.builder.build_phi(self.ty_to_ll_ty(Type::Bool), "");

                phi_value.add_incoming(&[
                    (&lhs_value as &dyn BasicValue, lhs_block),
                    (&rhs_value as &dyn BasicValue, final_rhs_block),
                ]);

                phi_value.as_basic_value()
            }
            _ => {
                let lhs_value = self.emit_expr(lhs);
                let rhs_value = self.emit_expr(rhs);
                let lhs_ty = lhs.ty.get().expect("type not resolved");
                let rhs_ty = rhs.ty.get().expect("type not resolved");
                if lhs_ty != rhs_ty {
                    panic!("types must be equal on both sides of a binop");
                }
                self.emit_binop_from_values(op, lhs_ty, lhs_value, rhs_value)
            }
        }
    }

    fn emit_binop_from_values(
        &mut self,
        op: BinOpKind,
        ty: Type,
        lhs_value: BasicValueEnum<'ctx>,
        rhs_value: BasicValueEnum<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        match op {
            BinOpKind::Plus => match ty {
                Type::Int => self
                    .builder
                    .build_int_add(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                Type::Float => self
                    .builder
                    .build_float_add(
                        lhs_value.into_float_value(),
                        rhs_value.into_float_value(),
                        "",
                    )
                    .into(),
                _ => unreachable!("can only add int or float"),
            },
            BinOpKind::Minus => match ty {
                Type::Int => self
                    .builder
                    .build_int_sub(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                Type::Float => self
                    .builder
                    .build_float_sub(
                        lhs_value.into_float_value(),
                        rhs_value.into_float_value(),
                        "",
                    )
                    .into(),
                _ => unreachable!("can only sub int or float"),
            },
            BinOpKind::Star => match ty {
                Type::Int => self
                    .builder
                    .build_int_mul(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                Type::Float => self
                    .builder
                    .build_float_mul(
                        lhs_value.into_float_value(),
                        rhs_value.into_float_value(),
                        "",
                    )
                    .into(),
                _ => unreachable!("can only mul int or float"),
            },
            BinOpKind::Slash => match ty {
                Type::Int => self
                    .builder
                    .build_int_signed_div(
                        lhs_value.into_int_value(),
                        rhs_value.into_int_value(),
                        "",
                    )
                    .into(),
                Type::Float => self
                    .builder
                    .build_float_div(
                        lhs_value.into_float_value(),
                        rhs_value.into_float_value(),
                        "",
                    )
                    .into(),
                _ => unreachable!("can only div int or float"),
            },
            BinOpKind::Percent => match ty {
                Type::Int => self
                    .builder
                    .build_int_signed_rem(
                        lhs_value.into_int_value(),
                        rhs_value.into_int_value(),
                        "",
                    )
                    .into(),
                Type::Float => self
                    .builder
                    .build_float_rem(
                        lhs_value.into_float_value(),
                        rhs_value.into_float_value(),
                        "",
                    )
                    .into(),
                _ => unreachable!("can only rem int or float"),
            },
            BinOpKind::Caret => match ty {
                Type::Int | Type::Bool => self
                    .builder
                    .build_xor(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                _ => unreachable!("can only xor int or bool"),
            },
            BinOpKind::Amp => match ty {
                Type::Int | Type::Bool => self
                    .builder
                    .build_and(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                _ => unreachable!("can only and int or bool"),
            },
            BinOpKind::Pipe => match ty {
                Type::Int | Type::Bool => self
                    .builder
                    .build_or(lhs_value.into_int_value(), rhs_value.into_int_value(), "")
                    .into(),
                _ => unreachable!("can only or int or bool"),
            },
            BinOpKind::AmpAmp => unreachable!("should have been caught earlier"),
            BinOpKind::PipePipe => unreachable!("should have been caught earlier"),
            BinOpKind::Lt => self
                .builder
                .build_int_compare(
                    IntPredicate::SLT,
                    lhs_value.into_int_value(),
                    rhs_value.into_int_value(),
                    "",
                )
                .into(),
            BinOpKind::LtLt => todo!(),
            BinOpKind::Le => self
                .builder
                .build_int_compare(
                    IntPredicate::SLE,
                    lhs_value.into_int_value(),
                    rhs_value.into_int_value(),
                    "",
                )
                .into(),
            BinOpKind::Gt => todo!(),
            BinOpKind::GtGt => todo!(),
            BinOpKind::Ge => todo!(),
            BinOpKind::EqEq => self
                .builder
                .build_int_compare(
                    IntPredicate::EQ,
                    lhs_value.into_int_value(),
                    rhs_value.into_int_value(),
                    "",
                )
                .into(),
            BinOpKind::Ne => todo!(),
        }
    }

    fn emit_assign_expr(
        &mut self,
        op: Option<BinOpKind>,
        lhs: &Expr,
        rhs: &Expr,
    ) -> BasicValueEnum<'ctx> {
        let rhs_value = self.emit_expr(rhs);
        let lhs_ptr = self.emit_place_expr(lhs).into_pointer_value();

        let lhs_ty = lhs.ty.get().expect("type not resolved");
        let rhs_ty = rhs.ty.get().expect("type not resolved");
        if lhs_ty != rhs_ty {
            panic!("types must be equal on both sides of an assignment");
        }

        let value = if let Some(op) = op {
            let lhs_value = self.builder.build_load(lhs_ptr, "");
            self.emit_binop_from_values(op, lhs_ty, lhs_value, rhs_value)
        } else {
            rhs_value
        };

        self.builder.build_store(lhs_ptr, value);
        self.unit_value()
    }

    fn build_struct(&mut self, name: &str, inits: &[(&str, &Expr)]) -> BasicValueEnum<'ctx> {
        let info = *self.struct_infos.get(name).expect("die!!!");
        let mut struct_value = self
            .module
            .get_struct_type(name)
            .expect("ICE: struct ll ty should be present")
            .get_undef();
        for (init_name, expr) in inits.iter() {
            let value = self.emit_expr(expr);
            let i = info.member_index(init_name);
            struct_value = self
                .builder
                .build_insert_value(struct_value, value, i as u32, "")
                .expect("struct index oob")
                .into_struct_value();
        }
        struct_value.into()
    }

    /// Creates a new stack allocation instruction in the entry block of the function.
    fn alloc_stack_slot(&self, ty: inkwell::types::BasicTypeEnum<'ctx>) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();

        let entry = self.curr_fn().get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }

        builder.build_alloca(ty, "")
    }

    fn unit_ty(&self) -> BasicTypeEnum<'ctx> {
        self.context.struct_type(&[], false).into()
    }

    fn generic_slice_ty(&self) -> StructType<'ctx> {
        self.context.struct_type(
            &[
                self.unit_ty().ptr_type(AddressSpace::Generic).into(),
                self.context.i32_type().into(),
            ],
            false,
        )
    }

    fn build_str_slice(&self, ptr: PointerValue<'ctx>, size: usize) -> BasicValueEnum<'ctx> {
        let mut slice = self.generic_slice_ty().get_undef();
        slice = self
            .builder
            .build_insert_value(slice, ptr, 0, "")
            .expect("index oob")
            .into_struct_value();
        slice = self
            .builder
            .build_insert_value(
                slice,
                self.context.i32_type().const_int(size as u64, false),
                1,
                "",
            )
            .expect("index oob")
            .into_struct_value();
        slice.as_basic_value_enum()
    }

    fn ty_to_ll_ty(&self, ty: Type) -> BasicTypeEnum<'ctx> {
        let context = self.context;
        match ty {
            Type::Bool => context.bool_type().into(),
            Type::Int => context.i32_type().into(),
            Type::Float => context.f32_type().into(),
            Type::Char => context.i8_type().into(),
            Type::Str => unreachable!("bare str cannot be repr'd as an llvm type"),
            Type::Struct(name) => self
                .module
                .get_struct_type(name)
                .expect("ICE: struct ll ty should be present")
                .as_basic_type_enum(),
            Type::Pointer(_, Type::Str) => self.generic_slice_ty().into(),
            Type::Pointer(_, Type::Slice(inner)) => self.generic_slice_ty().into(),
            Type::Pointer(_, inner) => self
                .ty_to_ll_ty(*inner)
                .ptr_type(AddressSpace::Generic)
                .into(),
            Type::Slice(_) => unreachable!("bare slice cannot be repr'd as an llvm type"),
            Type::Array(ty, size) => self.ty_to_ll_ty(*ty).array_type(size as u32).into(),
            Type::Tuple(tys) => {
                let tys: Vec<_> = tys.iter().map(|ty| self.ty_to_ll_ty(*ty)).collect();
                context.struct_type(tys.as_slice(), false).into()
            }
            Type::Fn(params, return_type) => {
                let ll_return_type = self.ty_to_ll_ty(*return_type);
                let ll_param_types: Vec<_> = params.iter().map(|p| self.ty_to_ll_ty(*p)).collect();
                ll_return_type
                    .fn_type(&ll_param_types, false)
                    .ptr_type(AddressSpace::Generic)
                    .into()
            }
        }
    }
}
