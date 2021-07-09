mod result;

use result::{TyckError, TyckResult};

use bumpalo::Bump;
use std::cell::Cell;
use std::{collections::HashMap, fmt::Display};

use std::str::FromStr;

use crate::parser::ast::{DeclInfo, Operator};
use crate::{
    parser::ast::{BinOpKind, ExprKind, Field, PrefixOpKind, Spanned, Stmt, StmtKind},
    util::Span,
};

use maplit::hashmap;

use crate::{
    parser::ast::{self, Defn, DefnKind, Expr, TypeKind},
    util::fresh_id,
};

pub const RANGE_TY_NAME: &str = "Range";
pub const RANGE_TO_TY_NAME: &str = "RangeTo";
pub const RANGE_FROM_TY_NAME: &str = "RangeFrom";
pub const RANGE_FULL_TY_NAME: &str = "RangeFull";

enum Source {
    Id(Span),
    Ptr(Span),
}

#[derive(Debug, Clone, Copy)]
pub struct StructInfo<'a> {
    pub name: &'a Spanned<&'a str>,
    pub members: &'a [&'a DeclInfo<'a>],
}

impl<'a> StructInfo<'a> {
    pub fn member_index(&self, name: &str) -> usize {
        self.members
            .iter()
            .enumerate()
            .find_map(|(i, member)| {
                if *member.name.item() == name {
                    Some(i)
                } else {
                    None
                }
            })
            .expect("ICE: tried to find index of a struct member but died trying")
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Type<'a> {
    Bool,
    Int,
    Float,
    Char,
    Str,
    Struct(&'a str),
    Pointer(bool, &'a Type<'a>),
    Slice(&'a Type<'a>),
    Array(&'a Type<'a>, usize),
    Tuple(&'a [Type<'a>]),
    Fn(&'a [Type<'a>], &'a Type<'a>),
}

impl<'a> Display for Type<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Bool => write!(f, "bool"),
            Type::Int => write!(f, "int"),
            Type::Float => write!(f, "float"),
            Type::Char => write!(f, "char"),
            Type::Str => write!(f, "str"),
            Type::Struct(s) => write!(f, "{}", s),
            Type::Pointer(m, ty) => {
                write!(f, "*")?;
                if *m {
                    write!(f, "mut ")?;
                }
                write!(f, "{}", ty)
            }
            Type::Slice(ty) => write!(f, "[{}]", ty),
            Type::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            Type::Tuple([]) => write!(f, "()"),
            Type::Tuple([ty]) => write!(f, "({},)", ty),
            Type::Tuple([ty, tys @ ..]) => {
                write!(f, "({}", ty)?;
                for ty in tys {
                    write!(f, ", {}", ty)?;
                }
                write!(f, ")")
            }
            Type::Fn([], rty) => write!(f, "fn() -> {}", rty),
            Type::Fn([ty, tys @ ..], rty) => {
                write!(f, "fn({}", ty)?;
                for ty in tys {
                    write!(f, ", {}", ty)?;
                }
                write!(f, ") -> {}", rty)
            }
        }
    }
}

fn human_type_name(ty: &Type) -> String {
    match ty {
        Type::Struct(ty) => format!("struct '{}'", ty),
        _ => format!("'{}'", ty),
    }
}

impl<'a> Type<'a> {
    pub fn unit() -> Type<'a> {
        Type::Tuple(&[])
    }

    pub fn from<'b>(checker: &TypeChecker<'b>, ty: &ast::Type) -> TyckResult<Type<'b>> {
        Ok(match &ty.kind {
            TypeKind::Id("bool") => Type::Bool,
            TypeKind::Id("int") => Type::Int,
            TypeKind::Id("float") => Type::Float,
            TypeKind::Id("char") => Type::Char,
            TypeKind::Id("str") => Type::Str,
            TypeKind::Id(other) => {
                if checker.struct_tys.contains_key(other) {
                    Type::Struct(checker.bump.alloc_str(other))
                } else {
                    return Err(TyckError::UndefinedType {
                        name: other.to_string(),
                        span: ty.span,
                    });
                }
            }
            TypeKind::Pointer(m, inner) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                Type::Pointer(*m, inner)
            }
            TypeKind::Slice(inner) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                Type::Slice(inner)
            }
            TypeKind::Array(inner, size) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                Type::Array(inner, *size)
            }
            TypeKind::Tuple([]) => Type::Tuple(&[]),
            TypeKind::Tuple(inners) => {
                let inners: TyckResult<Vec<_>> = inners
                    .iter()
                    .map(|inner| Type::from(checker, inner))
                    .collect();
                let inners = checker.bump.alloc_slice_fill_iter(inners?);
                Type::Tuple(inners)
            }
            TypeKind::Fn(params, return_ty) => {
                let params: TyckResult<Vec<_>> = params
                    .iter()
                    .map(|param| Type::from(checker, param))
                    .collect();
                let params = checker.bump.alloc_slice_fill_iter(params?);
                let return_ty = checker
                    .bump
                    .alloc(return_ty.map_or(Ok(Type::Tuple(&[])), |t| Type::from(checker, &t))?);
                Type::Fn(params, return_ty)
            }
        })
    }
}

pub struct Env<'alloc> {
    scopes: Vec<HashMap<&'alloc str, VarId>>,
}

impl<'alloc> Env<'alloc> {
    pub fn new() -> Env<'alloc> {
        Env {
            scopes: vec![hashmap! {}],
        }
    }

    pub fn push(&mut self) {
        self.scopes.push(hashmap! {});
    }

    pub fn pop(&mut self) {
        self.scopes.pop();
    }

    pub fn set(&mut self, name: &'alloc str, id: VarId) {
        self.scopes
            .last_mut()
            .expect("ICE: setting when no scope was opened")
            .insert(name, id);
    }

    pub fn get(&self, name: &str) -> Option<VarId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
            .map(|x| *x)
    }
}

pub type VarId = usize;

fn tyck_assign(
    Spanned(lhs_ty_span, lhs_ty): Spanned<&Type>,
    Spanned(rhs_ty_span, rhs_ty): Spanned<&Type>,
) -> TyckResult<()> {
    if is_assignable(lhs_ty, rhs_ty) {
        // Catch unsized types.
        if !is_sized(rhs_ty) {
            return Err(TyckError::CannotAssignUnsized {
                span: rhs_ty_span,
                ty_name: human_type_name(&rhs_ty),
            });
        }
        if !is_sized(lhs_ty) {
            return Err(TyckError::CannotAssignUnsized {
                span: lhs_ty_span,
                ty_name: human_type_name(&lhs_ty),
            });
        }

        Ok(())
    } else {
        Err(TyckError::MismatchedTypes {
            expected_ty: human_type_name(lhs_ty),
            got: rhs_ty_span,
            got_ty: human_type_name(rhs_ty),
        })
    }
}

fn indexed_ty<'alloc>(
    lhs_ty: Type<'alloc>,
    lhs_span: Span,
    index_ty: Type,
    index_span: Span,
) -> TyckResult<Type<'alloc>> {
    let elem_ty = match lhs_ty {
        Type::Slice(ty) => ty,
        Type::Array(ty, _) => ty,
        _ => {
            return Err(TyckError::TypeNotIndexable {
                span: lhs_span,
                ty_name: human_type_name(&lhs_ty),
            })
        }
    };
    let place_ty = match index_ty {
        Type::Int => *elem_ty,
        Type::Struct(
            RANGE_TY_NAME | RANGE_FROM_TY_NAME | RANGE_TO_TY_NAME | RANGE_FULL_TY_NAME,
        ) => Type::Slice(elem_ty),
        _ => {
            return Err(TyckError::TypeCannotBeUsedAsAnIndex {
                span: index_span,
                ty_name: human_type_name(&index_ty),
            })
        }
    };
    Ok(place_ty)
}

fn is_comparable(ty: &Type) -> bool {
    match ty {
        Type::Int | Type::Float | Type::Char => true,
        _ => false,
    }
}

fn tyck_is_of_type(Spanned(ty_span, ty): Spanned<&Type>, tys: &[Type]) -> TyckResult<()> {
    if tys.contains(ty) {
        Ok(())
    } else {
        if tys.is_empty() {
            panic!("ICE: tys should not be empty");
        }

        let alts = tys
            .iter()
            .map(|ty| human_type_name(ty))
            .reduce(|acc, ty| acc + &format!(" or {}", ty))
            .unwrap();

        Err(TyckError::MismatchedTypes {
            expected_ty: alts,
            got: ty_span,
            got_ty: human_type_name(ty),
        })
    }
}

fn is_sized(ty: &Type) -> bool {
    match ty {
        Type::Slice(_) => false,
        _ => true,
    }
}

fn is_assignable<'a>(assignee: &'a Type<'a>, ty: &'a Type<'a>) -> bool {
    match (assignee, ty) {
        (Type::Pointer(m1, t1), Type::Pointer(m2, t2)) => {
            // if assignee pointer is non-mut, then any mutability pointer can be assigned,
            // otherwise the mutability of the RHS determines assignability
            (!m1 || *m2) && is_assignable(t1, t2)
        }
        (Type::Slice(t1), Type::Slice(t2)) => is_assignable(t1, t2),
        (Type::Array(t1, n1), Type::Array(t2, n2)) => n1 == n2 && is_assignable(t1, t2),
        (Type::Tuple(ts1), Type::Tuple(ts2)) => {
            ts1.iter().zip(*ts2).all(|(t1, t2)| is_assignable(t1, t2))
        }
        (Type::Fn(p1, r1), Type::Fn(p2, r2)) => {
            p1.iter().zip(*p2).all(|(p1, p2)| is_assignable(p2, p1)) && is_assignable(r1, r2)
        }
        _ => assignee == ty,
    }
}

fn deref_until<'alloc, T>(
    mut ty: Type<'alloc>,
    num_derefs: &Cell<usize>,
    cond: impl FnOnce(Type<'alloc>) -> TyckResult<T>,
) -> TyckResult<T> {
    while let Type::Pointer(_, inner) = ty {
        num_derefs.set(num_derefs.get() + 1);
        ty = *inner;
    }
    cond(ty)
}

fn deref_place_until<'alloc, T>(
    mut ty: Type<'alloc>,
    place_mut: bool,
    num_derefs: &Cell<usize>,
    cond: impl FnOnce(bool, Type<'alloc>) -> TyckResult<T>,
) -> TyckResult<T> {
    let mut ptr_mut = true;
    while let Type::Pointer(m, inner) = ty {
        num_derefs.set(num_derefs.get() + 1);
        ty = *inner;
        ptr_mut &= m;
    }
    cond(
        if num_derefs.get() > 0 {
            ptr_mut
        } else {
            place_mut
        },
        ty,
    )
}

pub struct TypeChecker<'alloc> {
    pub var_decl: HashMap<VarId, &'alloc DeclInfo<'alloc>>,
    pub struct_tys: HashMap<&'alloc str, StructInfo<'alloc>>,
    pub func_decl: HashMap<&'alloc str, &'alloc DeclInfo<'alloc>>,
    pub bump: &'alloc Bump,
    pub env: Env<'alloc>,
}

impl<'check, 'alloc> TypeChecker<'alloc> {
    pub fn new(bump: &'alloc Bump) -> TypeChecker<'alloc> {
        TypeChecker {
            var_decl: hashmap! {},
            struct_tys: hashmap! {},
            func_decl: hashmap! {},
            bump,
            env: Env::new(),
        }
    }

    pub fn tyck_program(&mut self, program: &'alloc Vec<Defn<'alloc>>) -> TyckResult<()> {
        // Register all the definition declarations first before doing a full typecheck.
        for defn in program {
            match &defn.kind {
                DefnKind::Struct { name, members } => {
                    let mut seen_members = HashMap::new();
                    let members: TyckResult<Vec<_>> = members
                        .iter()
                        .map(|mem| {
                            if let Some(original_span) = seen_members.get(mem.name.item()) {
                                return Err(TyckError::StructHasDuplicateMember {
                                    duplicate_span: mem.name.span(),
                                    original_span: *original_span,
                                });
                            } else {
                                seen_members.insert(mem.name.item(), mem.name.span());
                            }
                            mem.ty.set(Some(Type::from(
                                self,
                                &mem.ty_ast.expect("struct members should have types"),
                            )?));
                            Ok(mem)
                        })
                        .collect();
                    let members = self.bump.alloc_slice_fill_iter(members?);
                    let name = name.map(|name| &*self.bump.alloc_str(name));
                    self.struct_tys.insert(
                        name.item(),
                        StructInfo {
                            name: self.bump.alloc(name),
                            members,
                        },
                    );
                }
                DefnKind::Fn { decl, .. } => {
                    decl.ty.set(Some(Type::from(
                        self,
                        &decl.ty_ast.expect("fn decls should have types"),
                    )?));
                    self.func_decl.insert(decl.name.item(), decl);
                }
                DefnKind::Static { decl, expr } => {
                    let expr_ty = self.tyck_expr(expr)?;
                    decl.ty.set(Some(expr_ty));
                    self.register_decl(decl);
                }
            }
        }

        // Go in and recursively typecheck each definition, now that we have the toplevel decls saved.
        for defn in program {
            self.tyck_defn(defn)?;
        }

        Ok(())
    }

    fn tyck_defn(&mut self, defn: &'alloc Defn<'alloc>) -> TyckResult<()> {
        match &defn.kind {
            DefnKind::Struct { .. } => {
                // Structs registered in 1st pass, nothing needs to be done here.
                Ok(())
            }
            DefnKind::Fn {
                decl,
                params,
                return_ty_ast: return_type,
                return_ty,
                body,
            } => self.tyck_fn(&decl.name, params, return_type, &return_ty, body),
            DefnKind::Static { decl, expr } => todo!(),
        }
    }

    fn register_decl(&mut self, decl: &'alloc DeclInfo<'alloc>) {
        let id = fresh_id();
        decl.id.set(Some(id));
        self.var_decl.insert(id, decl);
        self.env.set(decl.name.item(), id);
    }

    fn tyck_fn(
        &mut self,
        name: &Spanned<&'alloc str>,
        params: &'alloc [DeclInfo<'alloc>],
        return_ty_ast: &'alloc Option<ast::Type<'alloc>>,
        return_ty_cell: &Cell<Option<Type<'alloc>>>,
        body: &'alloc Expr<'alloc>,
    ) -> TyckResult<()> {
        let stmts = match &body.kind {
            ExprKind::Block(stmts) => stmts,
            kind => unreachable!("ICE: fn has a non-block body: {:?}", kind),
        };

        self.env.push();
        for param in params {
            param.ty.set(Some(Type::from(
                self,
                &param.ty_ast.expect("fn params always have a type"),
            )?));
            self.register_decl(param);
        }

        let return_ty = return_ty_ast.map_or(Ok(Type::Tuple(&[])), |t| Type::from(self, &t))?;
        let body_ty = self.tyck_block(stmts)?;
        self.env.pop();

        return_ty_cell.set(Some(return_ty));

        if is_assignable(&return_ty, &body_ty) {
            Ok(())
        } else {
            Err(TyckError::MismatchedTypes {
                expected_ty: human_type_name(&return_ty),
                got: stmts.last().map(|s| s.span).unwrap_or(body.span),
                got_ty: human_type_name(&body_ty),
            })
        }
    }

    fn tyck_expr(&mut self, expr: &'alloc Expr<'alloc>) -> TyckResult<Type<'alloc>> {
        let ty = match &expr.kind {
            ExprKind::Bool(_) => Ok(Type::Bool),
            ExprKind::Int(_) => Ok(Type::Int),
            ExprKind::Float(_) => Ok(Type::Float),
            ExprKind::Str(_) => Ok(Type::Str),
            ExprKind::Char(_) => Ok(Type::Char),
            ExprKind::Tuple(exprs) => {
                let mut tys = Vec::new();
                for expr in *exprs {
                    tys.push(self.tyck_expr(expr)?);
                }
                let tys = self.bump.alloc_slice_fill_iter(tys);
                Ok(Type::Tuple(tys))
            }
            ExprKind::Array(exprs, size) => {
                if let Some((first, rest)) = exprs.split_first() {
                    let ty = self.tyck_expr(first)?;
                    for expr in rest {
                        let other_ty = self.tyck_expr(expr)?;
                        tyck_assign(Spanned(expr.span, &ty), Spanned(expr.span, &other_ty))?;
                    }
                    let last = exprs.last().unwrap(); // Guaranteed since there is at least one elem at this point.
                    let size = if let &Some(Spanned(size_span, size)) = size {
                        if exprs.len() > size {
                            return Err(TyckError::InvalidArraySize {
                                elements_span: first.span.unite(last.span),
                                num_elements: exprs.len(),
                                size,
                                size_span,
                            });
                        }
                        size
                    } else {
                        exprs.len()
                    };

                    Ok(Type::Array(self.bump.alloc(ty), size))
                } else {
                    todo!("ummmmm no zero length arrays pls");
                }
            }
            ExprKind::Id(name, id, is_func) => {
                let decl = self.tyck_var(name, id, is_func, expr.span)?;
                Ok(decl.ty.get().expect("type"))
            }
            ExprKind::PrefixOp(op, expr) => self.tyck_prefix_op_expr(*op, expr),
            ExprKind::BinOp(Spanned(s, Operator::Simple(op)), lhs, rhs) => {
                self.tyck_simple_binop_expr(Spanned(*s, *op), lhs, rhs)
            }
            ExprKind::BinOp(Spanned(s, Operator::Assign(op)), lhs, rhs) => {
                self.tyck_assign_expr(Spanned(*s, *op), lhs, rhs)
            }
            ExprKind::Cast(_, ty) => Type::from(self, ty),
            ExprKind::Call(callee, args) => {
                let callee_ty = self.tyck_expr(callee)?;
                match callee_ty {
                    Type::Fn(param_tys, rty) => {
                        if param_tys.len() == args.len() {
                            for (param_ty, arg) in param_tys.iter().zip(*args) {
                                let arg_ty = self.tyck_expr(arg)?;
                                if !is_assignable(param_ty, &arg_ty) {
                                    return Err(TyckError::MismatchedTypes {
                                        expected_ty: human_type_name(param_ty),
                                        got: arg.span,
                                        got_ty: human_type_name(&arg_ty),
                                    });
                                }
                            }
                            Ok(*rty)
                        } else {
                            Err(TyckError::CallArityMismatch {
                                span: expr.span,
                                expected_arity: param_tys.len(),
                                actual_arity: args.len(),
                            })
                        }
                    }
                    _ => Err(TyckError::NotCallable { span: callee.span }),
                }
            }
            ExprKind::Block(stmts) => Ok(self.tyck_block(stmts)?),
            ExprKind::If(thens, els) => {
                let mut results = Vec::new();
                for (cond, then) in *thens {
                    let cond_ty = self.tyck_expr(cond)?;
                    if !is_assignable(&Type::Bool, &cond_ty) {
                        return Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&Type::Bool),
                            got: cond.span,
                            got_ty: human_type_name(&cond_ty),
                        });
                    }

                    let then_ty = self.tyck_expr(then)?;
                    results.push(then_ty);
                }

                let else_ty = match els {
                    Some(els) => self.tyck_expr(els)?,
                    None => Type::Tuple(&[]),
                };

                if !results.iter().all(|&ty| ty == else_ty) {
                    return Err(TyckError::MismatchedIfBranchTypes { span: expr.span });
                }

                Ok(else_ty)
            }
            ExprKind::Struct(Spanned(struct_name_span, struct_name), fields) => {
                let info = match self.struct_tys.get(struct_name) {
                    Some(info) => *info,
                    None => {
                        return Err(TyckError::UndefinedType {
                            span: *struct_name_span,
                            name: struct_name.to_string(),
                        })
                    }
                };

                let mut unused = HashMap::new();
                for member in info.members {
                    unused.insert(member.name.item(), member);
                }

                for (Spanned(field_name_span, field_name), expr) in *fields {
                    match unused.get(field_name) {
                        Some(&decl) => {
                            let expr_ty = self.tyck_expr(expr)?;
                            tyck_assign(
                                Spanned(*field_name_span, &decl.ty.get().expect("type")),
                                Spanned(expr.span, &expr_ty),
                            )?;
                            unused.remove(field_name);
                        }
                        None => {
                            return Err(TyckError::FieldDoesntExistOnStruct {
                                field: *field_name_span,
                                field_name: field_name.to_string(),
                                struct_name: struct_name.to_string(),
                            })
                        }
                    }
                }

                if !unused.is_empty() {
                    return Err(TyckError::MissingFieldsInStruct {
                        span: expr.span,
                        unused: format!(
                            "{}",
                            unused
                                .keys()
                                .map(|k| format!("'{}'", k))
                                .reduce(|acc, k| format!("{}, {}", acc, k))
                                .unwrap_or(String::new())
                        ),
                    });
                }

                let struct_name = self.bump.alloc_str(struct_name);
                Ok(Type::Struct(struct_name))
            }
            ExprKind::Field(expr, op_span, field, num_derefs) => {
                let expr_ty = self.tyck_expr(expr)?;
                deref_until(expr_ty, num_derefs, |ty| {
                    self.field_access_ty(Spanned(expr.span, ty), field)
                })
            }
            ExprKind::Group(expr) => self.tyck_expr(expr),
            ExprKind::Index(lhs, index, num_derefs) => {
                let lhs_ty = self.tyck_expr(lhs)?;
                let index_ty = self.tyck_expr(index)?;
                deref_until(lhs_ty, num_derefs, |ty| {
                    indexed_ty(ty, lhs.span, index_ty, index.span)
                })
            }
            ExprKind::Range(start, end) => {
                if let Some(start) = start {
                    let start_ty = self.tyck_expr(start)?;
                    tyck_is_of_type(Spanned(start.span, &start_ty), &[Type::Int])?;
                }

                if let Some(end) = end {
                    let end_ty = self.tyck_expr(end)?;
                    tyck_is_of_type(Spanned(end.span, &end_ty), &[Type::Int])?;
                }

                Ok(Type::Struct(match (start, end) {
                    (None, None) => RANGE_FULL_TY_NAME,
                    (Some(_), None) => RANGE_FROM_TY_NAME,
                    (None, Some(_)) => RANGE_TO_TY_NAME,
                    _ => RANGE_TY_NAME,
                }))
            }
        }?;

        expr.ty.set(Some(ty));

        Ok(ty)
    }

    fn tyck_var(
        &self,
        name: &str,
        id: &Cell<Option<usize>>,
        is_func: &Cell<bool>,
        span: Span,
    ) -> TyckResult<&'alloc DeclInfo<'alloc>> {
        if let Some(lookup_id) = self.env.get(name) {
            // Inject id from looking up in the env
            id.set(Some(lookup_id));
            let decl = self
                .var_decl
                .get(&lookup_id)
                .expect("ICE: var id in env was not in the var_decl map");
            is_func.set(false);
            Ok(decl)
        } else if let Some(decl) = self.func_decl.get(name) {
            is_func.set(true);
            Ok(decl)
        } else {
            Err(TyckError::UndefinedVar { span })
        }
    }

    fn tyck_assign_expr(
        &mut self,
        op: Spanned<Option<BinOpKind>>,
        lhs: &'alloc Expr<'alloc>,
        rhs: &'alloc Expr<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let (mutable, source, lhs_ty) = self.tyck_place_expr(lhs)?;

        if !mutable {
            match source {
                Some(Source::Id(s)) => {
                    return Err(TyckError::MutatingImmutableValue {
                        span: lhs.span,
                        source: s,
                    })
                }
                Some(Source::Ptr(s)) => {
                    return Err(TyckError::MutatingImmutableValueThroughPointer {
                        span: lhs.span,
                        ptr: s,
                    })
                }
                None => {
                    return Err(TyckError::MutatingImmutableValueOfUnknownCause { span: lhs.span })
                }
            }
        }

        let rhs_ty = self.tyck_expr(rhs)?;

        let Spanned(op_span, op) = op;
        let rhs_ty = match op {
            Some(op) => self.tyck_simple_binop_types(
                Spanned(op_span, op),
                Spanned(lhs.span, lhs_ty),
                Spanned(rhs.span, rhs_ty),
            )?,
            None => rhs_ty,
        };

        // Catch non-assignable types
        if let Type::Slice(_) = &lhs_ty {
            return Err(TyckError::CannotAssignUnsized {
                span: lhs.span,
                ty_name: human_type_name(&lhs_ty),
            });
        }

        if is_assignable(&lhs_ty, &rhs_ty) {
            Ok(Type::unit())
        } else {
            Err(TyckError::MismatchedTypes {
                expected_ty: human_type_name(&lhs_ty),
                got: rhs.span,
                got_ty: human_type_name(&rhs_ty),
            })
        }
    }

    fn tyck_simple_binop_types(
        &mut self,
        op: Spanned<BinOpKind>,
        Spanned(lhs_span, lhs_ty): Spanned<Type<'alloc>>,
        Spanned(rhs_span, rhs_ty): Spanned<Type<'alloc>>,
    ) -> TyckResult<Type<'alloc>> {
        let Spanned(op_span, op) = op;
        match op {
            // Additive operations
            BinOpKind::Plus | BinOpKind::Minus => match lhs_ty {
                Type::Int | Type::Float => {
                    tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int, Type::Float])?;
                    if lhs_ty != rhs_ty {
                        return Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&lhs_ty),
                            got: rhs_span,
                            got_ty: human_type_name(&rhs_ty),
                        });
                    }
                    Ok(lhs_ty)
                }
                Type::Char => {
                    tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int, Type::Char])?;
                    Ok(rhs_ty)
                }
                Type::Pointer(_, ptr_ty) => {
                    tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int])?;

                    if !is_sized(ptr_ty) {
                        return Err(TyckError::CannotDoArithmeticOnUnsizedPtr {
                            ptr_span: lhs_span,
                            ptr_ty: human_type_name(&ptr_ty),
                        });
                    }

                    Ok(lhs_ty)
                }
                _ => {
                    return Err(TyckError::MismatchedTypes {
                        expected_ty: "'int' or 'char' or 'float' or or a pointer".to_string(),
                        got: lhs_span,
                        got_ty: human_type_name(&lhs_ty),
                    })
                }
            },

            // Multiplicative operations
            BinOpKind::Star | BinOpKind::Slash | BinOpKind::Percent => {
                tyck_is_of_type(Spanned(lhs_span, &lhs_ty), &[Type::Int, Type::Float])?;
                tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int, Type::Float])?;
                if lhs_ty != rhs_ty {
                    return Err(TyckError::MismatchedTypes {
                        expected_ty: human_type_name(&lhs_ty),
                        got: rhs_span,
                        got_ty: human_type_name(&rhs_ty),
                    });
                }
                Ok(lhs_ty)
            }

            // Bit operations
            BinOpKind::Caret | BinOpKind::Amp | BinOpKind::Pipe => {
                tyck_is_of_type(Spanned(lhs_span, &lhs_ty), &[Type::Int, Type::Bool])?;
                tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int, Type::Bool])?;
                if lhs_ty != rhs_ty {
                    return Err(TyckError::MismatchedTypes {
                        expected_ty: human_type_name(&lhs_ty),
                        got: rhs_span,
                        got_ty: human_type_name(&rhs_ty),
                    });
                }
                Ok(lhs_ty)
            }

            // Shifts
            BinOpKind::LtLt | BinOpKind::GtGt => {
                tyck_is_of_type(Spanned(lhs_span, &lhs_ty), &[Type::Int])?;
                tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Int])?;
                Ok(lhs_ty)
            }

            // Bool operations
            BinOpKind::AmpAmp | BinOpKind::PipePipe => {
                tyck_is_of_type(Spanned(lhs_span, &lhs_ty), &[Type::Bool])?;
                tyck_is_of_type(Spanned(rhs_span, &rhs_ty), &[Type::Bool])?;
                Ok(Type::Bool)
            }

            // Comparison operations
            BinOpKind::Lt | BinOpKind::Le | BinOpKind::Gt | BinOpKind::Ge => {
                if !is_comparable(&lhs_ty) {
                    return Err(TyckError::TypeNotComparable {
                        ty_span: lhs_span,
                        ty_name: human_type_name(&lhs_ty),
                    });
                }

                if !is_comparable(&rhs_ty) {
                    return Err(TyckError::TypeNotComparable {
                        ty_span: rhs_span,
                        ty_name: human_type_name(&rhs_ty),
                    });
                }

                Ok(Type::Bool)
            }

            // Equality operations
            BinOpKind::EqEq | BinOpKind::Ne => {
                if lhs_ty != rhs_ty {
                    return Err(TyckError::MismatchedTypes {
                        expected_ty: human_type_name(&lhs_ty),
                        got: rhs_span,
                        got_ty: human_type_name(&rhs_ty),
                    });
                }

                Ok(Type::Bool)
            }
        }
    }

    fn tyck_simple_binop_expr(
        &mut self,
        op: Spanned<BinOpKind>,
        lhs: &'alloc Expr<'alloc>,
        rhs: &'alloc Expr<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let lhs_ty = self.tyck_expr(lhs)?;
        let rhs_ty = self.tyck_expr(rhs)?;
        self.tyck_simple_binop_types(op, Spanned(lhs.span, lhs_ty), Spanned(rhs.span, rhs_ty))
    }

    fn tyck_prefix_op_expr(
        &mut self,
        Spanned(op_span, op): Spanned<PrefixOpKind>,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let ty = self.tyck_expr(expr)?;
        match op {
            PrefixOpKind::Tilde => {
                tyck_is_of_type(Spanned(expr.span, &ty), &[Type::Int])?;
                Ok(Type::Int)
            }
            PrefixOpKind::Bang => {
                tyck_is_of_type(Spanned(expr.span, &ty), &[Type::Bool])?;
                Ok(Type::Bool)
            }
            PrefixOpKind::Plus => {
                tyck_is_of_type(Spanned(expr.span, &ty), &[Type::Int, Type::Float])?;
                Ok(ty)
            }
            PrefixOpKind::Minus => {
                tyck_is_of_type(Spanned(expr.span, &ty), &[Type::Int, Type::Float])?;
                Ok(ty)
            }
            PrefixOpKind::Star => match ty {
                Type::Pointer(_, of) => Ok(*of),
                _ => Err(TyckError::DereferencingNonPointer { span: expr.span }),
            },
            PrefixOpKind::Amp => {
                let (_, _, expr_ty) = self.tyck_place_expr(expr)?;
                Ok(Type::Pointer(false, self.bump.alloc(expr_ty)))
            }
            PrefixOpKind::AmpMut => {
                let (mutable, source, expr_ty) = self.tyck_place_expr(expr)?;
                if mutable {
                    Ok(Type::Pointer(true, self.bump.alloc(expr_ty)))
                } else {
                    match source.unwrap() {
                        Source::Id(s) | Source::Ptr(s) => {
                            Err(TyckError::MutPointerToNonMutLocation {
                                addressing_op: op_span,
                                location: expr.span,
                                source: s,
                            })
                        }
                    }
                }
            }
        }
    }

    /// Typechecks an expr, expecting it to be a "place" or "lval" expr - one that indicates a memory location.
    fn tyck_place_expr(
        &mut self,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<(bool, Option<Source>, Type<'alloc>)> {
        match &expr.kind {
            ExprKind::Id(name, id, is_func) => {
                let decl = self.tyck_var(name, id, is_func, expr.span)?;
                Ok((
                    decl.mutable,
                    Some(Source::Id(decl.name.span())),
                    decl.ty.get().expect("type"),
                ))
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Star), expr) => {
                let expr_ty = self.tyck_expr(expr)?;
                match expr_ty {
                    Type::Pointer(m, ty) => Ok((m, Some(Source::Ptr(expr.span)), *ty)),
                    _ => return Err(TyckError::DereferencingNonPointer { span: expr.span }),
                }
            }
            ExprKind::Field(expr, _, field, num_derefs) => {
                let (mutable, source, expr_ty) = self.tyck_place_expr(expr)?;
                deref_place_until(expr_ty, mutable, num_derefs, move |mutable, ty| {
                    let result_ty = self.field_access_ty(Spanned(expr.span, ty), field)?;
                    Ok((
                        mutable,
                        if num_derefs.get() > 0 { None } else { source },
                        result_ty,
                    ))
                })
            }
            ExprKind::Group(expr) => self.tyck_place_expr(expr),
            ExprKind::Index(lhs, index, num_derefs) => {
                let (mutable, source, lhs_ty) = self.tyck_place_expr(lhs)?;
                deref_place_until(lhs_ty, mutable, num_derefs, move |mutable, ty| {
                    let index_ty = self.tyck_expr(index)?;
                    let place_ty = indexed_ty(ty, lhs.span, index_ty, index.span)?;
                    Ok((
                        mutable,
                        if num_derefs.get() > 0 { None } else { source },
                        place_ty,
                    ))
                })
            }
            _ => Err(TyckError::NotAPlaceExpr { span: expr.span }),
        }
    }

    fn field_access_ty(
        &mut self,
        Spanned(expr_span, expr_ty): Spanned<Type<'alloc>>,
        Spanned(field_span, field): &Spanned<Field>,
    ) -> TyckResult<Type<'alloc>> {
        match expr_ty {
            Type::Struct(struct_name) => {
                let field_name = match field {
                    Field::Name(x, i) => {
                        let info = self
                            .struct_tys
                            .get(struct_name)
                            .expect("ICE: struct should be known");
                        i.set(Some(info.member_index(x)));
                        x
                    }
                    _ => {
                        return Err(TyckError::NoIndexFieldsOnStructs {
                            span: *field_span,
                            ty: human_type_name(&expr_ty),
                        })
                    }
                };

                let info = self
                    .struct_tys
                    .get(struct_name)
                    .expect("ICE: expr has struct ty that wasn't defined");

                match info
                    .members
                    .iter()
                    .find(|decl| decl.name.item() == field_name)
                {
                    Some(decl) => Ok(decl.ty.get().expect("type")),
                    _ => Err(TyckError::FieldDoesntExistOnStruct {
                        field: *field_span,
                        field_name: field_name.to_string(),
                        struct_name: struct_name.to_string(),
                    }),
                }
            }
            Type::Tuple(elems) => match field {
                Field::Index(i) => {
                    let i = usize::from_str(i)
                        .expect("ICE: index field access was not an unsigned integer");
                    if i < elems.len() {
                        Ok(elems[i])
                    } else {
                        Err(TyckError::TupleIndexOutOfBounds {
                            span: *field_span,
                            i,
                            ty: human_type_name(&expr_ty),
                        })
                    }
                }
                Field::Name(..) => Err(TyckError::NoNamedFieldsOnTuples {
                    span: *field_span,
                    ty: human_type_name(&expr_ty),
                }),
            },
            _ => Err(TyckError::TypeDoesNotSupportFieldAccess {
                span: expr_span,
                ty: human_type_name(&expr_ty),
            }),
        }
    }

    fn tyck_block(&mut self, stmts: &'alloc [Stmt<'alloc>]) -> TyckResult<Type<'alloc>> {
        self.env.push();
        let result = if let Some((last, init)) = stmts.split_last() {
            for stmt in init {
                self.tyck_stmt(stmt)?;
            }
            self.tyck_stmt(last)
        } else {
            Ok(Type::Tuple(&[]))
        };
        self.env.pop();
        result
    }

    fn tyck_stmt(&mut self, stmt: &'alloc Stmt<'alloc>) -> TyckResult<Type<'alloc>> {
        match &stmt.kind {
            StmtKind::Let(decl, expr) => {
                self.tyck_let(decl, expr)?;
                Ok(Type::Tuple(&[]))
            }
            StmtKind::Expr(expr) => self.tyck_expr(expr),
            StmtKind::Semi(expr) => {
                self.tyck_expr(expr)?;
                Ok(Type::Tuple(&[]))
            }
        }
    }

    fn tyck_let(
        &mut self,
        decl: &'alloc ast::DeclInfo<'alloc>,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<()> {
        let ty = self.tyck_expr(expr)?;
        let decl_ty = match &decl.ty_ast {
            Some(decl_ty_ast) => Type::from(self, decl_ty_ast)?,
            None => ty,
        };

        decl.ty.set(Some(decl_ty));
        self.register_decl(decl);

        tyck_assign(Spanned(decl.name.span(), &decl_ty), Spanned(expr.span, &ty))?;
        Ok(())
    }
}
