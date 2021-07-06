use bumpalo::Bump;
use codespan_derive::IntoDiagnostic;
use std::{collections::HashMap, fmt::Display};

use std::str::FromStr;

use crate::parser::ast::Operator;
use crate::{
    parser::ast::{BinOpKind, ExprKind, Field, PrefixOpKind, Spanned, Stmt, StmtKind},
    util::{FileId, Span},
};

use maplit::hashmap;

use crate::{
    parser::ast::{self, Defn, DefnKind, Expr, TypeKind},
    util::fresh_id,
};

const RANGE_TY_NAME: &str = "Range";
const RANGE_TO_TY_NAME: &str = "RangeTo";
const RANGE_FROM_TY_NAME: &str = "RangeFrom";
const RANGE_FULL_TY_NAME: &str = "RangeFull";

#[derive(Debug, Clone, Copy)]
pub struct StructInfo<'a> {
    name: &'a Spanned<&'a str>,
    members: &'a [DeclInfo<'a>],
}

#[derive(Debug)]
pub struct DeclInfo<'a> {
    /// The mutability of the variable.
    pub mutable: bool,

    /// The variable name.
    pub name: Spanned<&'a str>,

    /// The type of the variable, given in tyck phase.
    pub ty: Type<'a>,

    /// The type ascription span.
    pub ty_span: Option<Span>,

    /// The source span of the declaration.
    pub span: Span,

    /// Unique identifier for this variable, given in tyck phase.
    pub id: VarId,
}

impl<'a> DeclInfo<'a> {
    pub fn from<'ast, 'b>(
        bump: &'b Bump,
        state: &TyckState,
        decl: &'ast ast::DeclInfo<'ast>,
    ) -> TyckResult<DeclInfo<'b>> {
        Ok(DeclInfo {
            id: fresh_id(),
            mutable: decl.mutable,
            name: decl.name.map(|name| &*bump.alloc_str(name)),
            span: decl.span,
            ty: Type::from(bump, state, &decl.ty_ast)?,
            ty_span: Some(decl.ty_ast.span),
        })
    }

    pub fn from_let<'ast, 'b>(
        bump: &'b Bump,
        decl: &'ast ast::LetDeclInfo<'ast>,
        ty: Type<'b>,
    ) -> DeclInfo<'b> {
        DeclInfo {
            id: fresh_id(),
            mutable: decl.mutable,
            name: decl.name.map(|name| &*bump.alloc_str(name)),
            span: decl.span,
            ty,
            ty_span: decl.ty_ast.map(|ast| ast.span),
        }
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

    pub fn from<'b>(bump: &'b Bump, state: &TyckState, ty: &ast::Type) -> TyckResult<Type<'b>> {
        Ok(match &ty.kind {
            TypeKind::Id("bool") => Type::Bool,
            TypeKind::Id("int") => Type::Int,
            TypeKind::Id("float") => Type::Float,
            TypeKind::Id("char") => Type::Char,
            TypeKind::Id("str") => Type::Str,
            TypeKind::Id(other) => {
                if state.struct_tys.contains_key(other) {
                    Type::Struct(bump.alloc_str(other))
                } else {
                    return Err(TyckError::UndefinedType {
                        name: other.to_string(),
                        span: ty.span,
                    });
                }
            }
            TypeKind::Pointer(m, inner) => {
                let inner = bump.alloc(Type::from(bump, state, inner)?);
                Type::Pointer(*m, inner)
            }
            TypeKind::Slice(inner) => {
                let inner = bump.alloc(Type::from(bump, state, inner)?);
                Type::Slice(inner)
            }
            TypeKind::Array(inner, size) => {
                let inner = bump.alloc(Type::from(bump, state, inner)?);
                Type::Array(inner, *size)
            }
            TypeKind::Tuple([]) => Type::Tuple(&[]),
            TypeKind::Tuple(inners) => {
                let inners: TyckResult<Vec<_>> = inners
                    .iter()
                    .map(|inner| Type::from(bump, state, inner))
                    .collect();
                let inners = bump.alloc_slice_fill_iter(inners?);
                Type::Tuple(inners)
            }
            TypeKind::Fn(params, return_ty) => {
                let params: TyckResult<Vec<_>> = params
                    .iter()
                    .map(|param| Type::from(bump, state, param))
                    .collect();
                let params = bump.alloc_slice_fill_iter(params?);
                let return_ty = bump.alloc(
                    return_ty.map_or(Ok(Type::Tuple(&[])), |t| Type::from(bump, state, &t))?,
                );
                Type::Fn(params, return_ty)
            }
        })
    }
}

pub struct Env<'ast> {
    scopes: Vec<HashMap<&'ast str, VarId>>,
}

impl<'ast> Env<'ast> {
    pub fn new() -> Env<'ast> {
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

    pub fn set(&mut self, name: &'ast str, id: VarId) {
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

#[derive(IntoDiagnostic, Debug)]
#[file_id(FileId)]
pub enum TyckError {
    #[message = "Mismatched types"]
    MismatchedTypes {
        expected_ty: String,
        #[primary = "expected {expected_ty} but got {got_ty}"]
        got: Span,
        got_ty: String,
    },

    #[message = "Variable undefined"]
    UndefinedVar {
        #[primary]
        span: Span,
    },

    #[message = "Expected a place expression in this position"]
    NotAPlaceExpr {
        #[primary]
        span: Span,
    },

    #[message = "Dereferencing a non-pointer type"]
    DereferencingNonPointer {
        #[primary]
        span: Span,
    },

    #[message = "Cannot make a mut pointer to an immutable location"]
    MutPointerToNonMutLocation {
        #[primary]
        addressing_op: Span,
        #[secondary = "this place expression is immutable"]
        location: Span,
        #[secondary = "consider adding 'mut'"]
        source: Span,
    },

    #[message = "Type '{name}' is not defined anywhere"]
    UndefinedType {
        #[primary]
        span: Span,
        name: String,
    },

    #[message = "The field '{field_name}' does not exist on struct '{struct_name}'"]
    FieldDoesntExistOnStruct {
        #[primary]
        field: Span,
        field_name: String,
        struct_name: String,
    },

    #[message = "Struct expr has missing fields: {unused}"]
    MissingFieldsInStruct {
        #[primary]
        span: Span,
        unused: String,
    },

    #[message = "Not a callable expression"]
    NotCallable {
        #[primary]
        span: Span,
    },

    #[message = "This call expects {expected_arity} args but got {actual_arity}"]
    CallArityMismatch {
        #[primary]
        span: Span,
        expected_arity: usize,
        actual_arity: usize,
    },

    #[message = "Type {ty} does not support field access"]
    TypeDoesNotSupportFieldAccess {
        #[primary]
        span: Span,
        ty: String,
    },

    #[message = "Type {ty} is a struct; use a field name to access a field"]
    NoIndexFieldsOnStructs {
        #[primary]
        span: Span,
        ty: String,
    },

    #[message = "Type {ty} is a tuple; use an index to access an element"]
    NoNamedFieldsOnTuples {
        #[primary]
        span: Span,
        ty: String,
    },

    #[message = "Index {i} is out of bounds of tuple {ty}"]
    TupleIndexOutOfBounds {
        #[primary]
        span: Span,
        i: usize,
        ty: String,
    },

    #[message = "Branches of this if expression have mismatching types"]
    MismatchedIfBranchTypes {
        #[primary]
        span: Span,
    },

    #[message = "Cannot mutate immmutable value"]
    MutatingImmutableValue {
        #[primary]
        span: Span,
        #[secondary = "consider making this 'mut'"]
        source: Span,
    },

    #[message = "Cannot mutate immmutable value"]
    MutatingImmutableValueThroughPointer {
        #[secondary]
        span: Span,
        #[primary = "consider making this a mutable pointer"]
        ptr: Span,
    },

    #[message = "Invalid array size"]
    InvalidArraySize {
        #[primary = "given size is {size}"]
        size_span: Span,
        size: usize,
        #[secondary = "{num_elements} element(s) given"]
        elements_span: Span,
        num_elements: usize,
    },

    #[message = "Cannot assign an unsized type"]
    CannotAssignUnsized {
        #[primary = "size of {ty_name} is not known at compile-time"]
        span: Span,
        ty_name: String,
    },

    #[message = "Type {ty_name} is not indexable"]
    TypeNotIndexable {
        #[primary]
        span: Span,
        ty_name: String,
    },

    #[message = "Type {ty_name} cannot be used as an index"]
    TypeCannotBeUsedAsAnIndex {
        #[primary]
        span: Span,
        ty_name: String,
    },

    #[message = "Type {ty_name} cannot be compared"]
    TypeNotComparable {
        #[primary]
        ty_span: Span,
        ty_name: String,
    },

    #[message = "Cannot do arithmetic on unsized pointer type"]
    CannotDoArithmeticOnUnsizedPtr {
        #[primary = "this points to type {ptr_ty}, which is unsized"]
        ptr_span: Span,
        ptr_ty: String,
    },
}

type TyckResult<T> = Result<T, TyckError>;

pub struct TyckState<'alloc> {
    pub var_decl: HashMap<VarId, &'alloc DeclInfo<'alloc>>,
    pub struct_tys: HashMap<&'alloc str, StructInfo<'alloc>>,
}

impl<'ast> TyckState<'ast> {
    pub fn new() -> TyckState<'ast> {
        TyckState {
            var_decl: hashmap! {},
            struct_tys: hashmap! {},
        }
    }
}

pub fn tyck_program<'ast>(program: &'ast Vec<Defn<'ast>>) -> TyckResult<()> {
    let mut state = TyckState::new();
    let bump = Bump::new();
    let mut env = Env::new();

    // Register all the definition declarations first before doing a full typecheck.
    for defn in program {
        match &defn.kind {
            DefnKind::Struct { name, members } => {
                let members: TyckResult<Vec<_>> = members
                    .iter()
                    .map(|mem| DeclInfo::from(&bump, &state, mem))
                    .collect();
                let members = bump.alloc_slice_fill_iter(members?);
                let name = name.map(|name| &*bump.alloc_str(name));
                state.struct_tys.insert(
                    name.item(),
                    StructInfo {
                        name: bump.alloc(name),
                        members,
                    },
                );
            }
            DefnKind::Fn { decl, .. } => {
                let decl = bump.alloc(DeclInfo::from(&bump, &state, decl)?);
                register_decl(&mut state, &mut env, decl);
            }
            DefnKind::Static { decl, expr } => {
                let expr_ty = tyck_expr(&mut state, &bump, &mut env, expr)?;
                let decl = bump.alloc(DeclInfo::from_let(&bump, decl, expr_ty));
                register_decl(&mut state, &mut env, decl);
            }
        }
    }

    // Go in and recursively typecheck each definition, now that we have the toplevel decls saved.
    for defn in program {
        tyck_defn(&mut state, &bump, &mut env, defn)?;
    }

    Ok(())
}

fn tyck_defn<'ast, 'state, 'bump, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    defn: &'ast Defn<'ast>,
) -> TyckResult<()>
where
    'ast: 'bump,
{
    match &defn.kind {
        DefnKind::Struct { .. } => {
            // Structs registered in 1st pass, nothing needs to be done here.
            Ok(())
        }
        DefnKind::Fn {
            decl,
            params,
            return_type,
            body,
        } => {
            let params: TyckResult<Vec<_>> = params
                .iter()
                .map(|param| DeclInfo::from(bump, state, param))
                .collect();
            let params = bump.alloc_slice_fill_iter(params?);
            tyck_fn(state, bump, env, &decl.name, params, return_type, body)
        }
        DefnKind::Static { decl, expr } => todo!(),
    }
}

fn register_decl<'ast, 'state, 'bump, 'env>(
    state: &'state mut TyckState<'bump>,
    env: &'env mut Env<'bump>,
    decl: &'bump DeclInfo<'bump>,
) {
    state.var_decl.insert(decl.id, decl);
    env.set(decl.name.item(), decl.id);
}

fn tyck_fn<'ast, 'state, 'bump, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    name: &Spanned<&'bump str>,
    params: &'bump [DeclInfo<'bump>],
    return_ty_ast: &'ast Option<ast::Type<'ast>>,
    body: &'ast Expr<'ast>,
) -> TyckResult<()> {
    let stmts = match &body.kind {
        ExprKind::Block(stmts) => stmts,
        kind => unreachable!("ICE: fn has a non-block body: {:?}", kind),
    };

    for param in params {
        register_decl(state, env, param);
    }

    let return_ty = return_ty_ast.map_or(Ok(Type::Tuple(&[])), |t| Type::from(bump, state, &t))?;
    let body_ty = tyck_block(state, bump, env, stmts)?;

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

fn tyck_expr<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    expr: &'ast Expr<'ast>,
) -> TyckResult<Type<'bump>> {
    match &expr.kind {
        ExprKind::Bool(_) => Ok(Type::Bool),
        ExprKind::Int(_) => Ok(Type::Int),
        ExprKind::Float(_) => Ok(Type::Float),
        ExprKind::Str(_) => Ok(Type::Str),
        ExprKind::Char(_) => Ok(Type::Char),
        ExprKind::Tuple(exprs) => {
            let mut tys = Vec::new();
            for expr in *exprs {
                tys.push(tyck_expr(state, bump, env, expr)?);
            }
            let tys = bump.alloc_slice_fill_iter(tys);
            Ok(Type::Tuple(tys))
        }
        ExprKind::Array(exprs, size) => {
            if let Some((first, rest)) = exprs.split_first() {
                let ty = tyck_expr(state, bump, env, first)?;
                for expr in rest {
                    let other_ty = tyck_expr(state, bump, env, expr)?;
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

                Ok(Type::Array(bump.alloc(ty), size))
            } else {
                todo!("ummmmm no zero length arrays pls");
            }
        }
        ExprKind::Id(name, id) => {
            if let Some(lookup_id) = env.get(name) {
                // Inject id from looking up in the env
                id.set(Some(lookup_id));
                let decl = state
                    .var_decl
                    .get(&lookup_id)
                    .expect("ICE: var id in env was not in the var_decl map");
                Ok(decl.ty)
            } else {
                Err(TyckError::UndefinedVar { span: expr.span })
            }
        }
        ExprKind::PrefixOp(op, expr) => tyck_prefix_op_expr(state, bump, env, *op, expr),
        ExprKind::BinOp(Spanned(s, Operator::Simple(op)), lhs, rhs) => {
            tyck_simple_binop_expr(state, bump, env, Spanned(*s, *op), lhs, rhs)
        }
        ExprKind::BinOp(Spanned(s, Operator::Assign(op)), lhs, rhs) => {
            tyck_assign_expr(state, bump, env, Spanned(*s, *op), lhs, rhs)
        }
        ExprKind::Cast(_, ty) => Type::from(bump, state, ty),
        ExprKind::Call(callee, args) => {
            let callee_ty = tyck_expr(state, bump, env, callee)?;
            match callee_ty {
                Type::Fn(param_tys, rty) => {
                    if param_tys.len() == args.len() {
                        for (param_ty, arg) in param_tys.iter().zip(*args) {
                            let arg_ty = tyck_expr(state, bump, env, arg)?;
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
        ExprKind::Block(stmts) => Ok(tyck_block(state, bump, env, stmts)?),
        ExprKind::If(thens, els) => {
            let mut results = Vec::new();
            for (cond, then) in *thens {
                let cond_ty = tyck_expr(state, bump, env, cond)?;
                if !is_assignable(&Type::Bool, &cond_ty) {
                    return Err(TyckError::MismatchedTypes {
                        expected_ty: human_type_name(&Type::Bool),
                        got: cond.span,
                        got_ty: human_type_name(&cond_ty),
                    });
                }

                let then_ty = tyck_expr(state, bump, env, then)?;
                results.push(then_ty);
            }

            let else_ty = match els {
                Some(els) => tyck_expr(state, bump, env, els)?,
                None => Type::Tuple(&[]),
            };

            if !results.iter().all(|&ty| ty == else_ty) {
                return Err(TyckError::MismatchedIfBranchTypes { span: expr.span });
            }

            Ok(else_ty)
        }
        ExprKind::Struct(Spanned(struct_name_span, struct_name), fields) => {
            let info = match state.struct_tys.get(struct_name) {
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
                        let expr_ty = tyck_expr(state, bump, env, expr)?;
                        tyck_assign(
                            Spanned(*field_name_span, &decl.ty),
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

            let struct_name = bump.alloc_str(struct_name);
            Ok(Type::Struct(struct_name))
        }
        ExprKind::Field(expr, op_span, field) => {
            let expr_ty = tyck_expr(state, bump, env, expr)?;
            field_access_ty(state, Spanned(expr.span, expr_ty), field)
        }
        ExprKind::Group(expr) => tyck_expr(state, bump, env, expr),
        ExprKind::Index(lhs, index) => {
            let lhs_ty = tyck_expr(state, bump, env, lhs)?;
            let index_ty = tyck_expr(state, bump, env, index)?;
            Ok(indexed_ty(lhs_ty, lhs, index_ty, index)?)
        }
        ExprKind::Range(start, end) => {
            if let Some(start) = start {
                let start_ty = tyck_expr(state, bump, env, start)?;
                tyck_is_of_type(Spanned(start.span, &start_ty), &[Type::Int])?;
            }

            if let Some(end) = end {
                let end_ty = tyck_expr(state, bump, env, end)?;
                tyck_is_of_type(Spanned(end.span, &end_ty), &[Type::Int])?;
            }

            Ok(Type::Struct(match (start, end) {
                (None, None) => RANGE_FULL_TY_NAME,
                (Some(_), None) => RANGE_FROM_TY_NAME,
                (None, Some(_)) => RANGE_TO_TY_NAME,
                _ => RANGE_TY_NAME,
            }))
        }
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

fn tyck_assign_expr<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    op: Spanned<Option<BinOpKind>>,
    lhs: &'ast Expr<'ast>,
    rhs: &'ast Expr<'ast>,
) -> TyckResult<Type<'bump>> {
    let (mutable, source, lhs_ty) = tyck_place_expr(state, bump, env, lhs)?;

    if !mutable {
        match source.unwrap() {
            Source::Id(s) => {
                return Err(TyckError::MutatingImmutableValue {
                    span: lhs.span,
                    source: s,
                })
            }
            Source::Ptr(s) => {
                return Err(TyckError::MutatingImmutableValueThroughPointer {
                    span: lhs.span,
                    ptr: s,
                })
            }
        }
    }

    let rhs_ty = tyck_expr(state, bump, env, rhs)?;

    let Spanned(op_span, op) = op;
    let rhs_ty = match op {
        Some(op) => tyck_simple_binop_types(
            state,
            bump,
            env,
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

fn tyck_simple_binop_types<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    op: Spanned<BinOpKind>,
    Spanned(lhs_span, lhs_ty): Spanned<Type<'bump>>,
    Spanned(rhs_span, rhs_ty): Spanned<Type<'bump>>,
) -> TyckResult<Type<'bump>> {
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

fn is_comparable(ty: &Type) -> bool {
    match ty {
        Type::Int | Type::Float | Type::Char => true,
        _ => false,
    }
}

fn tyck_simple_binop_expr<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    op: Spanned<BinOpKind>,
    lhs: &'ast Expr<'ast>,
    rhs: &'ast Expr<'ast>,
) -> TyckResult<Type<'bump>> {
    let lhs_ty = tyck_expr(state, bump, env, lhs)?;
    let rhs_ty = tyck_expr(state, bump, env, rhs)?;
    tyck_simple_binop_types(
        state,
        bump,
        env,
        op,
        Spanned(lhs.span, lhs_ty),
        Spanned(rhs.span, rhs_ty),
    )
}

fn tyck_prefix_op_expr<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    Spanned(op_span, op): Spanned<PrefixOpKind>,
    expr: &'ast Expr<'ast>,
) -> TyckResult<Type<'bump>> {
    let ty = tyck_expr(state, bump, env, expr)?;
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
            let (_, _, expr_ty) = tyck_place_expr(state, bump, env, expr)?;
            Ok(Type::Pointer(false, bump.alloc(expr_ty)))
        }
        PrefixOpKind::AmpMut => {
            let (mutable, source, expr_ty) = tyck_place_expr(state, bump, env, expr)?;
            if mutable {
                Ok(Type::Pointer(true, bump.alloc(expr_ty)))
            } else {
                match source.unwrap() {
                    Source::Id(s) | Source::Ptr(s) => Err(TyckError::MutPointerToNonMutLocation {
                        addressing_op: op_span,
                        location: expr.span,
                        source: s,
                    }),
                }
            }
        }
    }
}

enum Source {
    Id(Span),
    Ptr(Span),
}

/// Typechecks an expr, expecting it to be a "place" or "lval" expr - one that indicates a memory location.
fn tyck_place_expr<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    expr: &'ast Expr<'ast>,
) -> TyckResult<(bool, Option<Source>, Type<'bump>)> {
    match &expr.kind {
        ExprKind::Id(name, id) => {
            if let Some(lookup_id) = env.get(name) {
                // Inject id from looking up in the env
                id.set(Some(lookup_id));
                let decl = state
                    .var_decl
                    .get(&lookup_id)
                    .expect("ICE: var id in env was not in the var_decl map");
                Ok((decl.mutable, Some(Source::Id(decl.name.span())), decl.ty))
            } else {
                return Err(TyckError::UndefinedVar { span: expr.span });
            }
        }
        ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Star), expr) => {
            let expr_ty = tyck_expr(state, bump, env, expr)?;
            match expr_ty {
                Type::Pointer(m, ty) => Ok((m, Some(Source::Ptr(expr.span)), *ty)),
                _ => return Err(TyckError::DereferencingNonPointer { span: expr.span }),
            }
        }
        ExprKind::Field(expr, _, field) => {
            let (mutable, source, expr_ty) = tyck_place_expr(state, bump, env, expr)?;
            let result_ty = field_access_ty(state, Spanned(expr.span, expr_ty), field)?;
            Ok((mutable, source, result_ty))
        }
        ExprKind::Group(expr) => tyck_place_expr(state, bump, env, expr),
        ExprKind::Index(lhs, index) => {
            let (mutable, source, lhs_ty) = tyck_place_expr(state, bump, env, lhs)?;
            let index_ty = tyck_expr(state, bump, env, index)?;
            let place_ty = indexed_ty(lhs_ty, lhs, index_ty, index)?;
            Ok((mutable, source, place_ty))
        }
        _ => Err(TyckError::NotAPlaceExpr { span: expr.span }),
    }
}

fn field_access_ty<'ast, 'state, 'bump>(
    state: &'state mut TyckState<'bump>,
    Spanned(expr_span, expr_ty): Spanned<Type<'bump>>,
    Spanned(field_span, field): &Spanned<Field>,
) -> TyckResult<Type<'bump>> {
    match expr_ty {
        Type::Struct(struct_name) => {
            let field_name = match field {
                Field::Name(x) => x,
                _ => {
                    return Err(TyckError::NoIndexFieldsOnStructs {
                        span: *field_span,
                        ty: human_type_name(&expr_ty),
                    })
                }
            };

            let info = state
                .struct_tys
                .get(struct_name)
                .expect("ICE: expr has struct ty that wasn't defined");

            match info
                .members
                .iter()
                .find(|decl| decl.name.item() == field_name)
            {
                Some(decl) => Ok(decl.ty),
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
            Field::Name(_) => Err(TyckError::NoNamedFieldsOnTuples {
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

fn indexed_ty<'bump>(
    lhs_ty: Type<'bump>,
    lhs: &Expr,
    index_ty: Type,
    index: &Expr,
) -> TyckResult<Type<'bump>> {
    let elem_ty = match lhs_ty {
        Type::Slice(ty) => ty,
        Type::Array(ty, _) => ty,
        _ => {
            return Err(TyckError::TypeNotIndexable {
                span: lhs.span,
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
                span: index.span,
                ty_name: human_type_name(&index_ty),
            })
        }
    };
    Ok(place_ty)
}

fn tyck_block<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    stmts: &'ast [Stmt<'ast>],
) -> TyckResult<Type<'bump>> {
    env.push();
    let result = if let Some((last, init)) = stmts.split_last() {
        for stmt in init {
            tyck_stmt(state, bump, env, stmt)?;
        }
        tyck_stmt(state, bump, env, last)
    } else {
        Ok(Type::Tuple(&[]))
    };
    env.pop();
    result
}

fn tyck_stmt<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    stmt: &'ast Stmt<'ast>,
) -> TyckResult<Type<'bump>> {
    match &stmt.kind {
        StmtKind::Let(decl, expr) => {
            tyck_let(state, bump, env, decl, expr)?;
            Ok(Type::Tuple(&[]))
        }
        StmtKind::Expr(expr) => tyck_expr(state, bump, env, expr),
        StmtKind::Semi(expr) => {
            tyck_expr(state, bump, env, expr)?;
            Ok(Type::Tuple(&[]))
        }
    }
}

fn tyck_let<'ast, 'bump, 'state, 'env>(
    state: &'state mut TyckState<'bump>,
    bump: &'bump Bump,
    env: &'env mut Env<'bump>,
    decl: &'ast ast::LetDeclInfo<'ast>,
    expr: &'ast Expr<'ast>,
) -> TyckResult<()> {
    let ty = tyck_expr(state, bump, env, expr)?;
    let decl_ty = match &decl.ty_ast {
        Some(decl_ty_ast) => Type::from(bump, state, decl_ty_ast)?,
        None => ty,
    };

    tyck_assign(Spanned(decl.name.span(), &decl_ty), Spanned(expr.span, &ty))?;
    let decl = bump.alloc(DeclInfo::from_let(bump, decl, ty));
    register_decl(state, env, decl);
    Ok(())
}
