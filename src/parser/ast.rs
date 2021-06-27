use bumpalo::Bump;
use std::{cell::Cell, str::FromStr};

use crate::{
    lowering::VarId,
    util::{FileId, Span},
};

#[derive(Debug, PartialEq, Eq)]
pub struct Spanned<T>(pub Span, pub T);

impl<T> Spanned<T> {
    pub fn new(span: Span, item: T) -> Spanned<T> {
        Spanned(span, item)
    }

    pub fn span(&self) -> Span {
        self.0
    }

    pub fn item(&self) -> &T {
        &self.1
    }

    pub fn take_item(self) -> T {
        self.1
    }

    pub fn map<F, U>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned(self.0, f(self.1))
    }
}

/// Contains the information for a variable declaration.
#[derive(Debug)]
pub struct DeclInfo<'a> {
    /// The mutability of the variable.
    pub mutable: bool,

    /// The variable name.
    pub name: &'a str,

    /// The type of the variable.
    pub ty: Type<'a>,

    /// The source span of the declaration.
    pub span: Span,
}

#[derive(Debug)]
pub enum DefnKind<'a> {
    Struct {
        name: Spanned<&'a str>,
        members: &'a [DeclInfo<'a>],
    },
    Fn {
        name: &'a str,
        params: &'a [DeclInfo<'a>],
        return_type: Option<Type<'a>>,
        body: Expr<'a>,
    },
    Static {
        decl: DeclInfo<'a>,
        expr: Expr<'a>,
    },
}

#[derive(Debug)]
pub struct Defn<'a> {
    pub kind: DefnKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum StmtKind<'a> {
    Let(DeclInfo<'a>, Expr<'a>),
    Expr(Expr<'a>),
    Semi(Expr<'a>),
}

#[derive(Debug)]
pub struct Stmt<'a> {
    pub kind: StmtKind<'a>,
    pub span: Span,
}

#[derive(Debug)]
pub enum ExprKind<'a> {
    Int(&'a str),
    Float(&'a str),
    Str(&'a str),
    Char(&'a str),

    Tuple(&'a [Expr<'a>]),
    Array(&'a [Expr<'a>], Option<usize>),

    Id(&'a str, Option<Cell<VarId>>),

    PrefixOp(Spanned<Operator>, &'a Expr<'a>),
    BinOp(Spanned<Operator>, &'a Expr<'a>, &'a Expr<'a>),
    Cast(&'a Expr<'a>, Type<'a>),

    Call(&'a Expr<'a>, &'a [Expr<'a>]),
    Block(&'a [Stmt<'a>]),

    If(&'a [(Expr<'a>, Expr<'a>)], &'a Option<Expr<'a>>),
}

#[derive(Debug)]
pub struct Expr<'a> {
    pub span: Span,
    pub kind: ExprKind<'a>,
}

impl<'a> Expr<'a> {
    pub fn new(kind: ExprKind<'a>, file_id: FileId, start: usize, end: usize) -> Expr {
        Expr {
            span: Span::new(file_id, start, end),
            kind,
        }
    }
}

#[derive(Debug)]
pub enum Operator {
    Simple(OpKind),
    CompoundAssignment(OpKind),
}

#[derive(Debug)]
pub enum OpKind {
    Tilde,
    Bang,

    Plus,
    Minus,

    Star,
    Slash,
    Percent,
    Caret,

    Amp,
    Pipe,
    AmpAmp,
    PipePipe,

    AmpMut,

    Lt,
    LtLt,
    Le,
    Gt,
    GtGt,
    Ge,
    Eq,
    EqEq,
    Ne,

    Dot,
    As,
}

#[derive(Debug)]
pub struct Type<'a> {
    kind: TypeKind<'a>,
    span: Span,
}

#[derive(Debug)]
pub enum TypeKind<'a> {
    Id(&'a str),
    Pointer(bool, &'a Type<'a>),
    Slice(&'a Type<'a>),
    Array(&'a Type<'a>, usize),
    Tuple(&'a [Type<'a>]),
    Fn(&'a [Type<'a>], &'a Option<Type<'a>>),
}

// Expression builder functions

pub fn make_lit_expr<'b, 'input, F>(
    bump: &'b Bump,
    _: FileId,
    kind: F,
    s: Spanned<&'input str>,
) -> Expr<'b>
where
    F: FnOnce(&str) -> ExprKind,
{
    let Spanned(span, s) = s;
    let s = bump.alloc_str(s);
    Expr {
        kind: kind(s),
        span,
    }
}

pub fn make_prefix_op_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    op: Spanned<Operator>,
    e: Expr<'b>,
) -> Expr<'b> {
    let (start, end) = (op.span().start, e.span.end);
    let e = bump.alloc(e);
    Expr::new(ExprKind::PrefixOp(op, e), file_id, start, end)
}

pub fn make_binop_expr<'b>(
    bump: &'b Bump,
    _: FileId,
    e1: Expr<'b>,
    op: Spanned<Operator>,
    e2: Expr<'b>,
) -> Expr<'b> {
    let e1 = bump.alloc(e1);
    let e2 = bump.alloc(e2);
    Expr {
        kind: ExprKind::BinOp(op, e1, e2),
        span: e1.span.unite(e2.span),
    }
}

pub fn make_cast_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    expr: Expr<'b>,
    _: Spanned<Operator>,
    ty: Type<'b>,
) -> Expr<'b> {
    let (start, end) = (expr.span.start, ty.span.end);
    let expr = bump.alloc(expr);
    Expr {
        kind: ExprKind::Cast(expr, ty),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_call_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    callee: Expr<'b>,
    args: Vec<Expr<'b>>,
    end_span: usize,
) -> Expr<'b> {
    let (start, end) = (callee.span.start, end_span);
    let callee = bump.alloc(callee);
    let args = bump.alloc_slice_fill_iter(args);
    Expr::new(ExprKind::Call(callee, args), file_id, start, end)
}

pub fn make_block_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    stmts: Option<Vec<Stmt<'b>>>,
    end: usize,
) -> Expr<'b> {
    let stmts = stmts.unwrap_or_else(|| vec![]);
    let stmts = bump.alloc_slice_fill_iter(stmts);
    Expr::new(ExprKind::Block(stmts), file_id, start, end)
}

pub fn make_tuple_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    exprs: Vec<Expr<'b>>,
    end: usize,
) -> Expr<'b> {
    let exprs = bump.alloc_slice_fill_iter(exprs);
    Expr {
        kind: ExprKind::Tuple(exprs),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_array_expr<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    exprs: Vec<Expr<'b>>,
    size: Option<&'input str>,
    end: usize,
) -> Expr<'b> {
    let exprs = bump.alloc_slice_fill_iter(exprs);
    let size = size
        .map(|size| usize::from_str(size).expect("ICE: array size should be an unsigned integer"));
    Expr {
        kind: ExprKind::Array(exprs, size),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_if_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    condition: Expr<'b>,
    then_block: Expr<'b>,
    mut inner_ifs: Vec<(Expr<'b>, Expr<'b>)>,
    else_block: Option<Expr<'b>>,
    end: usize,
) -> Expr<'b> {
    inner_ifs.insert(0, (condition, then_block));
    let inner_ifs = bump.alloc_slice_fill_iter(inner_ifs);
    let else_block = bump.alloc(else_block);
    Expr::new(ExprKind::If(inner_ifs, else_block), file_id, start, end)
}

// Statement builder functions

pub fn make_expr_stmt<'b>(_: &'b Bump, _: FileId, semi: bool, expr: Expr<'b>) -> Stmt<'b> {
    let span = expr.span;
    let kind = if semi {
        StmtKind::Semi(expr)
    } else {
        StmtKind::Expr(expr)
    };
    Stmt { kind, span }
}

pub fn make_let_stmt<'b, 'input>(
    _: &'b Bump,
    file_id: FileId,
    start: usize,
    decl_info: DeclInfo<'b>,
    expr: Expr<'b>,
    end: usize,
) -> Stmt<'b> {
    Stmt {
        kind: StmtKind::Let(decl_info, expr),
        span: Span::new(file_id, start, end),
    }
}

// Type builder functions

pub fn make_id_type<'b, 'input>(bump: &'b Bump, _: FileId, id: Spanned<&'input str>) -> Type<'b> {
    let Spanned(span, id) = id;
    let id = bump.alloc_str(id);
    Type {
        kind: TypeKind::Id(id),
        span,
    }
}

pub fn make_pointer_type<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    mut_kw: Option<&'input str>,
    ty: Type<'b>,
) -> Type<'b> {
    let ty = bump.alloc(ty);
    Type {
        kind: TypeKind::Pointer(mut_kw.is_some(), ty),
        span: Span::new(file_id, start, ty.span.end),
    }
}

pub fn make_slice_type<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    ty: Type<'b>,
    end: usize,
) -> Type<'b> {
    let ty = bump.alloc(ty);
    Type {
        kind: TypeKind::Slice(ty),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_array_type<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    ty: Type<'b>,
    int: &'input str,
    end: usize,
) -> Type<'b> {
    let ty = bump.alloc(ty);
    Type {
        kind: TypeKind::Array(
            ty,
            usize::from_str(int).expect("ICE: array size should be an unsigned integer."),
        ),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_tuple_type<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    tys: Vec<Type<'b>>,
    end: usize,
) -> Type<'b> {
    let tys = bump.alloc_slice_fill_iter(tys);
    Type {
        kind: TypeKind::Tuple(tys),
        span: Span::new(file_id, start, end),
    }
}

pub fn make_fn_type<'b>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    param_tys: Vec<Type<'b>>,
    return_ty: Option<Type<'b>>,
    end: usize,
) -> Type<'b> {
    let param_tys = bump.alloc_slice_fill_iter(param_tys);
    let return_ty = bump.alloc(return_ty);
    Type {
        kind: TypeKind::Fn(param_tys, return_ty),
        span: Span::new(file_id, start, end),
    }
}

// Definition builder functions

pub fn make_fn_defn<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    name: &'input str,
    params: Vec<DeclInfo<'b>>,
    return_type: Option<Type<'b>>,
    body: Expr<'b>,
    end: usize,
) -> Defn<'b> {
    let name = bump.alloc_str(name);
    let params = bump.alloc_slice_fill_iter(params);
    Defn {
        kind: DefnKind::Fn {
            name,
            params,
            return_type,
            body,
        },
        span: Span::new(file_id, start, end),
    }
}

pub fn make_static_defn<'b, 'input>(
    _: &'b Bump,
    file_id: FileId,
    start: usize,
    decl: DeclInfo<'b>,
    expr: Expr<'b>,
    end: usize,
) -> Defn<'b> {
    Defn {
        kind: DefnKind::Static { decl, expr },
        span: Span::new(file_id, start, end),
    }
}

pub fn make_struct_defn<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    name: Spanned<&'input str>,
    members: Vec<DeclInfo<'b>>,
    end: usize,
) -> Defn<'b> {
    let name = name.map(|s| &*bump.alloc_str(s));
    let members = bump.alloc_slice_fill_iter(members);
    Defn {
        kind: DefnKind::Struct { name, members },
        span: Span::new(file_id, start, end),
    }
}
