use bumpalo::Bump;
use std::{cell::Cell, str::FromStr};

use crate::{
    tyck::{self, VarId},
    util::{FileId, Span},
};

use super::result::{ParseError, ParseResult};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
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
    pub name: Spanned<&'a str>,

    /// The type of the variable, given in tyck phase.
    pub ty: Cell<Option<tyck::Type<'a>>>,

    /// The type ascription AST.
    pub ty_ast: Option<Type<'a>>,

    /// The source span of the declaration.
    pub span: Span,

    /// Unique identifier for this variable, given in tyck phase.
    pub id: Cell<Option<VarId>>,
}

#[derive(Debug)]
pub enum DefnKind<'a> {
    Struct {
        name: Spanned<&'a str>,
        members: &'a [DeclInfo<'a>],
    },
    Fn {
        decl: DeclInfo<'a>,
        params: &'a [DeclInfo<'a>],
        return_ty_ast: &'a Option<Type<'a>>,
        return_ty: Cell<Option<tyck::Type<'a>>>,
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
    Bool(bool),
    Int(&'a str),
    Float(&'a str),
    Str(&'a str),
    Char(&'a str),

    Tuple(&'a [Expr<'a>]),
    Array(&'a [Expr<'a>], Option<Spanned<usize>>),

    Id(&'a str, Cell<Option<VarId>>, Cell<bool>),

    PrefixOp(Spanned<PrefixOpKind>, &'a Expr<'a>),
    BinOp(Spanned<Operator>, &'a Expr<'a>, &'a Expr<'a>),
    Cast(&'a Expr<'a>, Type<'a>),

    Group(&'a Expr<'a>),
    Field(&'a Expr<'a>, Span, Spanned<Field<'a>>, Cell<usize>),
    Call(&'a Expr<'a>, &'a [Expr<'a>]),
    Index(&'a Expr<'a>, &'a Expr<'a>, Cell<usize>),
    Range(Option<&'a Expr<'a>>, Option<&'a Expr<'a>>),
    Block(&'a [Stmt<'a>]),

    Struct(Spanned<&'a str>, &'a [(Spanned<&'a str>, Expr<'a>)]),

    If(&'a [(Expr<'a>, Expr<'a>)], &'a Option<Expr<'a>>),
}

#[derive(Debug)]
pub struct Expr<'a> {
    pub span: Span,
    pub kind: ExprKind<'a>,
    pub ty: Cell<Option<tyck::Type<'a>>>,
}

#[derive(Debug)]
pub enum Operator {
    Simple(BinOpKind),
    Assign(Option<BinOpKind>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOpKind {
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

    Lt,
    LtLt,
    Le,
    Gt,
    GtGt,
    Ge,
    EqEq,
    Ne,
}

#[derive(Debug, Clone, Copy)]
pub enum PrefixOpKind {
    Tilde,
    Bang,

    Plus,
    Minus,

    Star,

    Amp,
    AmpMut,
}

#[derive(Debug)]
pub enum Field<'a> {
    Name(&'a str, Cell<Option<usize>>),
    Index(&'a str),
}

impl<'a> Field<'a> {
    pub fn get_index(&self) -> usize {
        match self {
            Field::Name(_, i) => i.get().expect("index should be there"),
            Field::Index(i) => i.parse().expect("index should be int"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Type<'a> {
    pub kind: TypeKind<'a>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy)]
pub enum TypeKind<'a> {
    Id(&'a str),
    Pointer(bool, &'a Type<'a>),
    Slice(&'a Type<'a>),
    Array(&'a Type<'a>, usize),
    Tuple(&'a [Type<'a>]),
    Fn(&'a [Type<'a>], &'a Option<Type<'a>>),
}

// Expression builder functions

pub fn make_spanned_expr<'b>(
    file_id: FileId,
    kind: ExprKind<'b>,
    start: usize,
    end: usize,
) -> Expr<'b> {
    Expr {
        kind,
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
}

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
        ty: Cell::new(None),
    }
}

pub fn make_prefix_op_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    op: Spanned<PrefixOpKind>,
    e: Expr<'b>,
) -> Expr<'b> {
    let (start, end) = (op.span().start, e.span.end);
    let e = bump.alloc(e);
    Expr {
        kind: ExprKind::PrefixOp(op, e),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
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
        ty: Cell::new(None),
    }
}

pub fn make_cast_expr<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    expr: Expr<'b>,
    _: Spanned<&'input str>,
    ty: Type<'b>,
) -> Expr<'b> {
    let (start, end) = (expr.span.start, ty.span.end);
    let expr = bump.alloc(expr);
    Expr {
        kind: ExprKind::Cast(expr, ty),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
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
    Expr {
        kind: ExprKind::Call(callee, args),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
}

pub fn make_index_expr<'b>(
    bump: &'b Bump,
    file_id: FileId,
    lhs: Expr<'b>,
    index: Expr<'b>,
    end_span: usize,
) -> Expr<'b> {
    let (start, end) = (lhs.span.start, end_span);
    let lhs = bump.alloc(lhs);
    let index = bump.alloc(index);
    Expr {
        kind: ExprKind::Index(lhs, index, Cell::new(0)),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
}

pub fn make_range_expr<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    op: Spanned<&'input str>,
    start: Option<Expr<'b>>,
    end: Option<Expr<'b>>,
) -> Expr<'b> {
    let start = start.map(|x| &*bump.alloc(x));
    let end = end.map(|x| &*bump.alloc(x));
    Expr {
        kind: ExprKind::Range(start, end),
        span: start
            .map(|s| s.span)
            .unwrap_or(op.span())
            .unite(end.map(|e| e.span).unwrap_or(op.span())),
        ty: Cell::new(None),
    }
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
    Expr {
        kind: ExprKind::Block(stmts),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
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
        ty: Cell::new(None),
    }
}

pub fn make_array_expr<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    exprs: Vec<Expr<'b>>,
    size: Option<Spanned<&'input str>>,
    end: usize,
) -> ParseResult<Expr<'b>> {
    let exprs = bump.alloc_slice_fill_iter(exprs);
    let size = match size {
        Some(size) => {
            let parsed_size = usize::from_str(size.item())
                .map_err(|_| ParseError::ArraySizeMustBeNonNegativeInteger { span: size.span() })?;
            Some(Spanned(size.span(), parsed_size))
        }
        None => None,
    };
    Ok(Expr {
        kind: ExprKind::Array(exprs, size),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    })
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
    Expr {
        kind: ExprKind::If(inner_ifs, else_block),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
}

pub fn make_struct_expr<'b, 'input>(
    bump: &'b Bump,
    file_id: FileId,
    start: usize,
    id: Spanned<&'input str>,
    fields: Vec<(Spanned<&'b str>, Expr<'b>)>,
    end: usize,
) -> Expr<'b> {
    let id = id.map(|id| &*bump.alloc_str(id));
    let fields = bump.alloc_slice_fill_iter(fields);
    Expr {
        kind: ExprKind::Struct(id, fields),
        span: Span::new(file_id, start, end),
        ty: Cell::new(None),
    }
}

pub fn make_field_expr<'b, 'input>(
    bump: &'b Bump,
    _: FileId,
    expr: Expr<'b>,
    op: Spanned<&'input str>,
    field: Spanned<Field<'b>>,
) -> Expr<'b> {
    let expr = bump.alloc(expr);
    let field_span = field.span();
    Expr {
        kind: ExprKind::Field(expr, op.span(), field, Cell::new(0)),
        span: expr.span.unite(field_span),
        ty: Cell::new(None),
    }
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
    int: Spanned<&'input str>,
    end: usize,
) -> ParseResult<Type<'b>> {
    let ty = bump.alloc(ty);
    Ok(Type {
        kind: TypeKind::Array(
            ty,
            usize::from_str(int.item())
                .map_err(|_| ParseError::ArraySizeMustBeNonNegativeInteger { span: int.span() })?,
        ),
        span: Span::new(file_id, start, end),
    })
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
    name: Spanned<&'input str>,
    params: Vec<DeclInfo<'b>>,
    return_type: Option<Type<'b>>,
    body: Expr<'b>,
    end: usize,
) -> Defn<'b> {
    let params = bump.alloc_slice_fill_iter(params);
    let param_tys = bump.alloc_slice_fill_iter(
        params
            .iter()
            .map(|decl| decl.ty_ast.expect("param types should be given")),
    );
    let return_type = bump.alloc(return_type);
    let span = Span::new(file_id, start, end);

    let name_span = name.span();
    let name = name.map(|name| &*bump.alloc_str(name));
    let decl = DeclInfo {
        mutable: false,
        name,
        ty: Cell::new(None),
        ty_ast: Some(Type {
            kind: TypeKind::Fn(param_tys, return_type),
            span,
        }),
        span: name_span,
        id: Cell::new(None),
    };

    Defn {
        kind: DefnKind::Fn {
            decl,
            params,
            return_ty_ast: return_type,
            return_ty: Cell::new(None),
            body,
        },
        span,
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
