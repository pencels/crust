pub mod result;

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
    parser::ast::{self, Defn, DefnKind, Expr},
    util::fresh_id,
};

pub const RANGE_TY_NAME: &str = "Range";
pub const RANGE_TO_TY_NAME: &str = "RangeTo";
pub const RANGE_FROM_TY_NAME: &str = "RangeFrom";
pub const RANGE_FULL_TY_NAME: &str = "RangeFull";

enum Source {
    Id(Span),
    Ptr(Span),
    Expr(Span),
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

#[derive(Debug, Clone, Copy)]
pub struct Type<'a, T = Span> {
    pub kind: TypeKind<'a>,
    pub data: T,
}

impl Eq for Type<'_, Span> {}
impl PartialEq for Type<'_, Span> {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
    }
}

#[derive(Debug, Eq, Clone, Copy)]
pub enum TypeKind<'a> {
    Bool,
    Integral(&'a str),
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    USize,
    ISize,
    Float,
    Double,
    Byte,
    Char,
    Str,
    Struct(&'a str),
    Pointer(bool, &'a Type<'a>),
    Slice(&'a Type<'a>),
    Array(&'a Type<'a>, usize),
    Tuple(&'a [Type<'a>]),
    Fn(&'a [Type<'a>], &'a Type<'a>),
}

impl PartialEq for TypeKind<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Struct(l0), Self::Struct(r0)) => l0 == r0,
            (Self::Pointer(l0, l1), Self::Pointer(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Slice(l0), Self::Slice(r0)) => l0 == r0,
            (Self::Array(l0, l1), Self::Array(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Tuple(l0), Self::Tuple(r0)) => l0 == r0,
            (Self::Fn(l0, l1), Self::Fn(r0, r1)) => l0 == r0 && l1 == r1,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl<'a> Display for Type<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.kind {
            TypeKind::Bool => write!(f, "bool"),
            TypeKind::Short => write!(f, "short"),
            TypeKind::UShort => write!(f, "ushort"),
            TypeKind::Integral(..) => write!(f, "{{integer}}"),
            TypeKind::Int => write!(f, "int"),
            TypeKind::UInt => write!(f, "uint"),
            TypeKind::Long => write!(f, "long"),
            TypeKind::ULong => write!(f, "ulong"),
            TypeKind::USize => write!(f, "usize"),
            TypeKind::ISize => write!(f, "isize"),
            TypeKind::Float => write!(f, "float"),
            TypeKind::Double => write!(f, "double"),
            TypeKind::Byte => write!(f, "byte"),
            TypeKind::Char => write!(f, "char"),
            TypeKind::Str => write!(f, "str"),
            TypeKind::Struct(s) => write!(f, "{}", s),
            TypeKind::Pointer(m, ty) => {
                write!(f, "*")?;
                if m {
                    write!(f, "mut ")?;
                }
                write!(f, "{}", ty)
            }
            TypeKind::Slice(ty) => write!(f, "[{}]", ty),
            TypeKind::Array(ty, size) => write!(f, "[{}; {}]", ty, size),
            TypeKind::Tuple([]) => write!(f, "()"),
            TypeKind::Tuple([ty]) => write!(f, "({},)", ty),
            TypeKind::Tuple([ty, tys @ ..]) => {
                write!(f, "({}", ty)?;
                for ty in tys {
                    write!(f, ", {}", ty)?;
                }
                write!(f, ")")
            }
            TypeKind::Fn([], rty) => write!(f, "fn() -> {}", rty),
            TypeKind::Fn([ty, tys @ ..], rty) => {
                write!(f, "fn({}", ty)?;
                for ty in tys {
                    write!(f, ", {}", ty)?;
                }
                write!(f, ") -> {}", rty)
            }
        }
    }
}

pub fn human_type_name(ty: &Type) -> String {
    match ty.kind {
        TypeKind::Struct(ty) => format!("struct '{}'", ty),
        _ => format!("'{}'", ty),
    }
}

impl<'a> Type<'a> {
    pub const UNIT: Type<'static> = Type {
        kind: TypeKind::Tuple(&[]),
        data: Span::dummy(),
    };

    pub const USIZE: Type<'static> = Type {
        kind: TypeKind::USize,
        data: Span::dummy(),
    };

    pub const ISIZE: Type<'static> = Type {
        kind: TypeKind::ISize,
        data: Span::dummy(),
    };

    pub const INT: Type<'static> = Type {
        kind: TypeKind::Int,
        data: Span::dummy(),
    };

    pub const CHAR: Type<'static> = Type {
        kind: TypeKind::Char,
        data: Span::dummy(),
    };

    pub const BOOL: Type<'static> = Type {
        kind: TypeKind::Bool,
        data: Span::dummy(),
    };

    pub fn is_unit(&self) -> bool {
        match self.kind {
            TypeKind::Tuple([]) => true,
            _ => false,
        }
    }

    pub fn new<'b>(kind: TypeKind<'b>, span: Span) -> Type<'b> {
        Type { kind, data: span }
    }

    pub fn new_dummy<'b>(kind: TypeKind<'b>) -> Type<'b> {
        Type {
            kind,
            data: Span::dummy(),
        }
    }

    pub fn ptr<'b>(m: bool, ty: &'b Type<'b>, span: Span) -> Type {
        Type {
            kind: TypeKind::Pointer(m, ty),
            data: span,
        }
    }

    pub fn is_int(&self) -> bool {
        match self.kind {
            TypeKind::Byte
            | TypeKind::Char
            | TypeKind::Integral(_)
            | TypeKind::Short
            | TypeKind::UShort
            | TypeKind::Int
            | TypeKind::UInt
            | TypeKind::Long
            | TypeKind::ULong
            | TypeKind::USize
            | TypeKind::ISize => true,
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self.kind {
            TypeKind::Float | TypeKind::Double => true,
            _ => false,
        }
    }

    pub fn is_sized_ptr(&self) -> bool {
        match self.kind {
            TypeKind::Pointer(_, t) if t.is_sized() => true,
            _ => false,
        }
    }

    pub fn is_slice_ptr(&self) -> bool {
        match self.kind {
            TypeKind::Pointer(_, t) if !t.is_sized() => true,
            _ => false,
        }
    }

    pub fn is_int_or_float(&self) -> bool {
        match self.kind {
            TypeKind::Byte
            | TypeKind::Integral(_)
            | TypeKind::Short
            | TypeKind::UShort
            | TypeKind::Int
            | TypeKind::UInt
            | TypeKind::Long
            | TypeKind::ULong
            | TypeKind::USize
            | TypeKind::ISize
            | TypeKind::Float
            | TypeKind::Double
            | TypeKind::Char => true,
            _ => false,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.is_int_or_float() || self.is_sized_ptr() || self == &Type::BOOL
    }

    pub fn is_sized(&self) -> bool {
        match self.kind {
            TypeKind::Slice(_) | TypeKind::Str => false,
            _ => true,
        }
    }

    pub fn is_signed(&self) -> Option<bool> {
        use TypeKind::*;
        match self.kind {
            Byte | Short | Int | Long | ISize => Some(true),
            Char | UShort | UInt | ULong | USize => Some(false),
            _ => None,
        }
    }

    pub fn bit_width(&self) -> Option<usize> {
        use TypeKind::*;
        Some(match self.kind {
            Bool => 1,
            Byte | Char => 8,
            Short | UShort => 16,
            Int | UInt | Float => 32,
            Long | ULong | Double => 64,
            USize | ISize | Pointer(..) => 64, // HACK: hmmm
            _ => return None,
        })
    }

    pub fn pointee_ty(&self) -> Option<&Type> {
        match self.kind {
            TypeKind::Pointer(_, t) => Some(t),
            _ => None,
        }
    }

    pub fn from<'b>(checker: &TypeChecker<'b>, ty: &ast::Type) -> TyckResult<Type<'b>> {
        let kind = match &ty.kind {
            ast::TypeKind::Id("bool") => TypeKind::Bool,
            ast::TypeKind::Id("short") => TypeKind::Short,
            ast::TypeKind::Id("ushort") => TypeKind::UShort,
            ast::TypeKind::Id("int") => TypeKind::Int,
            ast::TypeKind::Id("uint") => TypeKind::UInt,
            ast::TypeKind::Id("long") => TypeKind::Long,
            ast::TypeKind::Id("ulong") => TypeKind::ULong,
            ast::TypeKind::Id("usize") => TypeKind::USize,
            ast::TypeKind::Id("isize") => TypeKind::ISize,
            ast::TypeKind::Id("float") => TypeKind::Float,
            ast::TypeKind::Id("double") => TypeKind::Double,
            ast::TypeKind::Id("byte") => TypeKind::Byte,
            ast::TypeKind::Id("char") => TypeKind::Char,
            ast::TypeKind::Id("str") => TypeKind::Str,
            ast::TypeKind::Id(other) => {
                if checker.struct_tys.contains_key(other) {
                    TypeKind::Struct(checker.bump.alloc_str(other))
                } else {
                    return Err(TyckError::UndefinedType {
                        name: other.to_string(),
                        span: ty.span,
                    });
                }
            }
            ast::TypeKind::Pointer(m, inner) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                TypeKind::Pointer(*m, inner)
            }
            ast::TypeKind::Slice(inner) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                TypeKind::Slice(inner)
            }
            ast::TypeKind::Array(inner, size) => {
                let inner = checker.bump.alloc(Type::from(checker, inner)?);
                TypeKind::Array(inner, *size)
            }
            ast::TypeKind::Tuple([]) => TypeKind::Tuple(&[]),
            ast::TypeKind::Tuple(inners) => {
                let inners: TyckResult<Vec<_>> = inners
                    .iter()
                    .map(|inner| Type::from(checker, inner))
                    .collect();
                let inners = checker.bump.alloc_slice_fill_iter(inners?);
                TypeKind::Tuple(inners)
            }
            ast::TypeKind::Fn(params, return_ty) => {
                let params: TyckResult<Vec<_>> = params
                    .iter()
                    .map(|param| Type::from(checker, param))
                    .collect();
                let params = checker.bump.alloc_slice_fill_iter(params?);
                let return_ty = checker
                    .bump
                    .alloc(return_ty.map_or(Ok(Type::UNIT), |t| Type::from(checker, &t))?);
                TypeKind::Fn(params, return_ty)
            }
        };

        Ok(Type {
            kind,
            data: ty.span,
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
    if can_coerce(rhs_ty, lhs_ty) {
        // Catch unsized types.
        if !rhs_ty.is_sized() {
            return Err(TyckError::CannotAssignUnsized {
                span: rhs_ty_span,
                ty_name: human_type_name(&rhs_ty),
            });
        }
        if !lhs_ty.is_sized() {
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

fn tyck_is_int_value(Spanned(ty_span, ty): Spanned<&Type>) -> TyckResult<()> {
    match ty.kind {
        TypeKind::Bool
        | TypeKind::Integral(_)
        | TypeKind::Short
        | TypeKind::UShort
        | TypeKind::Int
        | TypeKind::UInt
        | TypeKind::Long
        | TypeKind::ULong
        | TypeKind::USize
        | TypeKind::ISize
        | TypeKind::Byte
        | TypeKind::Char => Ok(()),
        _ => Err(TyckError::MismatchedTypes {
            expected_ty: "an integer value type".to_string(),
            got: ty_span,
            got_ty: human_type_name(ty),
        }),
    }
}

fn tyck_is_scalar_value(Spanned(ty_span, ty): Spanned<&Type>) -> TyckResult<()> {
    match ty.kind {
        TypeKind::Bool
        | TypeKind::Integral(_)
        | TypeKind::Short
        | TypeKind::UShort
        | TypeKind::Int
        | TypeKind::UInt
        | TypeKind::Long
        | TypeKind::ULong
        | TypeKind::USize
        | TypeKind::ISize
        | TypeKind::Float
        | TypeKind::Double
        | TypeKind::Byte
        | TypeKind::Char => Ok(()),
        _ => Err(TyckError::MismatchedTypes {
            expected_ty: "a scalar value type".to_string(),
            got: ty_span,
            got_ty: human_type_name(ty),
        }),
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

fn autoderef_ptr<'alloc>(mut ty: Type<'alloc>) -> (usize, Type<'alloc>) {
    let mut num_derefs = 0;
    loop {
        match ty.kind {
            TypeKind::Array(..) => break,
            TypeKind::Pointer(
                _,
                Type {
                    kind: TypeKind::Slice(_) | TypeKind::Str,
                    ..
                },
            ) => break,
            TypeKind::Pointer(_, inner) => {
                num_derefs += 1;
                ty = *inner;
            }
            _ => break,
        }
    }
    (num_derefs, ty)
}

fn autoderef_place_ptr<'alloc>(mut ty: Type<'alloc>) -> (usize, bool, Type<'alloc>) {
    let mut num_derefs = 0;
    let mut ptr_mut = true;
    loop {
        match ty.kind {
            TypeKind::Pointer(
                _,
                Type {
                    kind: TypeKind::Slice(_) | TypeKind::Str,
                    ..
                },
            ) => break,
            TypeKind::Pointer(m, inner) => {
                num_derefs += 1;
                ty = *inner;
                ptr_mut &= m;
            }
            _ => break,
        }
    }
    (num_derefs, ptr_mut, ty)
}

pub struct TypeChecker<'alloc> {
    pub var_decl: HashMap<VarId, &'alloc DeclInfo<'alloc>>,
    pub struct_tys: HashMap<&'alloc str, Option<StructInfo<'alloc>>>,
    pub func_decl: HashMap<&'alloc str, &'alloc DeclInfo<'alloc>>,
    pub bump: &'alloc Bump,
    pub env: Env<'alloc>,
    pub return_ty: Option<Type<'alloc>>,
}

impl<'check, 'alloc> TypeChecker<'alloc> {
    pub fn new(bump: &'alloc Bump) -> TypeChecker<'alloc> {
        TypeChecker {
            var_decl: hashmap! {},
            struct_tys: hashmap! {},
            func_decl: hashmap! {},
            bump,
            env: Env::new(),
            return_ty: None,
        }
    }

    pub fn tyck_program(&mut self, program: &'alloc Vec<Defn<'alloc>>) -> TyckResult<()> {
        // Register all the definition declarations first before doing a full typecheck.
        for defn in program {
            match &defn.kind {
                DefnKind::Struct { name, .. } => {
                    let name = name.map(|name| &*self.bump.alloc_str(name));
                    self.struct_tys.insert(name.item(), None);
                }
                DefnKind::Fn { decl, .. } => {
                    decl.set_ty(Type::from(
                        self,
                        &decl.ty_ast.expect("fn decls should have types"),
                    )?)?;
                    self.func_decl.insert(decl.name.item(), decl);
                }
                DefnKind::Static { decl, expr } => {
                    let expr_ty = self.tyck_expr(expr)?;
                    decl.set_ty(expr_ty)?;
                    self.register_decl(decl);
                }
                DefnKind::ExternFn {
                    decl,
                    params,
                    return_ty_ast,
                    return_ty,
                    ..
                } => {
                    decl.set_ty(Type::from(
                        self,
                        &decl.ty_ast.expect("fn decls should have types"),
                    )?)?;
                    for param in *params {
                        param.set_ty(Type::from(
                            self,
                            &param.ty_ast.expect("params should always have a type"),
                        )?)?;
                    }
                    return_ty.set(Some(
                        return_ty_ast
                            .map(|ty| Type::from(self, &ty))
                            .unwrap_or(Ok(Type::UNIT))?,
                    ));
                    self.func_decl.insert(decl.name.item(), decl);
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
                        mem.set_ty(Type::from(
                            self,
                            &mem.ty_ast.expect("struct members should have types"),
                        )?)?;
                        Ok(mem)
                    })
                    .collect();
                let members = self.bump.alloc_slice_fill_iter(members?);
                self.struct_tys.insert(
                    name.item(),
                    Some(StructInfo {
                        name: self.bump.alloc(name),
                        members,
                    }),
                );
                Ok(())
            }
            DefnKind::Fn {
                decl,
                params,
                return_ty_ast: return_type,
                return_ty,
                body,
                ..
            } => self.tyck_fn(decl, params, return_type, &return_ty, body),
            DefnKind::Static { .. } => todo!("static definitions"),
            DefnKind::ExternFn { .. } => {
                // Extern fns have no body to typecheck.
                Ok(())
            }
        }
    }

    /// Registers a declaration in the current environment frame.
    fn register_decl(&mut self, decl: &'alloc DeclInfo<'alloc>) {
        let id = fresh_id();
        decl.id.set(Some(id));
        self.var_decl.insert(id, decl);
        self.env.set(decl.name.item(), id);
    }

    fn tyck_fn(
        &mut self,
        decl: &'alloc DeclInfo<'alloc>,
        params: &'alloc [DeclInfo<'alloc>],
        return_ty_ast: &'alloc Option<ast::Type<'alloc>>,
        return_ty_cell: &Cell<Option<Type<'alloc>>>,
        body: &'alloc Expr<'alloc>,
    ) -> TyckResult<()> {
        self.env.push();
        for param in params {
            param.set_ty(Type::from(
                self,
                &param.ty_ast.expect("fn params always have a type"),
            )?)?;
            self.register_decl(param);
        }

        let return_ty = return_ty_ast.map_or(
            Ok(Type {
                kind: TypeKind::Tuple(&[]),
                data: decl.span,
            }),
            |t| Type::from(self, &t),
        )?;
        self.return_ty = Some(return_ty);

        let ExprKind::Block(stmts) = body.kind else {
            panic!("ICE: fn body is not a block");
        };

        if let Some((last, init)) = stmts.split_last() {
            for stmt in init {
                self.tyck_stmt(stmt)?;
            }

            // Make sure end of function body returns a value that makes sense
            match &last.kind {
                StmtKind::Let(_, _)
                | StmtKind::Semi(_)
                | StmtKind::While(_, _)
                | StmtKind::Return(None) => {
                    self.tyck_stmt(last)?;
                    if !return_ty.is_unit() {
                        return Err(TyckError::NotAllPathsReturnAValue {
                            ret_ty_span: return_ty.data,
                            last_stmt_span: last.span,
                        });
                    }
                }
                StmtKind::Return(Some(e)) | StmtKind::Expr(e) => {
                    self.coerce(e, &return_ty)?;
                }
            }
        } else {
            if return_ty.is_unit() {
                self.coerce(
                    body,
                    &Type {
                        kind: TypeKind::Tuple(&[]),
                        data: return_ty_ast.map_or(decl.name.span(), |ast| ast.span),
                    },
                )?;
            } else {
                return Err(TyckError::NotAllPathsReturnAValue {
                    ret_ty_span: return_ty.data,
                    last_stmt_span: body.span,
                });
            }
        }

        self.return_ty = None;
        self.env.pop();

        return_ty_cell.set(Some(return_ty));
        Ok(())
    }

    /// Attempt to coerce an expression to a type.
    fn coerce(&mut self, expr: &Expr<'alloc>, to_ty: &Type<'alloc>) -> TyckResult<Type<'alloc>> {
        let from_ty = match expr.ty.get() {
            Some(ty) => ty,
            None => self.tyck_expr(expr)?,
        };

        // Do recursive coercion site analysis where applicable
        match (&expr.kind, to_ty.kind) {
            (ExprKind::Tuple(exprs), TypeKind::Tuple(tys)) => {
                // Each expression in the tuple literal must be coerceable into its corresponding type in the tuple type
                for (expr, target) in exprs.iter().zip(tys) {
                    self.coerce(expr, target)?;
                }
            }
            (ExprKind::Array(exprs, expr_size), TypeKind::Array(ty, ty_size)) => {
                // Each expression in the array literal must be coerceable into the array element type
                for expr in *exprs {
                    self.coerce(expr, ty)?;
                }

                // If the array sizes are incompatible
                let expr_size = match expr_size {
                    Some(Spanned(_, size)) => *size,
                    None => exprs.len(),
                };

                if expr_size != ty_size {
                    return Err(TyckError::InvalidArraySize {
                        size_span: ty.data,
                        size: ty_size,
                        elements_span: expr.span,
                        num_elements: expr_size,
                    });
                }
            }
            (ExprKind::Group(e), _) => {
                self.coerce(e, to_ty)?;
            }
            (ExprKind::Block(stmts), _) => {
                for stmt in stmts.iter() {
                    // Type check middle return statements in the block, should match the return type
                    match stmt {
                        Stmt {
                            kind: StmtKind::Return(ret_expr),
                            ..
                        } => {
                            let ret_ty = self.return_ty.unwrap();
                            match ret_expr {
                                Some(expr) => {
                                    self.coerce(expr, &ret_ty)?;
                                }
                                None => {
                                    if !ret_ty.is_unit() {
                                        return Err(TyckError::CannotCoerceType {
                                            expr_span: stmt.span,
                                            ty_span: to_ty.data,
                                            from_ty_name: human_type_name(&from_ty),
                                            to_ty_name: human_type_name(&to_ty),
                                        });
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                match stmts.last() {
                    // Empty blocks or blocks ending with a non-expr stmt
                    Some(Stmt {
                        kind: StmtKind::Let(..) | StmtKind::Semi(_) | StmtKind::While(..),
                        ..
                    })
                    | None => {
                        if !to_ty.is_unit() {
                            return Err(TyckError::CannotCoerceType {
                                expr_span: expr.span,
                                ty_span: to_ty.data,
                                from_ty_name: human_type_name(&from_ty),
                                to_ty_name: human_type_name(&to_ty),
                            });
                        }
                    }
                    Some(Stmt {
                        kind: StmtKind::Expr(e),
                        ..
                    }) => {
                        self.coerce(e, to_ty)?;
                    }
                    _ => {}
                }
            }
            (ExprKind::If(init, Some(last)), _) => {
                for (_, expr) in *init {
                    self.coerce(expr, to_ty)?;
                }
                self.coerce(last, to_ty)?;
            }
            _ => {
                if !can_coerce(&from_ty, to_ty) {
                    return Err(TyckError::CannotCoerceType {
                        expr_span: expr.span,
                        ty_span: to_ty.data,
                        from_ty_name: human_type_name(&from_ty),
                        to_ty_name: human_type_name(&to_ty),
                    });
                }
            }
        };

        expr.coerced_ty.set(Some(*to_ty));

        Ok(*to_ty)
    }

    fn tyck_expr(&mut self, expr: &Expr<'alloc>) -> TyckResult<Type<'alloc>> {
        let ty = match &expr.kind {
            ExprKind::Bool(_) => Ok(Type::new(TypeKind::Bool, expr.span)),
            ExprKind::Int(i) => Ok(Type::new(TypeKind::Integral(i), expr.span)),
            ExprKind::Float(_) => Ok(Type::new(TypeKind::Double, expr.span)),
            ExprKind::Str(_) => Ok(Type::ptr(
                false,
                self.bump.alloc(Type::new(TypeKind::Str, expr.span)),
                expr.span,
            )),
            ExprKind::Char(_) => Ok(Type::new(TypeKind::Char, expr.span)),
            ExprKind::Tuple(exprs) => {
                let mut tys = Vec::new();
                for expr in *exprs {
                    tys.push(self.tyck_expr(expr)?);
                }
                let tys = self.bump.alloc_slice_fill_iter(tys);
                Ok(Type::new(TypeKind::Tuple(tys), expr.span))
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

                    Ok(Type::new(
                        TypeKind::Array(self.bump.alloc(ty), size),
                        expr.span,
                    ))
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
            ExprKind::Cast(expr, to_ty_ast) => {
                let from_ty = self.tyck_expr(expr)?;
                let to_ty = Type::from(self, to_ty_ast)?;
                self.tyck_cast(Spanned(expr.span, from_ty), Spanned(to_ty_ast.span, to_ty))
            }
            ExprKind::Call(callee, args) => {
                let callee_ty = self.tyck_expr(callee)?;
                match callee_ty.kind {
                    TypeKind::Fn(param_tys, rty) => {
                        if param_tys.len() == args.len() {
                            for (param_ty, arg) in param_tys.iter().zip(*args) {
                                self.coerce(arg, param_ty)?;
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
                for (cond, then) in thens.iter() {
                    self.coerce(cond, &Type::new(TypeKind::Bool, cond.span))?;

                    let then_ty = self.tyck_expr(then)?;
                    results.push(then_ty);
                }

                let common_ty = match els {
                    Some(els) => {
                        let else_ty = self.tyck_expr(els)?;
                        results.push(else_ty);
                        match results.iter().fold(Some(&results[0]), |acc, x| {
                            acc.and_then(|acc| common_type(acc, x))
                        }) {
                            Some(t) => t,
                            None => {
                                return Err(TyckError::MismatchedIfBranchTypes { span: expr.span })
                            }
                        }
                    }
                    None => &Type::UNIT,
                };

                for (_, then) in thens.iter() {
                    self.coerce(then, common_ty)?;
                }
                if let Some(els) = els {
                    self.coerce(els, common_ty)?;
                }

                Ok(*common_ty)
            }
            ExprKind::Struct(Spanned(struct_name_span, struct_name), fields) => {
                let info = match self
                    .struct_tys
                    .get(struct_name)
                    .map(|i| i.as_ref())
                    .flatten()
                {
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
                            self.coerce(
                                expr,
                                &decl.ty.get().expect("struct field type unresolved"),
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
                Ok(Type::new(TypeKind::Struct(struct_name), expr.span))
            }
            ExprKind::Field(expr, _op_span, field, autoderef) => {
                let expr_ty = self.tyck_expr(expr)?;
                let (n, autoderef_ty) = autoderef_ptr(expr_ty);
                autoderef.set(Some((n, autoderef_ty)));
                self.field_access_ty(Spanned(expr.span, autoderef_ty), field)
            }
            ExprKind::Group(expr) => self.tyck_expr(expr),
            ExprKind::Index(lhs, index, autoderef) => {
                let lhs_ty = self.tyck_expr(lhs)?;
                let index_ty = self.tyck_expr(index)?;
                let (n, autoderef_ty) = autoderef_ptr(lhs_ty);
                autoderef.set(Some((n, autoderef_ty)));
                self.tyck_index(lhs, autoderef_ty, index, index_ty)
            }
            ExprKind::Range(start, end) => {
                if let Some(start) = start {
                    self.coerce(
                        start,
                        &Type {
                            kind: TypeKind::USize,
                            data: start.span,
                        },
                    )?;
                }

                if let Some(end) = end {
                    self.coerce(
                        end,
                        &Type {
                            kind: TypeKind::USize,
                            data: end.span,
                        },
                    )?;
                }

                Ok(Type::new(
                    TypeKind::Struct(match (start, end) {
                        (None, None) => RANGE_FULL_TY_NAME,
                        (Some(_), None) => RANGE_FROM_TY_NAME,
                        (None, Some(_)) => RANGE_TO_TY_NAME,
                        _ => RANGE_TY_NAME,
                    }),
                    expr.span,
                ))
            }
        }?;

        expr.ty.set(Some(ty));

        Ok(ty)
    }

    /// Retrieves the type that is indexed by the given index.
    fn tyck_index(
        &mut self,
        lhs: &Expr<'alloc>,
        autoderef_lhs_ty: Type<'alloc>,
        index: &Expr<'alloc>,
        index_ty: Type<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let (elem_ty, slice_ty) = match autoderef_lhs_ty.kind {
            TypeKind::Str => (&Type::CHAR, autoderef_lhs_ty),
            TypeKind::Slice(ty) => (ty, autoderef_lhs_ty),
            TypeKind::Pointer(
                _,
                s @ Type {
                    kind: TypeKind::Slice(ty),
                    ..
                },
            ) => (*ty, *s),
            TypeKind::Pointer(
                _,
                s @ Type {
                    kind: TypeKind::Str,
                    ..
                },
            ) => (&Type::CHAR, *s),
            TypeKind::Pointer(
                _,
                Type {
                    kind: TypeKind::Array(ty, _),
                    ..
                },
            ) => (
                *ty,
                Type {
                    kind: TypeKind::Slice(ty),
                    data: autoderef_lhs_ty.data,
                },
            ),
            TypeKind::Array(ty, _) => (
                ty,
                Type {
                    kind: TypeKind::Slice(ty),
                    data: autoderef_lhs_ty.data,
                },
            ),
            _ => {
                return Err(TyckError::TypeNotIndexable {
                    span: lhs.span,
                    ty_name: human_type_name(&autoderef_lhs_ty),
                })
            }
        };

        if index_ty.is_int() {
            self.coerce(
                index,
                &Type {
                    kind: TypeKind::USize,
                    data: index.span,
                },
            )?;
            Ok(*elem_ty)
        } else if let TypeKind::Struct(
            RANGE_TY_NAME | RANGE_FROM_TY_NAME | RANGE_TO_TY_NAME | RANGE_FULL_TY_NAME,
        ) = index_ty.kind
        {
            Ok(slice_ty)
        } else {
            Err(TyckError::TypeCannotBeUsedAsAnIndex {
                span: index.span,
                ty_name: human_type_name(&index_ty),
            })
        }
    }

    fn tyck_cast(
        &self,
        Spanned(from_ty_span, from_ty): Spanned<Type<'alloc>>,
        Spanned(to_ty_span, to_ty): Spanned<Type<'alloc>>,
    ) -> TyckResult<Type<'alloc>> {
        // If the types can be coerced then it's a trivial cast
        if can_coerce(&from_ty, &to_ty) {
            return Ok(to_ty);
        }

        // Numeric cast
        if from_ty.is_int_or_float() && to_ty.is_int_or_float() {
            return Ok(to_ty);
        }

        // Primitive to int cast
        if from_ty == Type::BOOL && to_ty.is_int() {
            return Ok(to_ty);
        }

        // Pointer to address cast
        if let TypeKind::Pointer(..) = from_ty.kind && to_ty.is_int() {
            return Ok(to_ty);
        }
        if let TypeKind::Fn(..) = from_ty.kind && to_ty.is_int() {
            return Ok(to_ty);
        }

        // Address to pointer cast
        if let TypeKind::Pointer(..) = to_ty.kind && from_ty.is_int() {
            return Ok(to_ty);
        }
        if let TypeKind::Fn(..) = to_ty.kind && from_ty.is_int() {
            return Ok(to_ty);
        }

        match (from_ty.kind, to_ty.kind) {
            (TypeKind::Pointer(_, from), TypeKind::Pointer(_, to)) => {
                if from.is_sized() && to.is_sized() || from == to {
                    return Ok(to_ty);
                }

                // Handle pointers to unsized data
                match (from.kind, to.kind) {
                    (TypeKind::Slice(x), y) if x.kind == y => Ok(to_ty),
                    (TypeKind::Str, _) | (_, TypeKind::Str) | (_, TypeKind::Slice(..)) => {
                        Err(TyckError::CannotCastType {
                            expr_span: from_ty_span,
                            ty_span: to_ty_span,
                            from_ty_name: human_type_name(&from_ty),
                            to_ty_name: human_type_name(&to_ty),
                        })
                    }
                    _ => Ok(to_ty),
                }
            }
            _ => Err(TyckError::CannotCastType {
                expr_span: from_ty_span,
                ty_span: to_ty_span,
                from_ty_name: human_type_name(&from_ty),
                to_ty_name: human_type_name(&to_ty),
            }),
        }
    }

    fn tyck_var(
        &self,
        name: &str,
        id: &Cell<Option<usize>>,
        decl_cell: &Cell<Option<&'alloc DeclInfo<'alloc>>>,
        span: Span,
    ) -> TyckResult<&'alloc DeclInfo<'alloc>> {
        if let Some(lookup_id) = self.env.get(name) {
            // Inject id from looking up in the env
            id.set(Some(lookup_id));
            let decl = self
                .var_decl
                .get(&lookup_id)
                .expect("ICE: var id in env was not in the var_decl map");
            decl_cell.set(Some(decl));
            Ok(decl)
        } else if let Some(decl) = self.func_decl.get(name) {
            decl_cell.set(Some(decl));
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

        if !lhs_ty.is_sized() {
            return Err(TyckError::CannotAssignUnsized {
                span: lhs.span,
                ty_name: human_type_name(&lhs_ty),
            });
        }

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
                _ => {
                    return Err(TyckError::MutatingImmutableValueOfUnknownCause { span: lhs.span })
                }
            }
        }

        self.coerce(rhs, &lhs_ty)?;

        let Spanned(op_span, op) = op;
        if let Some(op) = op {
            self.tyck_simple_binop_expr(Spanned(op_span, op), lhs, rhs)?;
        }

        Ok(Type::UNIT)
    }

    fn tyck_simple_binop_expr(
        &mut self,
        op: Spanned<BinOpKind>,
        lhs: &'alloc Expr<'alloc>,
        rhs: &'alloc Expr<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let lhs_ty = self.tyck_expr(lhs)?;
        let rhs_ty = self.tyck_expr(rhs)?;

        match op.item() {
            BinOpKind::Plus => {
                if lhs_ty.is_int_or_float() && rhs_ty.is_int_or_float() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(*common_ty)
                        }
                        None => Err(TyckError::NoCommonType {
                            lhs: human_type_name(&lhs_ty),
                            lhs_span: lhs.span,
                            rhs: human_type_name(&rhs_ty),
                            rhs_span: rhs.span,
                        }),
                    }
                } else if lhs_ty.is_sized_ptr() && rhs_ty.is_int() {
                    self.coerce(rhs, &Type::ISIZE)?;
                    Ok(lhs_ty)
                } else if rhs_ty.is_sized_ptr() && lhs_ty.is_int() {
                    self.coerce(lhs, &Type::ISIZE)?;
                    Ok(rhs_ty)
                } else {
                    Err(TyckError::IncorrectTypesForBinaryOperator {
                        op: BinOpKind::Plus,
                        span: lhs.span.unite(rhs.span),
                        lhs: human_type_name(&lhs_ty),
                        rhs: human_type_name(&rhs_ty),
                    })
                }
            }
            BinOpKind::Minus => {
                if lhs_ty.is_int_or_float() && rhs_ty.is_int_or_float() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(*common_ty)
                        }
                        None => Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&lhs_ty),
                            got: rhs.span,
                            got_ty: human_type_name(&rhs_ty),
                        }),
                    }
                } else if lhs_ty.is_sized_ptr() && rhs_ty.is_int() {
                    self.coerce(rhs, &Type::ISIZE)?;
                    Ok(lhs_ty)
                } else if lhs_ty.is_sized_ptr() && rhs_ty.is_sized_ptr() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(Type::ISIZE)
                        }
                        None => Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&lhs_ty),
                            got: rhs.span,
                            got_ty: human_type_name(&rhs_ty),
                        }),
                    }
                } else {
                    Err(TyckError::IncorrectTypesForBinaryOperator {
                        op: *op.item(),
                        span: lhs.span.unite(rhs.span),
                        lhs: human_type_name(&lhs_ty),
                        rhs: human_type_name(&rhs_ty),
                    })
                }
            }

            BinOpKind::Caret | BinOpKind::Amp | BinOpKind::Pipe => {
                if lhs_ty.is_int() && rhs_ty.is_int() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(*common_ty)
                        }
                        None => Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&lhs_ty),
                            got: rhs.span,
                            got_ty: human_type_name(&rhs_ty),
                        }),
                    }
                } else if lhs_ty == Type::BOOL && rhs_ty == Type::BOOL {
                    Ok(Type::BOOL)
                } else {
                    Err(TyckError::IncorrectTypesForBinaryOperator {
                        op: *op.item(),
                        span: lhs.span.unite(rhs.span),
                        lhs: human_type_name(&lhs_ty),
                        rhs: human_type_name(&rhs_ty),
                    })
                }
            }

            BinOpKind::LtLt | BinOpKind::GtGt => {
                if lhs_ty.is_int() && rhs_ty.is_int() {
                    self.coerce(rhs, &Type::USIZE)?;
                    Ok(lhs_ty)
                } else if !lhs_ty.is_int() {
                    Err(TyckError::MismatchedTypes {
                        expected_ty: "an integer type".to_owned(),
                        got: lhs.span,
                        got_ty: human_type_name(&lhs_ty),
                    })
                } else {
                    Err(TyckError::MismatchedTypes {
                        expected_ty: human_type_name(&Type::USIZE),
                        got: rhs.span,
                        got_ty: human_type_name(&rhs_ty),
                    })
                }
            }

            BinOpKind::AmpAmp | BinOpKind::PipePipe => {
                self.coerce(lhs, &Type::BOOL)?;
                self.coerce(rhs, &Type::BOOL)?;
                Ok(Type::BOOL)
            }

            BinOpKind::Star | BinOpKind::Slash | BinOpKind::Percent => {
                if lhs_ty.is_int_or_float() && rhs_ty.is_int_or_float() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(*common_ty)
                        }
                        None => Err(TyckError::NoCommonType {
                            lhs: human_type_name(&lhs_ty),
                            lhs_span: lhs.span,
                            rhs: human_type_name(&rhs_ty),
                            rhs_span: rhs.span,
                        }),
                    }
                } else {
                    Err(TyckError::IncorrectTypesForBinaryOperator {
                        op: *op.item(),
                        span: lhs.span.unite(rhs.span),
                        lhs: human_type_name(&lhs_ty),
                        rhs: human_type_name(&rhs_ty),
                    })
                }
            }

            BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge
            | BinOpKind::EqEq
            | BinOpKind::Ne => {
                if lhs_ty.is_scalar() && rhs_ty.is_scalar() {
                    match common_type(&lhs_ty, &rhs_ty) {
                        Some(common_ty) => {
                            self.coerce(lhs, &common_ty)?;
                            self.coerce(rhs, &common_ty)?;
                            Ok(*common_ty)
                        }
                        None => Err(TyckError::MismatchedTypes {
                            expected_ty: human_type_name(&lhs_ty),
                            got: rhs.span,
                            got_ty: human_type_name(&rhs_ty),
                        }),
                    }
                } else {
                    Err(TyckError::IncorrectTypesForBinaryOperator {
                        op: *op.item(),
                        span: lhs.span.unite(rhs.span),
                        lhs: human_type_name(&lhs_ty),
                        rhs: human_type_name(&rhs_ty),
                    })
                }
            }
        }
    }

    fn tyck_prefix_op_expr(
        &mut self,
        Spanned(op_span, op): Spanned<PrefixOpKind>,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<Type<'alloc>> {
        let ty = self.tyck_expr(expr)?;
        match op {
            PrefixOpKind::Len => {
                if ty.is_slice_ptr() {
                    Ok(Type {
                        kind: TypeKind::USize,
                        data: op_span,
                    })
                } else {
                    Err(TyckError::MismatchedTypes {
                        expected_ty: "a slice type".to_string(),
                        got: op_span,
                        got_ty: human_type_name(&ty),
                    })
                }
            }
            PrefixOpKind::Tilde => {
                tyck_is_int_value(Spanned(expr.span, &ty))?;
                Ok(ty)
            }
            PrefixOpKind::Bang => {
                self.coerce(expr, &Type::BOOL)?;
                Ok(Type::new(TypeKind::Bool, op_span.unite(expr.span)))
            }
            PrefixOpKind::Plus => {
                tyck_is_scalar_value(Spanned(expr.span, &ty))?;
                Ok(ty)
            }
            PrefixOpKind::Minus => {
                tyck_is_scalar_value(Spanned(expr.span, &ty))?;
                Ok(ty)
            }
            PrefixOpKind::Star => match ty.kind {
                TypeKind::Pointer(_, of) => Ok(*of),
                _ => Err(TyckError::DereferencingNonPointer { span: expr.span }),
            },
            PrefixOpKind::Amp => {
                let (_, _, expr_ty) = self.tyck_place_expr(expr)?;
                Ok(Type::new(
                    TypeKind::Pointer(false, self.bump.alloc(expr_ty)),
                    op_span.unite(expr.span),
                ))
            }
            PrefixOpKind::AmpMut => {
                let (mutable, source, expr_ty) = self.tyck_place_expr(expr)?;
                if mutable {
                    Ok(Type::new(
                        TypeKind::Pointer(true, self.bump.alloc(expr_ty)),
                        op_span.unite(expr.span),
                    ))
                } else {
                    match source {
                        Some(Source::Id(s) | Source::Ptr(s)) => {
                            Err(TyckError::MutPointerToNonMutLocation {
                                addressing_op: op_span,
                                location: expr.span,
                                source: s,
                            })
                        }
                        Some(Source::Expr(s)) => Err(TyckError::MutPointerToNonMutExpr {
                            addressing_op: op_span,
                            location: expr.span,
                            source: s,
                        }),
                        None => {
                            Err(TyckError::MutatingImmutableValueOfUnknownCause { span: expr.span })
                        }
                    }
                }
            }
        }
    }

    fn mutability_from_ty(&self, ty: &Type<'alloc>) -> (bool, Option<Source>) {
        match ty.kind {
            TypeKind::Pointer(m, _) => (m, Some(Source::Expr(ty.data))),
            _ => (false, None),
        }
    }

    fn mutability(&self, expr: &Expr<'alloc>) -> (bool, Option<Source>) {
        match &expr.kind {
            ExprKind::Id(_, _, decl) => {
                let (inner_mut, source) = self.mutability_from_ty(&expr.ty.get().unwrap());
                let decl = decl.get().unwrap();
                if source.is_some() {
                    (inner_mut, source)
                } else {
                    (decl.mutable, Some(Source::Id(decl.span)))
                }
            }
            ExprKind::Index(lhs, _, _) => self.mutability(lhs),
            ExprKind::Field(lhs, _, _, _) => self.mutability(lhs),
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Amp), inner) => {
                (false, Some(Source::Ptr(inner.span)))
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::AmpMut), inner) => {
                (true, Some(Source::Ptr(inner.span)))
            }
            _ => self.mutability_from_ty(&expr.ty.get().unwrap()),
        }
    }

    /// Typechecks an expr, expecting it to be a "place" or "lval" expr - one that indicates a memory location. Returns a tuple of:
    /// - the mutability of the place (`bool`)
    /// - the mutability source, if any (`Option<Source>`)
    /// - the type of the place expr (`tyck::Type`)
    fn tyck_place_expr(
        &mut self,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<(bool, Option<Source>, Type<'alloc>)> {
        let (mutable, source, ty) = match &expr.kind {
            ExprKind::Id(name, id, is_func) => {
                let decl = self.tyck_var(name, id, is_func, expr.span)?;
                Ok((
                    decl.mutable,
                    Some(Source::Id(decl.name.span())),
                    decl.ty.get().unwrap(),
                ))
            }
            ExprKind::PrefixOp(Spanned(_, PrefixOpKind::Star), expr) => {
                let expr_ty = self.tyck_expr(expr)?;
                let (m, source) = self.mutability(expr);
                match expr_ty.kind {
                    TypeKind::Pointer(_, ty) => Ok((m, source, *ty)),
                    _ => return Err(TyckError::DereferencingNonPointer { span: expr.span }),
                }
            }
            ExprKind::Field(lhs, _, field, autoderef) => {
                let lhs_ty = self.tyck_expr(lhs)?;
                let (place_mut, source) = self.mutability(lhs);
                let (n, ptr_mut, ty) = autoderef_place_ptr(lhs_ty);
                autoderef.set(Some((n, ty)));
                let result_ty = self.field_access_ty(Spanned(lhs.span, ty), field)?;

                Ok((
                    if n > 0 { ptr_mut } else { place_mut },
                    if n > 0 {
                        Some(Source::Ptr(lhs.span))
                    } else {
                        source
                    },
                    result_ty,
                ))
            }
            ExprKind::Group(expr) => self.tyck_place_expr(expr),
            ExprKind::Index(lhs, index, autoderef) => {
                let lhs_ty = self.tyck_expr(lhs)?;
                let (place_mut, source) = self.mutability(lhs);
                let (n, ptr_mut, ty) = autoderef_place_ptr(lhs_ty);
                autoderef.set(Some((n, ty)));
                let index_ty = self.tyck_expr(index)?;
                let place_ty = self.tyck_index(lhs, ty, index, index_ty)?;
                Ok((
                    if n > 0 { ptr_mut } else { place_mut },
                    if n > 0 {
                        Some(Source::Ptr(expr.span))
                    } else {
                        source
                    },
                    place_ty,
                ))
            }
            _ => Err(TyckError::NotAPlaceExpr { span: expr.span }),
        }?;

        expr.ty.set(Some(ty));

        Ok((mutable, source, ty))
    }

    fn field_access_ty(
        &mut self,
        Spanned(expr_span, expr_ty): Spanned<Type<'alloc>>,
        Spanned(field_span, field): &Spanned<Field>,
    ) -> TyckResult<Type<'alloc>> {
        match expr_ty.kind {
            TypeKind::Struct(struct_name) => {
                let info = self
                    .struct_tys
                    .get(struct_name)
                    .expect("ICE: struct should be known")
                    .expect("ICE: struct should be defined");

                let field_name = match field {
                    Field::Name(x, i) => {
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
            TypeKind::Tuple(elems) => match field {
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
            Ok(Type::UNIT)
        };
        self.env.pop();
        result
    }

    fn tyck_stmt(&mut self, stmt: &'alloc Stmt<'alloc>) -> TyckResult<Type<'alloc>> {
        match &stmt.kind {
            StmtKind::Let(decl, expr) => {
                self.tyck_let(decl, expr)?;
                Ok(Type::UNIT)
            }
            StmtKind::Expr(expr) => self.tyck_expr(expr),
            StmtKind::Semi(expr) => {
                self.tyck_expr(expr)?;
                Ok(Type::UNIT)
            }
            StmtKind::While(cond, expr) => {
                self.coerce(cond, &Type::BOOL)?;
                self.coerce(expr, &Type::UNIT)?;
                Ok(Type::UNIT)
            }
            StmtKind::Return(Some(expr)) => self.tyck_expr(expr),
            StmtKind::Return(None) => Ok(Type::UNIT),
        }
    }

    fn tyck_let(
        &mut self,
        decl: &'alloc ast::DeclInfo<'alloc>,
        expr: &'alloc Expr<'alloc>,
    ) -> TyckResult<()> {
        let decl_ty = match &decl.ty_ast {
            Some(decl_ty_ast) => self.coerce(expr, &Type::from(self, decl_ty_ast)?)?,
            None => {
                self.tyck_expr(expr)?;
                let ty = self.resolve_ty_vars(expr)?;
                expr.ty.set(Some(ty));
                ty
            }
        };

        decl.set_ty(decl_ty)?;
        self.register_decl(decl);

        Ok(())
    }

    fn resolve_ty_vars(&self, expr: &'alloc Expr<'alloc>) -> TyckResult<Type<'alloc>> {
        match expr.kind {
            ExprKind::Tuple(exprs) => {
                let results: TyckResult<Vec<_>> = exprs
                    .into_iter()
                    .map(|expr| self.resolve_ty_vars(expr))
                    .collect();
                Ok(Type {
                    kind: TypeKind::Tuple(self.bump.alloc_slice_fill_iter(results?)),
                    data: expr.span,
                })
            }
            ExprKind::Array(exprs, _) => {
                let TypeKind::Array(ty, size) = expr.effective_ty().unwrap().kind else {
                    unreachable!()
                };

                let ty = match ty.kind {
                    TypeKind::Integral(_) => Type {
                        kind: TypeKind::Int,
                        data: ty.data,
                    },
                    _ => *ty,
                };

                for expr in exprs {
                    self.resolve_ty_vars(expr)?;
                }

                Ok(Type {
                    kind: TypeKind::Array(self.bump.alloc(ty), size),
                    data: expr.span,
                })
            }
            ExprKind::Group(e) => self.resolve_ty_vars(e),
            _ => {
                let ty = expr.effective_ty().unwrap();
                let ty = match ty.kind {
                    TypeKind::Integral(_) => {
                        let ty = Type {
                            kind: TypeKind::Int,
                            data: expr.span,
                        };
                        expr.coerced_ty.set(Some(ty));
                        ty
                    }
                    _ => ty,
                };
                Ok(ty)
            }
        }
    }
}

/// Resolve the given type (if it's an indefinite integer type) to `int`
fn resolve_integral<'b, 'alloc>(ty: &'b Type<'alloc>) -> &'b Type<'alloc> {
    match ty.kind {
        TypeKind::Integral(_) => &Type::INT,
        _ => ty,
    }
}

fn can_coerce(from_ty: &Type<'_>, to_ty: &Type<'_>) -> bool {
    use TypeKind::*;

    macro_rules! order {
        ($(
            $($from:pat)|+ => $($coerced:pat)|+ $(if $e:expr)?
        ),+ $(,)?) => {
            match from_ty.kind {
                $(
                    $($from)|+ => match to_ty.kind {
                        $(
                            $coerced
                        )|+ $(if $e)? => true,
                        _ => false,
                    }
                )+
                _ => false,
            }
        };
    }

    // Coercion is trivial if both types are the same
    if from_ty == to_ty {
        return true;
    }

    // Allow "truthy" values
    if let Bool = to_ty.kind {
        match from_ty.kind {
            Str | Struct(_) | Slice(_) | Array(_, _) | Tuple(_) => {}
            _ => return true,
        }
    }

    // Allow implicit conversion from/to *()
    if let Pointer(_, t) = to_ty.kind && t.is_unit() {
        return true;
    }
    if let Pointer(_, t) = from_ty.kind && t.is_unit() {
        return true;
    }

    // Variable size integer check
    if let Integral(_) = from_ty.kind {
        // FIXME: temporarily just allow coercion to any numerical types ig
        match to_ty.kind {
            Byte | Char | Short | UShort | Int | UInt | Long | ULong | ISize | USize | Float
            | Double => return true,
            _ => return false,
        }
    }

    // Handle types that may need to be deconstructed and coerced elementwise
    match (from_ty.kind, to_ty.kind) {
        (TypeKind::Pointer(m1, t1), TypeKind::Pointer(m2, t2)) => {
            // if assignee pointer is non-mut, then any mutability pointer can be assigned,
            // otherwise the mutability of the RHS determines assignability
            return (!m1 || m2) && can_coerce(t2, t1);
        }
        (TypeKind::Array(t1, n1), TypeKind::Array(t2, n2)) => {
            return n1 == n2 && can_coerce(t2, t1)
        }
        (TypeKind::Tuple(ts1), TypeKind::Tuple(ts2)) => {
            return ts1.iter().zip(ts2).all(|(t1, t2)| can_coerce(t2, t1))
        }
        (TypeKind::Fn(p1, r1), TypeKind::Fn(p2, r2)) => {
            return p1.iter().zip(p2).all(|(p1, p2)| can_coerce(p2, p1)) && can_coerce(r1, r2)
        }
        _ => {}
    }

    order! {
        Bool => Byte | Char | Short | UShort | Int | UInt | Long | ULong | ISize | USize | Float | Double,
        Byte => Short | Int | Long | ISize | Float | Double,
        Char => Short | Int | Long | ISize | UShort | UInt | ULong | USize | Float | Double,
        Short => Int | Long | ISize | Float | Double,
        UShort => Int | Long | ISize | UInt | ULong | USize | Float | Double,
        Int => Long | ISize | Float | Double,
        UInt => Long | ULong | USize | Float | Double,
        Long => Float | Double,
        ULong => Float | Double,
        Float => Double,
        Pointer(_, x) => Pointer(false, y) if x == y,
        Fn(..) => Pointer(false, to_ty @ Type { kind: Fn(..), .. }) if can_coerce(from_ty, to_ty),
    }
}

fn common_type<'b, 'alloc>(
    lhs_ty: &'b Type<'alloc>,
    rhs_ty: &'b Type<'alloc>,
) -> Option<&'b Type<'alloc>> {
    let result = if can_coerce(rhs_ty, lhs_ty) {
        Some(lhs_ty)
    } else if can_coerce(lhs_ty, rhs_ty) {
        Some(rhs_ty)
    } else {
        None
    };

    result.map(|t| resolve_integral(t))
}
