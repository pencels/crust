use std::collections::HashMap;

use crate::parser::ast::{Expr, Type};

pub type VarId = usize;

struct TyEnv<'b> {
    pub tys: HashMap<VarId, Type<'b>>,
}

pub fn lower_program<'b>(defns: &'b Expr<'b>) {}
