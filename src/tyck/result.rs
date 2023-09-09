use crate::{
    parser::ast::BinOpKind,
    util::{FileId, Span},
};
use codespan_derive::IntoDiagnostic;

pub type TyckResult<T> = Result<T, TyckError>;

#[derive(Debug, IntoDiagnostic)]
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

    #[message = "Cannot assign to expression"]
    CannotAssignToExpr {
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

    #[message = "Cannot make a mut pointer to an immutable location"]
    #[note = "add a cast or modify the source type declaration"]
    MutPointerToNonMutExpr {
        #[primary]
        addressing_op: Span,
        #[secondary = "this place expression is immutable"]
        location: Span,
        #[secondary = "immutable type declared here"]
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
        #[primary]
        span: Span,
        #[secondary = "consider making this a mutable pointer"]
        ptr: Span,
    },

    #[message = "Cannot mutate immutable value"]
    MutatingImmutableValueOfUnknownCause {
        #[primary]
        span: Span,
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

    #[message = "Struct has duplicate member name"]
    StructHasDuplicateMember {
        #[primary = "fix it binch"]
        duplicate_span: Span,
        #[secondary = "first declared here"]
        original_span: Span,
    },

    #[message = "Cannot cast type"]
    CannotCastType {
        #[primary]
        expr_span: Span,
        #[secondary]
        ty_span: Span,
        from_ty_name: String,
        to_ty_name: String,
    },

    #[message = "Cannot coerce {from_ty_name} to {to_ty_name}"]
    #[note = "consider adding an explicit cast"]
    CannotCoerceType {
        #[primary = "expr being coerced"]
        expr_span: Span,
        #[secondary = "coercion implied by type"]
        ty_span: Span,
        from_ty_name: String,
        to_ty_name: String,
    },

    #[message = "Binary operator {op} is undefined for {lhs} and {rhs}"]
    IncorrectTypesForBinaryOperator {
        op: BinOpKind,
        #[primary]
        span: Span,
        lhs: String,
        rhs: String,
    },

    #[message = "No common type between {lhs} and {rhs}"]
    #[note = "consider adding an explicit cast"]
    NoCommonType {
        lhs: String,
        #[primary]
        lhs_span: Span,
        rhs: String,
        #[primary]
        rhs_span: Span,
    },

    #[message = "Not all code paths return a value"]
    NotAllPathsReturnAValue {
        #[secondary = "non-unit return type"]
        ret_ty_span: Span,
        #[primary = "control reaches end of function without returning a value"]
        last_stmt_span: Span,
    },

    #[message = "Statement cannot be used outside of a loop"]
    StatementCannotBeUsedOutsideOfLoop {
        #[primary]
        span: Span,
    },
}
