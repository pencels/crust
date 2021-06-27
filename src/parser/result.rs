use crate::{
    lexer::token::Token,
    util::{FileId, Span},
};
use codespan_derive::IntoDiagnostic;

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(IntoDiagnostic, Debug)]
#[file_id(FileId)]
pub enum ParseError {
    /* Lexing Issues */
    #[message = "Encountered unknown character: {c:?}"]
    UnknownChar {
        c: char,
        #[primary]
        span: Span,
    },

    #[message = "Encountered unknown string escape character: '\\{c}'"]
    UnknownStringEscape {
        c: char,
        #[primary]
        span: Span,
    },

    #[message = "Empty atoms are not allowed"]
    EmptyAtom {
        #[primary]
        span: Span,
    },

    #[message = "Reached end of line or file in delimited atom"]
    EofInDelimitedAtom {
        #[primary]
        span: Span,
    },

    #[message = "Character literal cannot be empty"]
    EmptyChar {
        #[primary]
        span: Span,
    },

    #[message = "Reached end of line or file in string"]
    EofInString {
        #[primary]
        span: Span,
    },

    #[message = "Reached end of line or file in delimited operator"]
    EofInDelimitedOperator {
        #[primary]
        span: Span,
    },

    #[message = "Expected a numerical value to follow the exponential"]
    NoNumeralAfterExponential {
        #[primary]
        span: Span,
    },

    #[message = "Undeclared operator: {op:?}"]
    UnknownOp {
        op: String,
        #[primary]
        span: Span,
    },

    #[message = "Expected more tokens but reached end of file"]
    UnexpectedEof {
        #[primary]
        span: Span,
    },

    #[message = "Expected {expected} but got {got:?}"]
    Expected {
        #[primary]
        span: Span,
        expected: String,
        got: Token,
    },

    #[message = "The given constraint would induce a cycle"]
    PrecConstraintAddsCycle {
        #[primary]
        span: Span,
    },

    #[message = "Operator was declared non-associative, add parentheses to clear ambiguity"]
    NonAssocOperator {
        #[primary]
        span: Span,
    },

    #[message = "Operator was already declared as {was}, cannot add declaration for {now}"]
    RedeclaringOp {
        #[primary = "Trying to redeclare the operator here"]
        span: Span,
        #[secondary = "Operator already declared here"]
        original_span: Span,
        was: &'static str,
        now: &'static str,
    },

    #[message = "Function definition expected at least {min} params or at most {max:?} params"]
    InvalidNumberOfFnParams {
        #[primary]
        name_span: Span,
        #[secondary = "{given} param(s) given"]
        last_param_span: Span,
        min: usize,
        max: Option<usize>,
        given: usize,
    },

    #[message = "Token could lead to ambiguity in unenclosed pattern, please add parentheses"]
    AmbiguousUnenclosedPattern {
        #[primary]
        span: Span,
    },

    #[message = "Spread patterns are only allowed at the end of list or argument patterns"]
    SpreadNotAtEnd {
        #[primary]
        span: Span,
    },

    #[message = "Only an identifier or ignore pattern is allowed in this position"]
    OnlyIdOrIgnore {
        #[primary]
        span: Span,
    },

    #[message = "Non-type used as head of destructuring pattern"]
    NonTypeIdInDestructuringPattern {
        #[primary]
        span: Span,
    },
}
