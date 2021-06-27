use crate::util::{FileId, Span};
use codespan_derive::IntoDiagnostic;
use lalrpop_util::lexer::Token;

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(IntoDiagnostic, Debug)]
#[file_id(FileId)]
pub enum ParseError {
    /* Lexing Issues */
    #[message = "Encountered unknown character"]
    UnknownChar {
        #[primary]
        span: Span,
    },

    #[message = "Encountered extra unexpected token"]
    ExtraToken {
        #[primary]
        span: Span,
    },

    #[message = "Character literal cannot be empty"]
    EmptyChar {
        #[primary]
        span: Span,
    },

    #[message = "Expected {expected} but reached end of file"]
    UnexpectedEof {
        #[primary]
        span: Span,
        expected: String,
    },

    #[message = "Expected {expected} but got \"{got}\""]
    Expected {
        #[primary]
        span: Span,
        expected: String,
        got: String,
    },

    #[message = "Array size must be a non-negative integer"]
    ArraySizeMustBeNonNegativeInteger {
        #[primary]
        span: Span,
    },
}

pub fn into_crust_error<'a>(
    error: lalrpop_util::ParseError<usize, Token<'a>, ParseError>,
    file_id: FileId,
) -> lalrpop_util::ParseError<usize, Token<'a>, ParseError> {
    match error {
        lalrpop_util::ParseError::InvalidToken { location } => lalrpop_util::ParseError::User {
            error: ParseError::UnknownChar {
                span: Span::new(file_id, location, location + 1),
            },
        },
        lalrpop_util::ParseError::UnrecognizedEOF { location, expected } => {
            lalrpop_util::ParseError::User {
                error: ParseError::UnexpectedEof {
                    span: Span::new(file_id, location, location + 1),
                    expected: if expected.is_empty() {
                        "more tokens".to_string()
                    } else {
                        expected.join(", ")
                    },
                },
            }
        }
        lalrpop_util::ParseError::UnrecognizedToken {
            token: (lo, token, hi),
            expected,
        } => lalrpop_util::ParseError::User {
            error: ParseError::Expected {
                span: Span::new(file_id, lo, hi),
                expected: if expected.is_empty() {
                    "another token".to_string()
                } else {
                    expected.join(", ")
                },
                got: format!("{}", token),
            },
        },
        lalrpop_util::ParseError::ExtraToken { token: (lo, _, hi) } => {
            lalrpop_util::ParseError::User {
                error: ParseError::ExtraToken {
                    span: Span::new(file_id, lo, hi),
                },
            }
        }
        _ => error,
    }
}
