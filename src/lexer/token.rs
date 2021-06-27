use crate::util::Span;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub span: Span,
    pub ty: TokenType,
}

impl Token {
    pub fn new(span: Span, ty: TokenType) -> Token {
        Token { span, ty }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenType {
    Int(String),
    Float(String),
    Str(String),
    Char(char),

    Id(String),

    InterpolateBegin(String),
    InterpolateContinue(String),
    InterpolateEnd(String),

    Let,
    Fn,
    Struct,
    If,
    Else,
    For,
    While,
    Break,
    Continue,
    Return,

    LParen,
    RParen,
    LSquare,
    RSquare,
    LCurly,
    RCurly,

    Underscore,

    Bang,
    Tilde,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Caret,
    Eq,
    EqEq,
    Lt,
    Le,
    Gt,
    Ge,
    LtLt,
    GtGt,
    Dot,
    DotDot,
    Colon,
    ColonColon,
    Question,
    Backslash,
    Arrow,

    Comma,
    Semicolon,
    Eof,
}
