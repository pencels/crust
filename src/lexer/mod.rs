pub mod token;

/*
use std::str::Chars;

use lookahead::{lookahead, Lookahead};

use self::token::{Token, TokenType};

use crate::{
    parser::result::{ParseError, ParseResult},
    util::{FileId, Span},
};

/// Mnemonic for the EOF end-of-file character.
const EOF: char = '\x00';

pub struct Lexer<'file> {
    file_id: FileId,
    stream: Lookahead<Chars<'file>>,

    /// The 0-indexed char position of the first char considered for the current token being lexed.
    start_pos: usize,
    /// The 0-indexed char position of `current_char`.
    current_pos: usize,
    /// The current character in the stream.
    current_char: char,
    /// The next character in the stream.
    next_char: char,

    /// Stack of string interpolation parentheses counts. Each stack position represents a string
    /// interpolation context. The value at each position represents how many open parens
    /// were encountered in the context.
    interp_parenthetical: Vec<usize>,
}

pub enum LexStringChar {
    Char(char),
    QuoteEnd,
    InterpolateBegin,
}

impl<'file> Lexer<'file> {
    pub fn new(file_id: FileId, source: &'file str) -> Lexer<'file> {
        let mut lex = Lexer {
            file_id,
            stream: lookahead(source.chars()),
            start_pos: 0,
            current_pos: 0,
            current_char: EOF,
            next_char: EOF,
            interp_parenthetical: vec![],
        };

        lex.bump(1);
        lex.current_pos = 0;
        lex
    }

    pub fn current_span(&self) -> Span {
        Span::new(self.file_id, self.current_pos, self.current_pos + 1)
    }

    fn spanned_lex(&mut self) -> ParseResult<Token> {
        self.discard_whitespace_or_comments()?;

        self.start_pos = self.current_pos;
        let ty = self.lex()?;

        Ok(Token::new(
            Span::new(self.file_id, self.start_pos, self.current_pos),
            ty,
        ))
    }

    /// Cautiously bumps the input stream, avoiding reading past a '\n' character.
    fn bump(&mut self, n: usize) {
        for _ in 0..n {
            self.current_char = self.stream.next().unwrap_or(EOF);
            self.next_char = self.stream.lookahead(0).map_or(EOF, |&c| c);
            self.current_pos += 1;
        }
    }

    fn discard_whitespace_or_comments(&mut self) -> ParseResult<()> {
        loop {
            // Consume any whitespace or comments before a real token
            match self.current_char {
                '-' => {
                    // May be a comment or an operator/numeric
                    if self.next_char == '-' {
                        self.scan_comment()?;
                    } else {
                        break; // Operator/numeric
                    }
                }
                ' ' | '\t' | '\r' | '\n' => self.bump(1),
                '\\' => {
                    if self.next_char == '\n' {
                        self.bump(2); // Backslashes escape newlines
                    } else {
                        break;
                    }
                }
                _ => {
                    break;
                }
            }
        }

        Ok(())
    }

    fn scan_comment(&mut self) -> ParseResult<()> {
        match self.next_char {
            '-' => {
                // Read until end of line.
                self.bump(2); // Read --.
                while self.current_char != '\r'
                    && self.current_char != '\n'
                    && self.current_char != EOF
                {
                    self.bump(1);
                }
            }
            _ => unreachable!("ICE: Expected -- to begin scanned comment"),
        }

        Ok(())
    }

    /// Returns the next lookahead token.
    fn lex(&mut self) -> ParseResult<TokenType> {
        let c = self.current_char;

        if c == EOF {
            Ok(TokenType::Eof)
        } else if c == '"' {
            self.scan_string()
        } else if c == '\'' {
            self.scan_char()
        } else if is_numeric(c) {
            self.scan_numeric_literal()
        } else if is_identifier_start(c) {
            self.scan_identifier_or_keyword()
        } else {
            match c {
                '.' => {
                    if self.next_char == '.' {
                        self.bump(2);
                        Ok(TokenType::DotDot)
                    } else {
                        self.bump(1);
                        Ok(TokenType::Dot)
                    }
                }
                ',' => {
                    self.bump(1);
                    Ok(TokenType::Comma)
                }
                ':' => {
                    if self.next_char == ':' {
                        self.bump(2);
                        Ok(TokenType::ColonColon)
                    } else {
                        self.bump(1);
                        Ok(TokenType::Colon)
                    }
                }
                ';' => {
                    self.bump(1);
                    Ok(TokenType::Semicolon)
                }
                '{' => {
                    self.bump(1);
                    Ok(TokenType::LCurly)
                }
                '}' => {
                    self.bump(1);
                    Ok(TokenType::RCurly)
                }
                '[' => {
                    self.bump(1);
                    Ok(TokenType::LSquare)
                }
                ']' => {
                    self.bump(1);
                    Ok(TokenType::RSquare)
                }
                '(' => {
                    self.bump(1);

                    if let Some(nest) = self.interp_parenthetical.last_mut() {
                        *nest += 1;
                    }

                    Ok(TokenType::LParen)
                }
                ')' => {
                    if let Some(0) = self.interp_parenthetical.last() {
                        self.scan_interp_continue()
                    } else {
                        if let Some(n) = self.interp_parenthetical.last_mut() {
                            *n -= 1;
                        }

                        self.bump(1);
                        Ok(TokenType::RParen)
                    }
                }
                '=' => {
                    if self.next_char == '=' {
                        self.bump(2);
                        Ok(TokenType::EqEq)
                    } else {
                        self.bump(1);
                        Ok(TokenType::Eq)
                    }
                }
                '-' => {
                    if is_numeric(self.next_char) {
                        self.scan_numeric_literal()
                    } else if self.next_char == '>' {
                        self.bump(2);
                        Ok(TokenType::Arrow)
                    } else {
                        self.bump(1);
                        Ok(TokenType::Minus)
                    }
                }
                '+' => {
                    if is_numeric(self.next_char) {
                        self.scan_numeric_literal()
                    } else {
                        self.bump(1);
                        Ok(TokenType::Plus)
                    }
                }
                '\\' => {
                    self.bump(1);
                    Ok(TokenType::Backslash)
                }
                c => {
                    self.bump(1);
                    Err(ParseError::UnknownChar {
                        c,
                        span: self.current_span(),
                    })
                }
            }
        }
    }

    /// Scans a new parsed string token.
    fn scan_string(&mut self) -> ParseResult<TokenType> {
        self.bump(1); // Blindly consume the quote character
        let mut string = String::new();

        loop {
            match self.scan_string_char()? {
                LexStringChar::Char(c) => {
                    string.push(c);
                }
                LexStringChar::QuoteEnd => {
                    return Ok(TokenType::Str(string));
                }
                LexStringChar::InterpolateBegin => {
                    self.interp_parenthetical.push(0);
                    return Ok(TokenType::InterpolateBegin(string));
                }
            }
        }
    }

    fn scan_interp_continue(&mut self) -> ParseResult<TokenType> {
        self.bump(1); // Blindly consume the rparen character
        let mut string = String::new();

        loop {
            match self.scan_string_char()? {
                LexStringChar::Char(c) => {
                    string.push(c);
                }
                LexStringChar::QuoteEnd => {
                    self.interp_parenthetical.pop();
                    return Ok(TokenType::InterpolateEnd(string));
                }
                LexStringChar::InterpolateBegin => {
                    return Ok(TokenType::InterpolateContinue(string));
                }
            }
        }
    }

    fn scan_string_char(&mut self) -> ParseResult<LexStringChar> {
        let ret;

        match self.current_char {
            '\\' => {
                match self.next_char {
                    'r' => {
                        ret = LexStringChar::Char('\r');
                    }
                    'n' => {
                        ret = LexStringChar::Char('\n');
                    }
                    't' => {
                        ret = LexStringChar::Char('\t');
                    }
                    '"' => {
                        ret = LexStringChar::Char('\"');
                    }
                    '\'' => {
                        ret = LexStringChar::Char('\'');
                    }
                    '\\' => {
                        ret = LexStringChar::Char('\\');
                    }
                    '(' => {
                        ret = LexStringChar::InterpolateBegin;
                    }
                    c => {
                        return Err(ParseError::UnknownStringEscape {
                            c,
                            span: self.current_span(),
                        });
                    }
                }
                self.bump(2);
            }
            '\"' => {
                ret = LexStringChar::QuoteEnd;
                self.bump(1);
            }
            '\r' | '\n' | EOF => {
                return Err(ParseError::EofInString {
                    span: self.current_span(),
                });
            }
            c => {
                ret = LexStringChar::Char(c);
                self.bump(1);
            }
        }

        Ok(ret)
    }

    /// Scans a character literal.
    fn scan_char(&mut self) -> ParseResult<TokenType> {
        let c = match self.current_char {
            '\\' => {
                let c = match self.next_char {
                    '0' => '\0',
                    'r' => '\r',
                    'n' => '\n',
                    't' => '\t',
                    '\\' => '\\',
                    c => c,
                };
                self.bump(2);
                c
            }
            '\'' => {
                self.bump(1);
                return Err(ParseError::EmptyChar {
                    span: self.current_span(),
                });
            }
            '\r' | '\n' | EOF => {
                return Err(ParseError::EofInString {
                    span: self.current_span(),
                });
            }
            c => {
                self.bump(1);
                c
            }
        };

        Ok(TokenType::Char(c))
    }

    /// Scans a numeric literal, consuming it and converting it to a token in
    /// the process.
    fn scan_numeric_literal(&mut self) -> ParseResult<TokenType> {
        let mut string = String::new();

        if self.current_char == '-' || self.current_char == '+' {
            if self.current_char == '-' {
                string.push(self.current_char);
            }

            self.bump(1);
        }

        while is_numeric(self.current_char) {
            string.push(self.current_char);
            self.bump(1);
        }

        // Check for fractional part and switch on the emitted token type.
        if self.current_char == '.' {
            string += ".";
            self.bump(1);

            while is_numeric(self.current_char) {
                string.push(self.current_char);
                self.bump(1);
            }

            if self.current_char == 'e' || self.current_char == 'E' {
                self.bump(1);
                string.push('e');

                if self.current_char == '+' || self.current_char == '-' {
                    string.push(self.current_char);
                    self.bump(1);
                }

                let mut expect_number = false;

                while is_numeric(self.current_char) {
                    expect_number = true;
                    string.push(self.current_char);
                    self.bump(1);
                }

                if !expect_number {
                    return Err(ParseError::NoNumeralAfterExponential {
                        span: self.current_span(),
                    });
                }
            }
            Ok(TokenType::Float(string))
        } else {
            Ok(TokenType::Int(string))
        }
    }

    // Scans an identifier, unless it matches a keyword.
    fn scan_identifier_or_keyword(&mut self) -> ParseResult<TokenType> {
        let mut string = String::new();

        string.push(self.current_char);
        self.bump(1);

        while is_identifier_continuer(self.current_char) {
            string.push(self.current_char);
            self.bump(1);
        }

        let token = match string.as_str() {
            "_" => TokenType::Underscore,

            "let" => TokenType::Let,
            "fn" => TokenType::Fn,
            "struct" => TokenType::Struct,

            _ => match string.chars().nth(0).unwrap() {
                'A'..='Z' | 'a'..='z' | '_' => TokenType::Id(string),
                _ => unreachable!("ICE: non-identifier character found in identifier"),
            },
        };

        Ok(token)
    }
}

impl<'file> Iterator for Lexer<'file> {
    type Item = ParseResult<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.spanned_lex() {
            Ok(Token {
                ty: TokenType::Eof, ..
            }) => None,
            t => Some(t),
        }
    }
}

/// Returns whether the character is a valid part of a number.
fn is_numeric(c: char) -> bool {
    match c {
        '0'..='9' => true,
        _ => false,
    }
}

/// Returns whether the character is a valid beginning of an identifier.
fn is_identifier_start(c: char) -> bool {
    match c {
        'a'..='z' => true,
        'A'..='Z' => true,
        '_' => true,
        _ => false,
    }
}

/// Returns whether the character is a valid non-initial part of an identifier.
fn is_identifier_continuer(c: char) -> bool {
    match c {
        'a'..='z' => true,
        'A'..='Z' => true,
        '0'..='9' => true,
        '_' => true,
        _ => false,
    }
}

 */
