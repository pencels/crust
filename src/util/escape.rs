#[derive(Debug)]
pub struct UnescapeError {
    idx: usize,
    reason: UnescapeErrorReason,
}

#[derive(Debug)]
pub enum UnescapeErrorReason {
    NonAsciiValue,
    UnsupportedEscapeChar(char),
    UnexpectedEof,
}

pub fn unescape(s: &str) -> Result<String, UnescapeError> {
    let mut unescaped = String::new();
    let mut chars = s.char_indices();

    while let Some((i, c)) = chars.next() {
        let c = match c {
            '\\' => match chars.next() {
                Some((_, c)) => match c {
                    'a' => '\u{07}',
                    'b' => '\u{08}',
                    'v' => '\u{0b}',
                    'f' => '\u{0c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '0' => '\0',
                    '\\' => '\\',
                    '\'' => '\'',
                    '\"' => '\"',
                    ' ' => ' ',
                    _ => {
                        return Err(UnescapeError {
                            idx: i,
                            reason: UnescapeErrorReason::UnsupportedEscapeChar(c),
                        })
                    }
                },
                _ => {
                    return Err(UnescapeError {
                        idx: i,
                        reason: UnescapeErrorReason::UnexpectedEof,
                    })
                }
            },
            _ => {
                if c.is_ascii() {
                    c
                } else {
                    return Err(UnescapeError {
                        idx: i,
                        reason: UnescapeErrorReason::NonAsciiValue,
                    });
                }
            }
        };
        unescaped.push(c);
    }

    Ok(unescaped)
}
