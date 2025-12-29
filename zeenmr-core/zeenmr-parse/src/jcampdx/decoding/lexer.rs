use crate::jcampdx::decoding::error::{Error, Result};
use crate::{Cursor, Location, Position};
use logos::{Lexer, Logos};
use std::num::IntErrorKind;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Logos)]
#[logos(extras = Cursor)]
#[logos(error(Error, invalid_literal))]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)")]
#[logos(skip r"(?&comment)")]
pub(crate) enum EncodedToken {
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)", check_point)]
    CheckPoint,
    /// Numeric values, `AFFN` in the JCAMP-DX standard.
    #[regex(r"[+-]?(0|[1-9]\d*)(\.\d+)?([eE][+-]\d+)?", affn)]
    Numeric(i64),
    /// Compressed value, `SQZ` in the JCAMP-DX standard.
    #[regex(r"[@A-Ia-i]\d*", asdf)]
    Compressed(i64),
    /// Difference value, `DIF` in the JCAMP-DX standard.
    #[regex(r"[%J-Rj-r]\d*", asdf)]
    Difference(i64),
    /// Duplicate previous value, `DUP` in the JCAMP-DX standard.
    #[regex(r"[S-Zs]\d*", asdf)]
    Duplicate(i64),
    /// Invalid data point, marked as `?` in the JCAMP-DX standard.
    #[token("?", Location::location)]
    Invalid(Position),
    /// An encoded block optionally ends with the next JCAMP-DX header key.
    #[token("##")]
    #[token("##.")]
    #[token("##$")]
    End,
}

/// Literals that could not be matched to any token.
fn invalid_literal(lexer: &Lexer<EncodedToken>) -> Error {
    Error::invalid_literal(lexer.location())
}

/// Updates the cursor at the start of every line.
fn check_point(lexer: &mut Lexer<EncodedToken>) {
    lexer.extras.line += 1;
    lexer.extras.index = lexer.span().end;
}

/// Parse an `AFFN` numeric value.
fn affn(lexer: &Lexer<EncodedToken>) -> Result<i64> {
    match lexer.slice().parse::<i64>() {
        Ok(int) => Ok(int),
        Err(e) => match e.kind() {
            IntErrorKind::InvalidDigit => {
                let float = lexer
                    .slice()
                    .parse::<f64>()
                    .map_err(|_| Error::unsupported_format(lexer.location()))?;

                if float.fract() == 0.0 {
                    Ok(float as i64)
                } else {
                    Err(Error::unsupported_format(lexer.location()))
                }
            }
            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                Err(Error::overflow(lexer.location()))
            }
            _ => unreachable!(),
        },
    }
}

/// Parse an `ASDF` compressed, difference, or duplicate value.
///
/// # Encoding
///
/// An `ASDF` encoded value is composed of a leading character, which determines
/// the type of encoding, and a trailing numeric sequence.
fn asdf(lexer: &Lexer<EncodedToken>) -> Result<i64> {
    let encoded = &lexer.slice()[..1];
    let decoded: i64 = match encoded {
        "@" | "%" => 0,
        "A" | "a" | "J" | "j" | "S" => 1,
        "B" | "b" | "K" | "k" | "T" => 2,
        "C" | "c" | "L" | "l" | "U" => 3,
        "D" | "d" | "M" | "m" | "V" => 4,
        "E" | "e" | "N" | "n" | "W" => 5,
        "F" | "f" | "O" | "o" | "X" => 6,
        "G" | "g" | "P" | "p" | "Y" => 7,
        "H" | "h" | "Q" | "q" | "Z" => 8,
        "I" | "i" | "R" | "r" | "s" => 9,
        _ => unreachable!("invalid ASDF character: {}", encoded),
    };
    let sign = match encoded.chars().next().unwrap() {
        '@' | '%' | 'A'..='Z' | 's' => 1,
        'a'..='r' => -1,
        _ => unreachable!("invalid ASDF character: {}", encoded),
    };
    let numeric = &lexer.slice()[1..];
    let order = numeric.len() as u32;

    match numeric.parse::<i64>() {
        Ok(numeric) => Ok(sign * (decoded * 10_i64.pow(order) + numeric)),
        Err(e) => match e.kind() {
            IntErrorKind::Empty => Ok(sign * (decoded * 10_i64.pow(order))),
            IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                Err(Error::overflow(lexer.location()))
            }
            _ => unreachable!(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! lexer_test {
        ($name:ident, $data:tt) => {
            #[test]
            fn $name() {
                let data = $data;
                let mut lexer = EncodedToken::lexer(data);
                if let Some(e) = lexer.find_map(Result::err) {
                    panic!("lexer error: {e}");
                }
            }
        };
    }

    lexer_test!(affn, "9 1 2 3 4 5 6 7 8 9 10");
    lexer_test!(pac, "9 +1+2+3+4+5+6+7+8+9+10");
    lexer_test!(sqz, "9 ABCDEFGHIA0");
    lexer_test!(dif, "9 AJJJJJJJJJ");
    lexer_test!(difdup, "9 AJs");
    lexer_test!(invalid, "9 1 2 3 ? 5 6 7 8 9 10");

    macro_rules! error_test {
        ($name:ident, $data:tt, $error:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let error = $error;
                let mut lexer = EncodedToken::lexer(data);
                if let Some(e) = lexer.find_map(Result::err) {
                    assert_eq!(e, error);
                } else {
                    panic!("no lexer error");
                }
            }
        };
    }

    error_test!(
        affn_positive_overflow,
        "0 +900000000000000000000",
        Error::overflow(Position { line: 0, column: 2 })
    );
    error_test!(
        affn_negative_overflow,
        "0 -900000000000000000000",
        Error::overflow(Position { line: 0, column: 2 })
    );
    error_test!(
        sqz_positive_overflow,
        "0 I900000000000000000000",
        Error::overflow(Position { line: 0, column: 2 })
    );
    error_test!(
        sqz_negative_overflow,
        "0 i900000000000000000000",
        Error::overflow(Position { line: 0, column: 2 })
    );
}
