use crate::error::{CurrentPosition, LineCounter, UpdateLineCounter};
use crate::jcampdx::decoding::error::Error;
use logos::{Lexer, Logos};

/// JCAMP-DX encoded block lexer.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Logos)]
#[logos(extras = LineCounter)]
#[logos(error(Error, invalid_literal))]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+")]
#[logos(skip r"(?&comment)")]
pub(crate) enum EncodedToken {
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)", UpdateLineCounter::newline)]
    CheckPoint,
    /// Numeric values, `AFFN` in the JCAMP-DX standard.
    #[regex(r"[+-]?(0|[1-9]\d*)(\.\d+)?([eE][+-]\d+)?")]
    Numeric,
    /// Compressed value, `SQZ` in the JCAMP-DX standard.
    #[regex(r"[@A-Ia-i]\d*")]
    Compressed,
    /// Difference value, `DIF` in the JCAMP-DX standard.
    #[regex(r"[%J-Rj-r]\d*")]
    Difference,
    /// Duplicate previous value, `DUP` in the JCAMP-DX standard.
    #[regex(r"[S-Zs]\d*")]
    Duplicate,
    /// Invalid data point, marked as `?` in the JCAMP-DX standard.
    #[token("?")]
    Invalid,
    /// An encoded block optionally ends with the next JCAMP-DX header key.
    #[token("##")]
    #[token("##.")]
    #[token("##$")]
    End,
}

/// Literals that could not be matched to any token.
fn invalid_literal(lexer: &Lexer<EncodedToken>) -> Error {
    Error::invalid_literal(lexer.position())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jcampdx::decoding::error::Result;

    macro_rules! lexer_test {
        ($name:ident, $data:expr, $tokens:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $tokens;
                let tokens = EncodedToken::lexer(data).collect::<Vec<Result<EncodedToken>>>();
                assert_eq!(tokens, expected);
            }
        };
    }

    lexer_test!(
        affn,
        "9 1 2 3 4 5\n\
         5 6 7 8 9 10",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::CheckPoint),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
        ]
    );
    lexer_test!(
        pac,
        "9 +1+2+3+4+5\n\
         5 +6+7+8+9+10",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::CheckPoint),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
        ]
    );
    lexer_test!(
        sqz,
        "9 ABCDEFGHIA0",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Compressed),
        ]
    );
    lexer_test!(
        dif,
        "9 AJJJJJJJJJ\n\
         0 A0",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::CheckPoint),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Compressed),
        ]
    );
    lexer_test!(
        difdup,
        "9 AJs\n\
         0 A0",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Compressed),
            Ok(EncodedToken::Difference),
            Ok(EncodedToken::Duplicate),
            Ok(EncodedToken::CheckPoint),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Compressed),
        ]
    );
    lexer_test!(
        invalid,
        "9 1 2 3 ? 5\n\
         5 6 7 8 9 10",
        [
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Invalid),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::CheckPoint),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
            Ok(EncodedToken::Numeric),
        ]
    );
}
