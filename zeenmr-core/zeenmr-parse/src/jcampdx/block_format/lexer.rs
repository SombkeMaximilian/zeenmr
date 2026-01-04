use crate::jcampdx::block_format::error::Error;
use crate::{Cursor, Location, UpdateCursor};
use logos::{Lexer, Logos};

#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(extras = Cursor)]
#[logos(error(Error, invalid_literal))]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern parentheses = r"[()]")]
#[logos(skip r"(?&parentheses)")]
pub(crate) enum FormatToken {
    /// Identifier for a quantity.
    ///
    /// Commonly used in the JCAMP-DX standard:
    /// - `X` for abscissa
    /// - `Y`, `R`, `I` for ordinates (any, real, imaginary)
    /// - `Z` for a third dimension
    /// - `W` for peak width
    /// - `M` for multiplicity
    /// - `A` for assignment
    #[regex(r"[A-Za-z][0-9]*")]
    Identifier,
    /// Abscissa is incremented by the number of values in a line.
    #[token("++")]
    Increment,
    /// Repeat current identifier until line ends.
    #[token("..")]
    Repeat,
    /// A `DATA TABLE` data block contains the exact type of data after a comma.
    #[regex(r",[^\n\r]+")]
    DataBlockKind,
    /// A new line ends a data block format specifier.
    #[regex(r"(?&newline)", UpdateCursor::newline)]
    End,
}

fn invalid_literal(lexer: &Lexer<FormatToken>) -> Error {
    Error::invalid_literal(lexer.location())
}

#[cfg(test)]
mod tests {
    use crate::location::Position;
    use super::*;

    macro_rules! lexer_test {
        ($name:ident, $data:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let mut lexer = FormatToken::lexer(data);
                if let Some(e) = lexer.find_map(Result::err) {
                    panic!("lexer error: {e}");
                }
            }
        };
    }

    lexer_test!(repeating, "(X++(Y..Y))\n");
    lexer_test!(repeating_block_kind, "(X++(R..R)), XYDATA\n");
    lexer_test!(multi_group, "(XY..XY)\n");
    lexer_test!(multi_group_block_kind, "(XY..XY), PEAKS\n");
    lexer_test!(single_group, "(XYWA)\n");
    lexer_test!(single_group_block_kind, "(XYWA), PEAK ASSIGNMENTS\n");

    macro_rules! error_test {
        ($name:ident, $data:expr, $error:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let error = $error;
                let mut lexer = FormatToken::lexer(data);
                if let Some(e) = lexer.find_map(Result::err) {
                    assert_eq!(e, error);
                } else {
                    panic!("no lexer error");
                }
            }
        };
    }

    error_test!(
        invalid_literal_repeat,
        "(X.Y)",
        Error::invalid_literal(Position { line: 0, column: 2 })
    );
    error_test!(
        invalid_literal_numeric,
        "(1..3)",
        Error::invalid_literal(Position { line: 0, column: 1 })
    );
}
