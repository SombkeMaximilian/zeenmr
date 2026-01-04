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
    use super::*;
    use crate::Position;

    macro_rules! lexer_test {
        ($name:ident, $data:expr, $tokens:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $tokens;
                let tokens = FormatToken::lexer(data)
                    .collect::<Vec<crate::jcampdx::block_format::error::Result<FormatToken>>>();
                assert_eq!(tokens, expected);
            }
        };
    }

    lexer_test!(
        repeating,
        "(X++(Y..Y))\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Increment),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Repeat),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        repeating_block_kind,
        "(X++(R..R)), XYDATA\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Increment),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Repeat),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::DataBlockKind),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        multi_group,
        "(XY..XY)\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Repeat),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        multi_group_block_kind,
        "(XY..XY), PEAKS\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Repeat),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::DataBlockKind),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        single_group,
        "(XYWA)\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        single_group_block_kind,
        "(XYWA), PEAK ASSIGNMENTS\n",
        [
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::Identifier),
            Ok(FormatToken::DataBlockKind),
            Ok(FormatToken::End),
        ]
    );
    lexer_test!(
        invalid_literal_repeat,
        "(X.Y)",
        [
            Ok(FormatToken::Identifier),
            Err(Error::invalid_literal(Position { line: 0, column: 2 })),
            Ok(FormatToken::Identifier),
        ]
    );
    lexer_test!(
        invalid_literal_numeric,
        "(1..3)",
        [
            Err(Error::invalid_literal(Position { line: 0, column: 1 })),
            Ok(FormatToken::Repeat),
            Err(Error::invalid_literal(Position { line: 0, column: 4 })),
        ]
    );
}
