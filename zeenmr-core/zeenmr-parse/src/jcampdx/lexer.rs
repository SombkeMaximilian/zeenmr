use crate::jcampdx::error::Error;
use crate::{Cursor, Location, UpdateCursor};
use logos::{Lexer, Logos};

/// JCAMP-DX lexer.
#[derive(Clone, PartialEq, Debug, Logos)]
#[logos(extras = Cursor)]
#[logos(error(Error, invalid_literal))]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(skip(r"(?&newline)", UpdateCursor::newline))]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+(?&newline)?")]
#[logos(skip r"(?&comment)")]
pub(crate) enum Token {
    /// JCAMP-DX header keys start with `##` or `##.`. Bruker-specific keys
    /// start with `##$` but follow the same rules otherwise.
    #[token("##")]
    #[token("##.")]
    #[token("##$")]
    Key,
    /// The separator between key and value is always an equals sign.
    #[token("=")]
    Equals,
    /// Commas separating multiple values.
    #[token(",")]
    Comma,
    /// Opening parenthesis to start compound values or ranges.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end compound values or ranges.
    #[token(")")]
    CloseParenthesis,
    /// Starting angle brackets for strings with whitespace.
    #[token("<")]
    OpenAngle,
    /// Ending angle brackets for strings with whitespace.
    #[token(">")]
    CloseAngle,
    /// Integer or floating point values in standard or scientific notation.
    #[regex(r"-?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?")]
    Numeric,
    /// Anything not numeric is a string.
    #[regex(r"[^ \t\r\n=#,<>\(\)]*")]
    String,
    /// A dataset always begins with a `TITLE` key.
    #[token("TITLE")]
    Title,
    /// An `NTUPLES` key indicates that multiple data blocks belong to the same
    /// dataset.
    #[token("NTUPLES")]
    Tuples,
    /// Dataset identifier within an `NTUPLES` dataset.
    #[token("PAGE")]
    Page,
    /// The following data block potentially uses the `ASDF` encoding.
    #[token("XYDATA")]
    EncodedBlock,
    /// The following data block only contains UTF-8 and numeric values, and
    /// each entry contains grouped data (e.g., `XYWM` for coordinates, width
    /// and multiplicity).
    #[token("XYPOINTS")]
    #[regex(r"PEAK[\s_-]TABLE")]
    #[regex(r"PEAK[\s_-]ASSIGNMENTS")]
    GroupedBlock,
    /// A data table may be either an encoded or a grouped block.
    #[regex(r"DATA[\s_-]TABLE")]
    AmbiguousBlock,
    /// End of the current dataset.
    #[token("END")]
    #[token("END NTUPLES")]
    End,
}

/// Literals that could not be matched to any token.
fn invalid_literal(lexer: &Lexer<Token>) -> Error {
    Error::invalid_literal(lexer.location())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jcampdx::error::Result;

    macro_rules! lexer_test {
        ($name:ident, $data:expr, $tokens:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $tokens;
                let tokens = Token::lexer(data).collect::<Vec<Result<Token>>>();
                assert_eq!(tokens, expected);
            }
        };
    }

    lexer_test!(
        title,
        "##TITLE= Example Title",
        [
            Ok(Token::Key),
            Ok(Token::Title),
            Ok(Token::Equals),
            Ok(Token::String),
            Ok(Token::String),
        ]
    );
    lexer_test!(
        key_value,
        "##EXAMPLE KEY= (VALUE, <VALUE 123>)",
        [
            Ok(Token::Key),
            Ok(Token::String),
            Ok(Token::String),
            Ok(Token::Equals),
            Ok(Token::OpenParenthesis),
            Ok(Token::String),
            Ok(Token::Comma),
            Ok(Token::OpenAngle),
            Ok(Token::String),
            Ok(Token::Numeric),
            Ok(Token::CloseAngle),
            Ok(Token::CloseParenthesis),
        ]
    );
    lexer_test!(
        n_tuples,
        "##NTUPLES= NMR SPECTRUM",
        [
            Ok(Token::Key),
            Ok(Token::Tuples),
            Ok(Token::Equals),
            Ok(Token::String),
            Ok(Token::String),
        ]
    );
    lexer_test!(
        page,
        "##PAGE= KEY=VALUE",
        [
            Ok(Token::Key),
            Ok(Token::Page),
            Ok(Token::Equals),
            Ok(Token::String),
            Ok(Token::Equals),
            Ok(Token::String),
        ]
    );
    lexer_test!(
        end,
        "##END=",
        [Ok(Token::Key), Ok(Token::End), Ok(Token::Equals),]
    );
    lexer_test!(
        end_ntuples,
        "##END NTUPLES= NMR SPECTRUM",
        [
            Ok(Token::Key),
            Ok(Token::End),
            Ok(Token::Equals),
            Ok(Token::String),
            Ok(Token::String),
        ]
    );
    lexer_test!(
        comments,
        "##TITLE= Example Title $$ comment",
        [
            Ok(Token::Key),
            Ok(Token::Title),
            Ok(Token::Equals),
            Ok(Token::String),
            Ok(Token::String),
        ]
    );
}
