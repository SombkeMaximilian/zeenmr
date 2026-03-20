use crate::jcampdx::tabulation::error::Error;
use crate::{Cursor, Location, UpdateCursor};
use logos::{Lexer, Logos};

/// JCAMP-DX grouped block lexer.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Logos)]
#[logos(extras = Cursor)]
#[logos(error(Error, invalid_literal))]
#[logos(subpattern newline = r"\n|\r\n|\r")]
#[logos(subpattern space = r"[ \t]")]
#[logos(skip r"(?&space)")]
#[logos(subpattern comment = r"\$\$[^\r\n]+")]
#[logos(skip r"(?&comment)")]
pub(crate) enum GroupToken {
    /// Every new line is a checkpoint.
    #[regex(r"(?&newline)", UpdateCursor::newline)]
    Checkpoint,
    /// Commas separate multiple values.
    #[token(",")]
    Comma,
    /// Semicolons separate groups.
    #[token(";")]
    Semicolon,
    /// Opening parenthesis to start a new group.
    #[token("(")]
    OpenParenthesis,
    /// Closing parenthesis to end a group.
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
    #[regex(r"[^ \t\r\n#,;<>\(\)]+")]
    String,
    /// A grouped block optionally ends with the next JCAMP-DX header key.
    #[token("##")]
    #[token("##.")]
    #[token("##$")]
    End,
}

/// Literals that could not be matched to any token.
fn invalid_literal(lexer: &Lexer<GroupToken>) -> Error {
    Error::invalid_literal(lexer.location())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jcampdx::tabulation::error::Result;

    macro_rules! lexer_test {
        ($name:ident, $data:expr, $tokens:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let expected = $tokens;
                let tokens = GroupToken::lexer(data).collect::<Vec<Result<GroupToken>>>();
                assert_eq!(tokens, expected);
            }
        };
    }

    lexer_test!(
        xypoints_newline,
        "(1, 1)\n\
         (2, 2)\n\
         (3, 3)\n\
         ##",
        [
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::End),
        ]
    );
    lexer_test!(
        xypoints_semicolon,
        "1, 1; 2, 2; 3, 3\n\
         ##",
        [
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Semicolon),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Semicolon),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::End),
        ]
    );
    lexer_test!(
        peak_table,
        "(1, 1, <5>)\n\
         (2, 2, <4>)\n\
         (3, 3, <7>)\n\
         ##",
        [
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::End),
        ]
    );
    lexer_test!(
        peak_assignment,
        "(1, 1, <CH3>)\n\
         (2, 2, <NH2>)\n\
         (3, 3, <OH>)\n\
         ##",
        [
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::String),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::String),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::OpenParenthesis),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::Numeric),
            Ok(GroupToken::Comma),
            Ok(GroupToken::OpenAngle),
            Ok(GroupToken::String),
            Ok(GroupToken::CloseAngle),
            Ok(GroupToken::CloseParenthesis),
            Ok(GroupToken::Checkpoint),
            Ok(GroupToken::End),
        ]
    );
}
