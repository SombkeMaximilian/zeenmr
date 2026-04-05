use crate::Cursor;
use crate::jcampdx::tabulation::error::{Error, Result};
use crate::jcampdx::tabulation::{GroupToken, TabulatedBlock, TabulatedBlockBuilder};
use crate::location::Location;
use logos::{Lexer, Logos};
use std::marker::PhantomData;
use std::num::IntErrorKind;

/// State machine of the [`TableParser`].
///
/// Tracks which kind of token is expected next.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
enum Awaits {
    /// Expecting a value token: [`Numeric`], [`String`] or [`OpenAngle`].
    ///
    /// [`Numeric`]: GroupToken::Numeric
    /// [`String`]: GroupToken::String
    /// [`OpenAngle`]: GroupToken::OpenAngle
    #[default]
    Value,
    /// Expecting a [`Comma`] token.
    ///
    /// [`Comma`]: GroupToken::Comma
    Comma,
    /// Expecting a group terminator token: [`Checkpoint`], [`Semicolon`],
    /// [`CloseParenthesis`], or one of the value tokens.
    ///
    /// [`Checkpoint`]: GroupToken::Checkpoint
    /// [`Semicolon`]: GroupToken::Semicolon
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    Terminator,
}

/// State machine of the [`TableParser`].
///
/// Tracks whether current inside parentheses or not.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
enum State {
    /// After a [`CloseParenthesis`] token or before any [`OpenParenthesis`]
    /// token.
    ///
    /// [`OpenParenthesis`]: GroupToken::OpenParenthesis
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    #[default]
    OutsideParentheses,
    /// After a [`OpenParenthesis`] token but before the next
    /// [`CloseParenthesis`] token
    ///
    /// [`OpenParenthesis`]: GroupToken::OpenParenthesis
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    InsideParentheses,
}

/// Pre-initialization marker.
#[derive(Debug)]
struct NeedsLayout;

/// Post-initialization marker.
#[derive(Debug)]
struct HasLayout;

/// Parser for tables in row-major, grouped format.
#[derive(Debug)]
pub(crate) struct TableParser<'source, L> {
    /// Lexer for the table format.
    lexer: Lexer<'source, GroupToken>,
    /// Next expected token.
    awaits: Awaits,
    /// Current state for end-of-group checks.
    state: State,
    /// Table being constructed.
    builder: TabulatedBlockBuilder,
    /// Initialization status.
    layout: PhantomData<L>,
}

impl<'source> From<&'source str> for TableParser<'source, NeedsLayout> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: GroupToken::lexer(value),
            awaits: Awaits::default(),
            state: State::default(),
            builder: TabulatedBlockBuilder::default(),
            layout: PhantomData,
        }
    }
}

impl<'source, T> From<Lexer<'source, T>> for TableParser<'source, NeedsLayout>
where
    T: Logos<'source, Source = str> + Clone,
    T::Extras: Clone + Into<Cursor>,
{
    fn from(value: Lexer<'source, T>) -> Self {
        Self {
            lexer: value.morph(),
            awaits: Awaits::default(),
            state: State::default(),
            builder: TabulatedBlockBuilder::default(),
            layout: PhantomData,
        }
    }
}

impl<'source, L> TableParser<'source, L> {
    /// Recovers the [`Lexer`] from the `TableParser`.
    pub(crate) fn into_lexer(self) -> Lexer<'source, GroupToken> {
        self.lexer
    }
}

impl<'source> TableParser<'source, NeedsLayout> {
    /// Initializes the parser with the column identifiers and group size.
    pub(crate) fn with_identifiers(
        mut self,
        identifiers: Vec<String>,
    ) -> TableParser<'source, HasLayout> {
        self.builder.set_columns(identifiers);

        TableParser::<'source, HasLayout> {
            lexer: self.lexer,
            awaits: Awaits::default(),
            state: State::default(),
            builder: self.builder,
            layout: PhantomData,
        }
    }
}

impl<'source> TableParser<'source, HasLayout> {
    /// Tabulates the source into a `TabulatedBlock`.
    ///
    /// # Errors
    ///
    /// Returns an error for fatal tabulation failures, such as invalid
    /// literals or an inconsistent number of columns.
    pub(crate) fn tabulate_source(&mut self) -> Result<TabulatedBlock> {
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                GroupToken::Checkpoint => self.check_point(),
                GroupToken::Comma => self.comma()?,
                GroupToken::Semicolon => self.semicolon()?,
                GroupToken::OpenParenthesis => self.open_parenthesis()?,
                GroupToken::CloseParenthesis => self.close_parenthesis()?,
                GroupToken::OpenAngle | GroupToken::Numeric | GroupToken::String => {
                    self.value_token(token)?
                }
                GroupToken::CloseAngle => self.close_angle()?,
                GroupToken::End => {
                    self.builder.header_key_exit();
                    break;
                }
            }
        }

        Ok(std::mem::take(&mut self.builder).finalize())
    }

    /// Returns `true` if the next value gets pushed to the first column.
    fn at_start_of_group(&self) -> bool {
        self.builder.current_column_index() == 0
    }

    /// Returns `true` if the next value gets pushed to the last column.
    fn at_end_of_group(&self) -> bool {
        self.builder.current_column_index() == (self.builder.row_len() - 1)
    }

    /// Handles [`Checkpoint`] tokens.
    ///
    /// [`Checkpoint`]: GroupToken::Checkpoint
    ///
    /// A newline character serves as a potential group separator. Groups within
    /// parentheses can extend past a newline, in which case this is a simple
    /// NOP.
    fn check_point(&mut self) {
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Value if self.at_start_of_group() => {}
            Awaits::Value | Awaits::Comma => match self.state {
                State::OutsideParentheses => self
                    .builder
                    .push_error(Error::cross_line_group(self.lexer.location())),
                State::InsideParentheses => {}
            },
            Awaits::Terminator => {}
        }
        self.awaits = Awaits::Value;
    }

    /// Handles [`Comma`] tokens.
    ///
    /// [`Comma`]: GroupToken::Comma
    ///
    /// Switches to awaiting a value and inserts an empty value in the previous
    /// column if already awaiting a value.
    ///
    /// # Errors
    ///
    /// Returns an error if the expected group size is exceeded.
    fn comma(&mut self) -> Result<()> {
        match self.awaits {
            Awaits::Value if !self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Comma => self.awaits = Awaits::Value,
            _ => return Err(Error::mismatched_group_size(self.lexer.location())),
        }

        Ok(())
    }

    /// Handles [`Semicolon`] tokens.
    ///
    /// [`Semicolon`]: GroupToken::Semicolon
    ///
    /// Checks if the current group was correctly finished and inserts a final
    /// empty value if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if the current group is not yet completed.
    fn semicolon(&mut self) -> Result<()> {
        if self.state == State::InsideParentheses {
            self.builder
                .push_error(Error::mismatched_group_delimiter(self.lexer.location()));
            self.state = State::OutsideParentheses;
        }
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Terminator => self.awaits = Awaits::Value,
            _ => return Err(Error::mismatched_group_size(self.lexer.location())),
        }

        Ok(())
    }

    /// Handles [`OpenParenthesis`] tokens.
    ///
    /// [`OpenParenthesis`]: GroupToken::OpenParenthesis
    ///
    /// Checks if the previous group was already finished and transitions to
    /// [`State::InsideParentheses`].
    ///
    /// # Errors
    ///
    /// Returns an error if already inside a pair of parentheses, or if the
    /// current group is not yet completed.
    fn open_parenthesis(&mut self) -> Result<()> {
        if self.state == State::InsideParentheses {
            return Err(Error::mismatched_group_delimiter(self.lexer.location()));
        }
        match self.awaits {
            Awaits::Value if self.at_start_of_group() => {}
            _ => return Err(Error::mismatched_group_delimiter(self.lexer.location())),
        }
        self.state = State::InsideParentheses;

        Ok(())
    }

    /// Handles [`CloseParenthesis`] tokens.
    ///
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    ///
    /// Checks if the current group was correctly finished and inserts a final
    /// empty value if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if there was no matching [`OpenParenthesis`] token, or
    /// if the current group is not yet completed.
    ///
    /// [`OpenParenthesis`]: GroupToken::OpenParenthesis
    fn close_parenthesis(&mut self) -> Result<()> {
        if self.state == State::OutsideParentheses {
            return Err(Error::mismatched_group_delimiter(self.lexer.location()));
        }
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Terminator => self.awaits = Awaits::Value,
            _ => return Err(Error::mismatched_group_size(self.lexer.location())),
        }
        self.state = State::OutsideParentheses;

        Ok(())
    }

    /// Handles the value tokens: [`OpenAngle`], [`Numeric`] and [`String`].
    ///
    /// [`OpenAngle`]: GroupToken::OpenAngle
    /// [`Numeric`]: GroupToken::Numeric
    /// [`String`]: GroupToken::String
    ///
    /// Parses a value and inserts it into the current column. Outside
    /// parentheses, completes the current group if necessary to allow
    /// whitespace to function as a group separator.
    ///
    /// In the case of an [`OpenAngle`] token, advances the lexer until finding
    /// either a matching [`CloseAngle`] token, or an [`End`] token, or the end
    /// of the input.
    ///
    /// [`OpenAngle`]: GroupToken::OpenAngle
    /// [`CloseAngle`]: GroupToken::CloseAngle
    /// [`End`]: GroupToken::End
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not separated from the previous one by
    /// a [`Comma`] token, or if the expected group size is exceeded, or if no
    /// matching [`CloseAngle`] token is encountered.
    ///
    /// [`Comma`]: GroupToken::Comma
    /// [`CloseAngle`]: GroupToken::CloseAngle
    ///
    /// # Panics
    ///
    /// Panics if a non-value token is passed. The caller is responsible for
    /// ensuring this does not happen.
    fn value_token(&mut self, token: GroupToken) -> Result<()> {
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.awaits = Awaits::Terminator,
            Awaits::Value => self.awaits = Awaits::Comma,
            Awaits::Comma => return Err(Error::non_separated_values(self.lexer.location())),
            Awaits::Terminator if self.state == State::OutsideParentheses => {
                self.awaits = Awaits::Comma
            }
            Awaits::Terminator => return Err(Error::mismatched_group_size(self.lexer.location())),
        }
        match token {
            GroupToken::OpenAngle => {
                let start = self.lexer.span().end;
                let mut found_closing = false;
                while let Some(token) = self.lexer.next().transpose()? {
                    match token {
                        GroupToken::CloseAngle => {
                            found_closing = true;
                            break;
                        }
                        GroupToken::End => break,
                        _ => continue,
                    }
                }
                if !found_closing {
                    return Err(Error::unmatched_string_delimiter(self.lexer.location()));
                }
                let end = self.lexer.span().start;
                let value = self.lexer.source()[start..end].trim();
                self.builder.push_string(value);
            }
            GroupToken::Numeric => match self.lexer.slice().parse::<i64>() {
                Ok(value) => self.builder.push_i64(value),
                Err(e) => match e.kind() {
                    IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                        self.builder
                            .push_error(Error::overflow(self.lexer.location()));
                        self.builder.push_i64(i64::MIN);
                    }
                    _ => {
                        let value = self
                            .lexer
                            .slice()
                            .parse::<f64>()
                            .expect("lexer (regex) should not match unparseable values");
                        if value.fract() == 0.0 {
                            self.builder.push_i64(value as i64);
                        } else {
                            self.builder.push_f64(value);
                        }
                    }
                },
            },
            GroupToken::String => self.builder.push_string(self.lexer.slice()),
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Handles [`CloseAngle`] tokens.
    ///
    /// [`CloseAngle`]: GroupToken::CloseAngle
    ///
    /// Since the [`OpenAngle`] token handler automatically looks for a matching
    /// [`CloseAngle`] token, reaching this point is always an error that
    /// cannot be recovered from, as everything before it would become a string.
    fn close_angle(&mut self) -> Result<()> {
        Err(Error::unmatched_string_delimiter(self.lexer.location()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jcampdx::{ChildParserExit, RawColumn, Table, Value};
    use std::sync::LazyLock;

    const IDENTIFIERS: [&str; 5] = ["X", "Y", "M", "S", "A"];

    static EXPECTED: LazyLock<TabulatedBlock> = LazyLock::new(|| {
        let mut table = Table::new();
        table.set_id("XYPOINTS");
        table.push(RawColumn::<f64> {
            id: "X".to_string(),
            values: vec![1.148, 1.226, 1.306, 2.610, 3.574, 3.651, 3.687, 3.727],
        });
        table.push(RawColumn::<i64> {
            id: "Y".to_string(),
            values: vec![209, 416, 205, 95, 63, 182, 167, 55],
        });
        table.push(RawColumn::<i64> {
            id: "M".to_string(),
            values: vec![3, 3, 3, 1, 4, 4, 4, 4],
        });
        table.push(RawColumn::<Value> {
            id: "S".to_string(),
            values: vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(1),
                Value::Integer(1),
                Value::String("1.00 - skewed".to_string()),
                Value::String("3.00 - skewed".to_string()),
                Value::String("0.87 - skewed".to_string()),
                Value::String("2.65 - skewed".to_string()),
            ],
        });
        table.push(RawColumn::<String> {
            id: "A".to_string(),
            values: ["CH3", "CH3", "CH3", "OH", "CH2", "CH2", "CH2", "CH2"]
                .into_iter()
                .map(ToString::to_string)
                .collect(),
        });

        TabulatedBlock {
            exit: ChildParserExit::EndOfInput,
            table,
            errors: Vec::new(),
        }
    });

    macro_rules! parser_test {
        ($name:ident, $data:expr) => {
            #[test]
            fn $name() {
                let data = $data;
                let tabulated = TableParser::from(data)
                    .with_identifiers(IDENTIFIERS.map(ToString::to_string).to_vec())
                    .tabulate_source()
                    .unwrap();
                assert_eq!(tabulated, *EXPECTED);
            }
        };
    }

    parser_test!(
        semicolon_separated_multiple_per_line,
        "\
            1.148, 209, 3,               1, CH3; 1.226, 416, 3,               2, CH3\n\
            1.306, 205, 3,               1, CH3; 2.610,  95, 1,               1,  OH\n\
            3.574,  63, 4, <1.00 - skewed>, CH2; 3.651, 182, 4, <3.00 - skewed>, CH2\n\
            3.687, 167, 4, <0.87 - skewed>, CH2; 3.727,  55, 4, <2.65 - skewed>, CH2"
    );
    parser_test!(
        whitespace_separated_multiple_per_line,
        "\
            1.148, 209, 3,               1, CH3  1.226, 416, 3,               2, CH3\n\
            1.306, 205, 3,               1, CH3  2.610,  95, 1,               1,  OH\n\
            3.574,  63, 4, <1.00 - skewed>, CH2  3.651, 182, 4, <3.00 - skewed>, CH2\n\
            3.687, 167, 4, <0.87 - skewed>, CH2  3.727,  55, 4, <2.65 - skewed>, CH2"
    );
    parser_test!(
        parenthesis_enclosed_multiple_per_line,
        "\
            (1.148, 209, 3,               1, CH3) (1.226, 416, 3,               2, CH3)\n\
            (1.306, 205, 3,               1, CH3) (2.610,  95, 1,               1,  OH)\n\
            (3.574,  63, 4, <1.00 - skewed>, CH2) (3.651, 182, 4, <3.00 - skewed>, CH2)\n\
            (3.687, 167, 4, <0.87 - skewed>, CH2) (3.727,  55, 4, <2.65 - skewed>, CH2)"
    );
    parser_test!(
        semicolon_separated_one_per_line,
        "\
            1.148, 209, 3,               1, CH3;\n\
            1.226, 416, 3,               2, CH3;\n\
            1.306, 205, 3,               1, CH3;\n\
            2.610,  95,  1,              1,  OH;\n\
            3.574,  63, 4, <1.00 - skewed>, CH2;\n\
            3.651, 182, 4, <3.00 - skewed>, CH2;\n\
            3.687, 167, 4, <0.87 - skewed>, CH2;\n\
            3.727,  55, 4, <2.65 - skewed>, CH2;"
    );
    parser_test!(
        whitespace_separated_one_per_line,
        "\
            1.148, 209, 3,               1, CH3\n\
            1.226, 416, 3,               2, CH3\n\
            1.306, 205, 3,               1, CH3\n\
            2.610,  95, 1,               1,  OH\n\
            3.574,  63, 4, <1.00 - skewed>, CH2\n\
            3.651, 182, 4, <3.00 - skewed>, CH2\n\
            3.687, 167, 4, <0.87 - skewed>, CH2\n\
            3.727,  55, 4, <2.65 - skewed>, CH2"
    );
    parser_test!(
        parenthesis_enclosed_one_per_line,
        "\
            (1.148, 209, 3,               1, CH3)\n\
            (1.226, 416, 3,               2, CH3)\n\
            (1.306, 205, 3,               1, CH3)\n\
            (2.610,  95, 1,               1,  OH)\n\
            (3.574,  63, 4, <1.00 - skewed>, CH2)\n\
            (3.651, 182, 4, <3.00 - skewed>, CH2)\n\
            (3.687, 167, 4, <0.87 - skewed>, CH2)\n\
            (3.727,  55, 4, <2.65 - skewed>, CH2)"
    );
}
