use crate::jcampdx::tabulation::error::{Error, Result};
use crate::jcampdx::tabulation::{GroupToken, TabulatedBlock, TabulatedBlockBuilder};
use logos::{Lexer, Logos};
use std::borrow::Cow;
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
pub(crate) struct NeedsLayout;

/// Post-initialization marker.
#[derive(Debug)]
pub(crate) struct HasLayout;

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
    builder: TabulatedBlockBuilder<'source>,
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
    T: Logos<'source, Source = str, Extras = ()>,
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

    /// Sets the title of the dataset.
    pub(crate) fn set_title<T: Into<Cow<'source, str>>>(&mut self, title: T) {
        self.builder.set_title(title);
    }
}

impl<'source> TableParser<'source, NeedsLayout> {
    /// Initializes the parser with the column identifiers and group size.
    pub(crate) fn with_identifiers(
        mut self,
        identifiers: Vec<&'source str>,
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
    pub(crate) fn tabulate_source(&mut self) -> Result<TabulatedBlock<'source>> {
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
        if self.awaits == Awaits::Value && self.at_end_of_group() {
            self.builder.skip_current();
        }
        if !self.at_start_of_group() {
            return Err(Error::mismatched_group_size(self.lexer.span().into()));
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
    /// parentheses can extend past a newline. Always transitions to awaiting a
    /// value.
    fn check_point(&mut self) {
        if self.state == State::OutsideParentheses {
            match self.awaits {
                Awaits::Value if self.at_start_of_group() => {}
                Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
                Awaits::Value | Awaits::Comma => self
                    .builder
                    .push_error(Error::cross_line_group(self.lexer.span().into())),
                Awaits::Terminator => {}
            }
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
            _ => return Err(Error::mismatched_group_size(self.lexer.span().into())),
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
                .push_error(Error::mismatched_group_delimiter(self.lexer.span().into()));
            self.state = State::OutsideParentheses;
        }
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Terminator => self.awaits = Awaits::Value,
            _ => return Err(Error::mismatched_group_size(self.lexer.span().into())),
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
            return Err(Error::mismatched_group_delimiter(self.lexer.span().into()));
        }
        match self.awaits {
            Awaits::Value if self.at_start_of_group() => {}
            _ => return Err(Error::mismatched_group_delimiter(self.lexer.span().into())),
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
            return Err(Error::mismatched_group_delimiter(self.lexer.span().into()));
        }
        match self.awaits {
            Awaits::Value if self.at_end_of_group() => self.builder.skip_current(),
            Awaits::Terminator => self.awaits = Awaits::Value,
            _ => return Err(Error::mismatched_group_size(self.lexer.span().into())),
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
            Awaits::Comma => return Err(Error::non_separated_values(self.lexer.span().into())),
            Awaits::Terminator if self.state == State::OutsideParentheses => {
                self.awaits = Awaits::Comma
            }
            Awaits::Terminator => {
                return Err(Error::mismatched_group_size(self.lexer.span().into()));
            }
        }
        match token {
            GroupToken::OpenAngle => {
                let start = self.lexer.span().end;
                let mut found_closing = false;
                while let Some(token) = self.lexer.next().transpose()? {
                    match token {
                        GroupToken::OpenAngle => {
                            return Err(Error::unmatched_string_delimiter(
                                self.lexer.span().into(),
                            ));
                        }
                        GroupToken::CloseAngle => {
                            found_closing = true;
                            break;
                        }
                        GroupToken::End => break,
                        _ => continue,
                    }
                }
                if !found_closing {
                    return Err(Error::unmatched_string_delimiter(self.lexer.span().into()));
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
                            .push_error(Error::overflow(self.lexer.span().into()));
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
        Err(Error::unmatched_string_delimiter(self.lexer.span().into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataTable, Value};
    use crate::error::ByteRange;
    use crate::jcampdx::parser::ChildParserExit;
    use std::sync::LazyLock;

    static EXPECTED: LazyLock<TabulatedBlock> = LazyLock::new(|| {
        let mut table = DataTable::new_with_id("XYPOINTS");
        table.insert(
            "X".into(),
            vec![1.148, 1.226, 1.306, 2.610, 3.574, 3.651, 3.687, 3.727].into(),
        );
        table.insert("Y".into(), vec![209, 416, 205, 95, 63, 182, 167, 55].into());
        table.insert("M".into(), vec![3, 3, 3, 1, 4, 4, 4, 4].into());
        table.insert(
            "S".into(),
            vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(1),
                Value::Integer(1),
                Value::from("1.00 - skewed"),
                Value::from("3.00 - skewed"),
                Value::from("0.87 - skewed"),
                Value::from("2.65 - skewed"),
            ]
            .into(),
        );
        table.insert(
            "A".into(),
            ["CH3", "CH3", "CH3", "OH", "CH2", "CH2", "CH2", "CH2"]
                .into_iter()
                .collect(),
        );

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
                let mut tabulator =
                    TableParser::from(data).with_identifiers(vec!["X", "Y", "M", "S", "A"]);
                let tabulated = tabulator.tabulate_source().unwrap();
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
    parser_test!(
        parenthesis_enclosed_cross_line,
        "\
            (1.148, 209, 3, 1, CH3) (1.226, 416, 3, 2,   \n\
            CH3) (1.306, 205, 3, 1, CH3) (2.610, 95,     \n\
            1, 1, OH) (3.574,  63, 4, <1.00 - skewed>,   \n\
            CH2) (3.651, 182, 4, <3.00 - skewed>, CH2)   \n\
            (3.687, 167, 4, <0.87 - skewed>, CH2) (3.727,\n\
            55, 4, <2.65 - skewed>, CH2)"
    );

    macro_rules! fatal_error_test {
        ($name:ident, $columns:expr, $data:expr, $error:expr) => {
            #[test]
            fn $name() {
                let identifiers = std::iter::repeat("A").take($columns).collect();
                let data = $data;
                let error = $error;
                let tabulated = TableParser::from(data)
                    .with_identifiers(identifiers)
                    .tabulate_source()
                    .unwrap_err();
                assert_eq!(tabulated, error);
            }
        };
    }

    fatal_error_test!(
        group_too_large_semicolon,
        2,
        "\
            1, 1; 2, 2; 3, 3\n\
            4, 4; 5, 5, 5;",
        Error::mismatched_group_size(ByteRange::new(27, 28))
    );
    fatal_error_test!(
        group_too_large_whitespace,
        2,
        "\
            1, 1  2, 2  3, 3\n\
            4, 4  5, 5, 5",
        Error::mismatched_group_size(ByteRange::new(27, 28))
    );
    fatal_error_test!(
        group_too_large_parentheses,
        2,
        "\
            (1, 1) (2, 2) (3, 3)\n\
            (4, 4) (5, 5, 5)",
        Error::mismatched_group_size(ByteRange::new(33, 34))
    );
    fatal_error_test!(
        group_too_small_semicolon,
        3,
        "\
            1, 1, 1; 2, 2, 2; 3, 3, 3\n\
            4, 4, 4; 5, 5;",
        Error::mismatched_group_size(ByteRange::new(39, 40))
    );
    fatal_error_test!(
        group_too_small_whitespace,
        3,
        "\
            1, 1, 1  2, 2, 2  3, 3, 3\n\
            4, 4, 4  5, 5",
        Error::mismatched_group_size(ByteRange::new(39, 39))
    );
    fatal_error_test!(
        group_too_small_parentheses,
        3,
        "\
            (1, 1, 1) (2, 2, 2) (3, 3, 3)\n\
            (4, 4, 4) (5, 5)",
        Error::mismatched_group_size(ByteRange::new(45, 46))
    );

    fatal_error_test!(
        mismatched_group_delimiter_unclosed_parenthesis,
        2,
        "\
            (1, 1) (2, 2  (3, 3)\n\
            (4, 4) (5, 5)",
        Error::mismatched_group_delimiter(ByteRange::new(14, 15))
    );
    fatal_error_test!(
        mismatched_group_delimiter_unopened_parenthesis,
        2,
        "\
            (1, 1)  2, 2) (3, 3)\n\
            (4, 4) (5, 5)",
        Error::mismatched_group_delimiter(ByteRange::new(12, 13))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unclosed_angle_semicolon,
        3,
        "\
            1, 1, <1>; 2, 2, <2>; 3, 3, <3>\n\
            4, 4, <4>; 5, 5, <5 ;",
        Error::unmatched_string_delimiter(ByteRange::new(53, 53))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unopened_angle_semicolon,
        3,
        "\
            1, 1,  1>; 2, 2, <2>; 3, 3, <3>\n\
            4, 4, <4>; 5, 5, <5>;",
        Error::unmatched_string_delimiter(ByteRange::new(8, 9))
    );
    fatal_error_test!(
        unmatched_string_delimiter_multiple_opened_angle_semicolon,
        3,
        "\
            1, 1, <1 ; 2, 2, <2>; 3, 3, <3>\n\
            4, 4, <4>; 5, 5, <5>;",
        Error::unmatched_string_delimiter(ByteRange::new(17, 18))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unclosed_angle_parentheses,
        3,
        "\
            (1, 1, <1>) (2, 2, <2>) (3, 3, <3>)\n\
            (4, 4, <4>) (5, 5, <5 )",
        Error::unmatched_string_delimiter(ByteRange::new(59, 59))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unopened_angle_parentheses,
        3,
        "\
            (1, 1,  1>) (2, 2, <2>) (3, 3, <3>)\n\
            (4, 4, <4>) (5, 5, <5>)",
        Error::unmatched_string_delimiter(ByteRange::new(9, 10))
    );
    fatal_error_test!(
        unmatched_string_delimiter_multiple_opened_angle_parentheses,
        3,
        "\
            (1, 1, <1 ) (2, 2, <2>) (3, 3, <3>)\n\
            (4, 4, <4>) (5, 5, <5>)",
        Error::unmatched_string_delimiter(ByteRange::new(19, 20))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unclosed_angle_whitespace,
        3,
        "\
            1, 1, <1> 2, 2, <2> 3, 3, <3>\n\
            4, 4, <4> 5, 5, <5 ",
        Error::unmatched_string_delimiter(ByteRange::new(49, 49))
    );
    fatal_error_test!(
        unmatched_string_delimiter_unopened_angle_whitespace,
        3,
        "\
            1, 1,  1> 2, 2, <2> 3, 3, <3>\n\
            4, 4, <4> 5, 5, <5>",
        Error::unmatched_string_delimiter(ByteRange::new(8, 9))
    );
    fatal_error_test!(
        unmatched_string_delimiter_multiple_opened_angle_whitespace,
        3,
        "\
            1, 1, <1  2, 2, <2> 3, 3, <3>\n\
            4, 4, <4> 5, 5, <5>",
        Error::unmatched_string_delimiter(ByteRange::new(16, 17))
    );
    fatal_error_test!(
        non_separated_values_semicolon,
        2,
        "\
            1, 1; 2  2; 3, 3\n\
            4, 4; 5, 5",
        Error::non_separated_values(ByteRange::new(9, 10))
    );
    fatal_error_test!(
        non_separated_values_parentheses,
        2,
        "\
            (1, 1) (2  2) (3, 3)\n\
            (4, 4) (5, 5)",
        Error::non_separated_values(ByteRange::new(11, 12))
    );
    fatal_error_test!(
        non_separated_values_whitespace,
        2,
        "\
            1, 1 2  2 3, 3\n\
            4, 4 5, 5",
        Error::non_separated_values(ByteRange::new(8, 9))
    );

    macro_rules! recoverable_error_test {
        ($name:ident, $columns:expr, $data:expr, $errors:expr) => {
            #[test]
            fn $name() {
                let identifiers = std::iter::repeat("A").take($columns).collect();
                let data = $data;
                let errors = $errors;
                let mut tabulator = TableParser::from(data).with_identifiers(identifiers);
                let tabulated = tabulator.tabulate_source().unwrap();
                assert_eq!(tabulated.errors, errors)
            }
        };
    }

    recoverable_error_test!(
        overflow_semicolon,
        2,
        "1, 10000000000000000000; 1, -10000000000000000000",
        [
            Error::overflow(ByteRange::new(3, 23)),
            Error::overflow(ByteRange::new(28, 49)),
        ]
    );
    recoverable_error_test!(
        overflow_parentheses,
        2,
        "(1, 10000000000000000000) (1, -10000000000000000000)",
        [
            Error::overflow(ByteRange::new(4, 24)),
            Error::overflow(ByteRange::new(30, 51)),
        ]
    );
    recoverable_error_test!(
        overflow_whitespace,
        2,
        "1, 10000000000000000000 1, -10000000000000000000",
        [
            Error::overflow(ByteRange::new(3, 23)),
            Error::overflow(ByteRange::new(27, 48)),
        ]
    );
    recoverable_error_test!(
        semicolon_closes_parenthesis,
        2,
        "(1, 1) (2, 2; (3, 3)",
        [Error::mismatched_group_delimiter(ByteRange::new(12, 13))]
    );
}
