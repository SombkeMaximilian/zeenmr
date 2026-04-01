use std::marker::PhantomData;
use crate::Cursor;
use crate::jcampdx::tabulation::error::{Error, Result};
use crate::jcampdx::tabulation::{GroupToken, TabulatedBlock, TabulatedBlockBuilder};
use logos::{Lexer, Logos};
use crate::location::Location;
use std::num::IntErrorKind;

/// State of the [`TableParser`].
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
    /// Current state for end-of-group checks.
    state: State,
    /// Table being constructed.
    builder: TabulatedBlockBuilder,
    /// Expected size of the groups.
    group_size: usize,
    /// Values are separated by commas.
    comma_count: usize,
    /// Initialization status.
    layout: PhantomData<L>,
}

impl<'source> From<&'source str> for TableParser<'source, NeedsLayout> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: GroupToken::lexer(value),
            state: State::default(),
            builder: TabulatedBlockBuilder::default(),
            group_size: 0,
            comma_count: 0,
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
            state: State::default(),
            builder: TabulatedBlockBuilder::default(),
            group_size: 0,
            comma_count: 0,
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
    pub(crate) fn with_identifiers(mut self, identifiers: Vec<String>) -> TableParser<'source, HasLayout> {
        self.group_size = identifiers.len();
        self.builder.set_columns(identifiers);

        TableParser::<'source, HasLayout> {
            lexer: self.lexer,
            state: State::default(),
            builder: self.builder,
            group_size: self.group_size,
            comma_count: self.comma_count,
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
                GroupToken::Checkpoint => self.check_point()?,
                GroupToken::Comma => self.comma()?,
                GroupToken::Semicolon => self.semicolon()?,
                GroupToken::OpenParenthesis => self.open_parenthesis()?,
                GroupToken::CloseParenthesis => self.close_parenthesis()?,
                GroupToken::OpenAngle => self.open_angle()?,
                GroupToken::CloseAngle => self.close_angle()?,
                GroupToken::Numeric => self.numeric()?,
                GroupToken::String => self.string()?,
                GroupToken::End => {
                    self.builder.header_key_exit();
                    break;
                }
            }
        }

        Ok(std::mem::take(&mut self.builder).finalize())
    }

    /// Handles [`Checkpoint`] tokens.
    ///
    /// [`Checkpoint`]: GroupToken::Checkpoint
    ///
    /// A newline character serves as a potential group separator. Groups within
    /// parentheses can extend past a newline, in which case this is a simple
    /// NOP.
    ///
    /// # Errors
    ///
    /// Returns an error if the comma count plus one is greater than the
    /// expected group size.
    fn check_point(&mut self) -> Result<()> {
        match self.state {
            State::OutsideParentheses if self.comma_count + 1 == self.group_size => {
                if self.builder.current_column_index() != 0 {
                    self.builder.skip_current();
                    debug_assert_eq!(self.builder.current_column_index(), 0);
                }
                self.comma_count = 0;
            }
            State::OutsideParentheses if self.comma_count + 1 < self.group_size => {
                self.builder.push_error(Error::cross_line_group(self.lexer.location()));
                self.comma_count += 1;
            }
            State::OutsideParentheses if self.comma_count + 1 > self.group_size => {
                return Err(Error::mismatched_group_size(self.lexer.location()));
            }
            State::OutsideParentheses => unreachable!(),
            State::InsideParentheses => {}
        }

        Ok(())
    }

    /// Handles [`Comma`] tokens.
    ///
    /// [`Comma`]: GroupToken::Comma
    ///
    /// Increments the comma counter and inserts an empty value in the previous
    /// column if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if the comma count is equal to or greater than the
    /// expected group size.
    fn comma(&mut self) -> Result<()> {
        self.comma_count += 1;
        if self.comma_count > self.builder.current_column_index() {
            self.builder.skip_current();
        }
        debug_assert_eq!(self.comma_count, self.builder.current_column_index());

        if self.comma_count < self.group_size {
            Ok(())
        } else {
            Err(Error::mismatched_group_size(self.lexer.location()))
        }
    }

    /// Handles [`Semicolon`] tokens.
    ///
    /// [`Semicolon`]: GroupToken::Semicolon
    ///
    /// Checks if the current group was correctly finished, inserts a final
    /// empty value if necessary, and resets the comma counter.
    ///
    /// # Errors
    ///
    /// Returns an error if the comma count plus one is not equal to the
    /// expected group size, or if the current group was started by an
    /// opening parenthesis which has not yet been closed.
    fn semicolon(&mut self) -> Result<()> {
        if self.comma_count + 1 != self.group_size {
            return Err(Error::mismatched_group_size(self.lexer.location()));
        }
        if self.builder.current_column_index() != 0 {
            self.builder.skip_current();
            debug_assert_eq!(self.builder.current_column_index(), 0);
        }
        if self.state == State::InsideParentheses {
            self.builder.push_error(Error::mismatched_group_terminator(self.lexer.location()));
        }
        self.state = State::OutsideParentheses;
        self.comma_count = 0;

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
    /// Returns an error if it is called before fully finishing the previous
    /// group by encountering a [`CloseParenthesis`] token.
    ///
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    fn open_parenthesis(&mut self) -> Result<()> {
        if self.comma_count != 0 || self.builder.current_column_index() != 0 {
            return Err(Error::mismatched_group_size(self.lexer.location()));
        }
        if self.state == State::InsideParentheses {
            self.builder.push_error(Error::mismatched_group_terminator(self.lexer.location()));
        }
        self.state = State::InsideParentheses;

        Ok(())
    }

    /// Handles [`CloseParenthesis`] tokens.
    ///
    /// [`CloseParenthesis`]: GroupToken::CloseParenthesis
    ///
    /// Checks if the current group was correctly finished, inserts a final
    /// empty value if necessary, and resets the comma counter.
    ///
    /// # Errors
    ///
    /// Returns an error if the comma count plus one is not equal to the
    /// expected group size.
    fn close_parenthesis(&mut self) -> Result<()> {
        if self.comma_count + 1 != self.group_size {
            return Err(Error::mismatched_group_size(self.lexer.location()));
        }
        if self.builder.current_column_index() != 0 {
            self.builder.skip_current();
            debug_assert_eq!(self.builder.current_column_index(), 0);
        }
        if self.state == State::OutsideParentheses {
            self.builder.push_error(Error::mismatched_group_terminator(self.lexer.location()));
        }
        self.state = State::OutsideParentheses;
        self.comma_count = 0;

        Ok(())
    }

    /// Handles [`OpenAngle`] tokens.
    ///
    /// [`OpenAngle`]: GroupToken::OpenAngle
    ///
    /// Advances the lexer until finding either a matching [`CloseAngle`] token,
    /// or an [`End`] token, or the end of the input.
    ///
    /// [`CloseAngle`]: GroupToken::CloseAngle
    /// [`End`]: GroupToken::End
    ///
    /// # Errors
    ///
    /// Returns an error if no matching [`CloseAngle`] token is encountered.
    fn open_angle(&mut self) -> Result<()> {
        let start = self.lexer.span().end;
        let mut found_closing = false;
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                GroupToken::CloseAngle => {
                    found_closing = true;
                    break;
                },
                GroupToken::End => break,
                _ => continue,
            }
        }
        if !found_closing {
            return Err(Error::unmatched_string_delimiter(self.lexer.location()));
        }
        let end = self.lexer.span().start;
        self.builder.push_string(self.lexer.source()[start..end].trim());

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

    /// Handles [`Numeric`] tokens.
    ///
    /// [`Numeric`]: GroupToken::Numeric
    ///
    /// Checks if the current value is properly separated from the previous one
    /// and inserts it into the current column. Outside parentheses, completes
    /// the current group if necessary to allow whitespace to function as a
    /// group separator.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not properly separated from the
    /// previous one, or if the current group is already completed.
    fn numeric(&mut self) -> Result<()> {
        if self.comma_count != self.builder.current_column_index() {
            return Err(Error::non_separated_values(self.lexer.location()));
        }

        match self.state {
            State::OutsideParentheses if self.comma_count + 1 <= self.group_size => {
                if self.comma_count + 1 == self.group_size {
                    self.comma_count = 0;
                }
            }
            State::InsideParentheses if self.comma_count + 1 < self.group_size => {}
            _ => return Err(Error::mismatched_group_size(self.lexer.location())),
        }

        match self.lexer.slice().parse::<i64>() {
            Ok(value) => self.builder.push_i64(value),
            Err(e) => match e.kind() {
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                    self.builder.push_error(Error::overflow(self.lexer.location()));
                    self.builder.push_i64(i64::MIN);
                }
                _ => {
                    let value = self.lexer
                        .slice()
                        .parse::<f64>()
                        .expect("lexer (regex) should not match unparseable values");
                    if value.fract() == 0.0 {
                        self.builder.push_i64(value as i64);
                    } else {
                        self.builder.push_f64(value);
                    }
                }
            }
        }

        Ok(())
    }

    /// Handles [`String`] tokens.
    ///
    /// [`String`]: GroupToken::String
    ///
    /// Checks if the current value is properly separated from the previous one
    /// and inserts it into the current column. Outside parentheses, completes
    /// the current group if necessary to allow whitespace to function as a
    /// group separator.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not properly separated from the
    /// previous one, or if the current group is already completed.
    fn string(&mut self) -> Result<()> {
        if self.comma_count != self.builder.current_column_index() {
            return Err(Error::non_separated_values(self.lexer.location()));
        }

        match self.state {
            State::OutsideParentheses if self.comma_count + 1 <= self.group_size => {
                if self.comma_count + 1 == self.group_size {
                    self.comma_count = 0;
                }
                self.builder.push_string(self.lexer.slice());
            }
            State::InsideParentheses if self.comma_count + 1 < self.group_size => {
                self.builder.push_string(self.lexer.slice());
            }
            _ => return Err(Error::mismatched_group_size(self.lexer.location())),
        }

        Ok(())
    }
}
