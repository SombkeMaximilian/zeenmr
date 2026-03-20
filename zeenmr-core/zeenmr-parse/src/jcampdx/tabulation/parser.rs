use crate::Cursor;
use crate::jcampdx::tabulation::error::{Error, Result};
use crate::jcampdx::tabulation::{GroupToken, TabulatedBlock, TabulatedBlockBuilder};
use logos::{Lexer, Logos};

/// Expected layout of the lines.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
enum LineLayout {
    /// Multiple groups per line.
    #[default]
    MultiGroup,
    /// One group per line.
    SingleGroup,
}

/// Parser for tables row-major, grouped format.
#[derive(Debug)]
pub(crate) struct TableParser<'source> {
    /// Lexer for the table format.
    lexer: Lexer<'source, GroupToken>,
    /// Table being constructed.
    builder: TabulatedBlockBuilder,
    /// Expected layout of the lines.
    layout: LineLayout,
    /// Expected size of the groups.
    group_size: usize,
    /// Start of string enclosed by angle brackets.
    string_start: Option<usize>,
}

impl<'source> From<&'source str> for TableParser<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: GroupToken::lexer(value),
            builder: TabulatedBlockBuilder::default(),
            layout: LineLayout::default(),
            group_size: 0,
            string_start: None,
        }
    }
}

impl<'source, T> From<Lexer<'source, T>> for TableParser<'source>
where
    T: Logos<'source, Source = str> + Clone,
    T::Extras: Clone + Into<Cursor>,
{
    fn from(value: Lexer<'source, T>) -> Self {
        Self {
            lexer: value.morph(),
            builder: TabulatedBlockBuilder::default(),
            layout: LineLayout::default(),
            group_size: 0,
            string_start: None,
        }
    }
}

impl<'source> TableParser<'source> {
    /// Recovers the [`Lexer`] from the `TableParser`.
    pub(crate) fn into_lexer(self) -> Lexer<'source, GroupToken> {
        self.lexer
    }

    pub(crate) fn set_multi_group_identifiers(&mut self, identifiers: Vec<String>) {
        self.layout = LineLayout::MultiGroup;
        self.group_size = identifiers.len();
        self.builder.set_columns(identifiers);
    }

    pub(crate) fn set_single_group_identifiers(&mut self, identifiers: Vec<String>) {
        self.layout = LineLayout::SingleGroup;
        self.group_size = identifiers.len();
        self.builder.set_columns(identifiers);
    }

    /// Tabulates the source into a `TabulatedBlock`.
    ///
    /// # Errors
    ///
    /// Returns an error for fatal tabulation failures, such as invalid
    /// literals or an inconsistent number of columns.
    pub(crate) fn tabulate_source(&mut self) -> Result<TabulatedBlock> {
        debug_assert!(self.group_size > 0);
        while let Some(token) = self.lexer.next().transpose()? {
            match token {
                GroupToken::Checkpoint => {}
                GroupToken::Comma => {}
                GroupToken::Semicolon => {}
                GroupToken::OpenParenthesis => {}
                GroupToken::CloseParenthesis => {}
                GroupToken::OpenAngle => {}
                GroupToken::CloseAngle => {}
                GroupToken::Numeric => {}
                GroupToken::String => {}
                GroupToken::End => {}
            }
        }

        Ok(std::mem::take(&mut self.builder).finalize())
    }
}
