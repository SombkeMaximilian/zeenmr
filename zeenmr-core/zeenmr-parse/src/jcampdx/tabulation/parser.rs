use crate::Cursor;
use crate::jcampdx::tabulation::{GroupToken, TabulatedBlockBuilder};
use logos::{Lexer, Logos};

/// Parser for tables row-major, grouped format.
#[derive(Debug)]
pub(crate) struct TableParser<'source> {
    /// Lexer for the table format.
    lexer: Lexer<'source, GroupToken>,
    /// Table being constructed.
    builder: TabulatedBlockBuilder,
}

impl<'source> From<&'source str> for TableParser<'source> {
    fn from(value: &'source str) -> Self {
        Self {
            lexer: GroupToken::lexer(value),
            builder: TabulatedBlockBuilder::default(),
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
        }
    }
}

impl<'source> TableParser<'source> {
    /// Recovers the [`Lexer`] from the `TableParser`.
    pub(crate) fn into_lexer(self) -> Lexer<'source, GroupToken> {
        self.lexer
    }
}
