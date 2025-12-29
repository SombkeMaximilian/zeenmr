use logos::{Lexer, Logos};

/// File cursor for reporting locations in errors.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Cursor {
    /// Line in the source.
    pub(crate) line: usize,
    /// Character index in the source.
    pub(crate) index: usize,
}

/// Position of a token in the source.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Position {
    /// Line in the source.
    pub(crate) line: usize,
    /// Column in the source.
    pub(crate) column: usize,
}

/// Trait for getting the current location in the source.
pub(crate) trait Location {
    /// Returns the current position in the source.
    fn location(&self) -> Position;
}

impl<'source, T> Location for Lexer<'source, T>
where
    T: Logos<'source, Extras = Cursor>,
{
    fn location(&self) -> Position {
        Position {
            line: self.extras.line,
            column: self.span().start - self.extras.index,
        }
    }
}
