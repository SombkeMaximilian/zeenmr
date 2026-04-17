use logos::{Lexer, Logos};

/// Position of a token in the source.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Position {
    /// Line in the source.
    pub line: usize,
    /// Column in the source.
    pub column: usize,
}

/// File cursor for reporting locations in errors.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Cursor {
    /// Line in the source.
    pub(crate) line: usize,
    /// Character index in the source.
    pub(crate) index: usize,
}

/// Trait for updating the [`Cursor`] within a Lexer.
pub(crate) trait UpdateCursor {
    /// Updates the [`Cursor`] on a new line.
    fn newline(&mut self);
}

impl<'source, T> UpdateCursor for Lexer<'source, T>
where
    T: Logos<'source, Extras = Cursor>,
{
    fn newline(&mut self) {
        self.extras.line += 1;
        self.extras.index = self.span().end;
    }
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
