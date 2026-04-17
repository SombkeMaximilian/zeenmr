use logos::{Lexer, Logos};
use std::ops::Range;

/// Position of a token in the source.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Position {
    /// Start of the byte range.
    pub start: usize,
    /// End of the byte range.
    pub end: usize,
    /// Line number.
    pub line: usize,
}

impl Position {
    /// Constructs a new `Position` from a byte range and a line number.
    pub fn new(byte_range: Range<usize>, line: usize) -> Self {
        Self {
            start: byte_range.start,
            end: byte_range.end,
            line,
        }
    }

    /// Returns the byte range in the source.
    pub fn byte_range(&self) -> Range<usize> {
        self.start..self.end
    }
}

/// Line counter for reporting locations in errors.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct LineCounter {
    /// Line in the source.
    pub(crate) line: usize,
}

/// Trait for updating the [`LineCounter`] within a Lexer.
pub(crate) trait UpdateLineCounter {
    /// Updates the [`LineCounter`] on a new line.
    fn newline(&mut self);
}

impl<'source, T> UpdateLineCounter for Lexer<'source, T>
where
    T: Logos<'source, Extras = LineCounter>,
{
    fn newline(&mut self) {
        self.extras.line += 1;
    }
}

/// Trait for getting the current location in the source.
pub(crate) trait CurrentPosition {
    /// Returns the current position in the source.
    fn position(&self) -> Position;
}

impl<'source, T> CurrentPosition for Lexer<'source, T>
where
    T: Logos<'source, Extras = LineCounter>,
{
    fn position(&self) -> Position {
        let span = self.span();

        Position {
            start: span.start,
            end: span.end,
            line: self.extras.line,
        }
    }
}
