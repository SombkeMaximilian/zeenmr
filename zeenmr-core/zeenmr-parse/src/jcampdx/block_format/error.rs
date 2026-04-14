//! JCAMP-DX block format parsing error types.

use crate::Position;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    kind: Kind,
    position: Position,
}

#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    #[default]
    InvalidLiteral,
    EndOfInput,
    EmptyFormat,
    EmptyIncrement,
    EmptyRepeat,
    MismatchedRepeat,
    MultipleIncrement,
    MultipleRepeat,
    IncrementAfterRepeat,
    MultipleIdentifierIncrement,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::InvalidLiteral => "invalid literal",
            Kind::EndOfInput => "end of input",
            Kind::EmptyFormat => "empty format",
            Kind::EmptyIncrement => "empty input",
            Kind::EmptyRepeat => "empty group",
            Kind::MismatchedRepeat => "mismatched repeat",
            Kind::MultipleIncrement => "multiple increment",
            Kind::MultipleRepeat => "multiple block kind",
            Kind::IncrementAfterRepeat => "increment after repeat",
            Kind::MultipleIdentifierIncrement => "multiple identifier increment",
        };

        write!(f, "{description}")
    }
}

impl Error {
    /// Creates an [`InvalidLiteral`] error.
    ///
    /// [`InvalidLiteral`]: Kind::InvalidLiteral
    pub(crate) fn invalid_literal(position: Position) -> Self {
        Self {
            kind: Kind::InvalidLiteral,
            position,
        }
    }

    /// Creates an [`EndOfInput`] error.
    ///
    /// [`EndOfInput`]: Kind::EndOfInput
    pub(crate) fn end_of_input(position: Position) -> Self {
        Self {
            kind: Kind::EndOfInput,
            position,
        }
    }

    /// Creates an [`EmptyFormat`] error.
    ///
    /// [`EmptyFormat`]: Kind::EmptyFormat
    pub(crate) fn empty_format(position: Position) -> Self {
        Self {
            kind: Kind::EmptyFormat,
            position,
        }
    }

    /// Creates an [`EmptyIncrement`] error.
    ///
    /// [`EmptyIncrement`]: Kind::EmptyIncrement
    pub(crate) fn empty_increment(position: Position) -> Self {
        Self {
            kind: Kind::EmptyIncrement,
            position,
        }
    }

    /// Creates an [`EmptyRepeat`] error.
    ///
    /// [`EmptyRepeat`]: Kind::EmptyRepeat
    pub(crate) fn empty_repeat(position: Position) -> Self {
        Self {
            kind: Kind::EmptyRepeat,
            position,
        }
    }

    /// Creates a [`MismatchedRepeat`] error.
    ///
    /// [`MismatchedRepeat`]: Kind::MismatchedRepeat
    pub(crate) fn mismatched_repeat(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedRepeat,
            position,
        }
    }

    /// Creates a [`MultipleIncrement`] error.
    ///
    /// [`MultipleIncrement`]: Kind::MultipleIncrement
    pub(crate) fn multiple_increment(position: Position) -> Self {
        Self {
            kind: Kind::MultipleIncrement,
            position,
        }
    }

    /// Creates a [`MultipleRepeat`] error.
    ///
    /// [`MultipleRepeat`]: Kind::MultipleRepeat
    pub(crate) fn multiple_repeat(position: Position) -> Self {
        Self {
            kind: Kind::MultipleRepeat,
            position,
        }
    }

    /// Creates an [`IncrementAfterRepeat`] error.
    ///
    /// [`IncrementAfterRepeat`]: Kind::IncrementAfterRepeat
    pub(crate) fn increment_after_repeat(position: Position) -> Self {
        Self {
            kind: Kind::IncrementAfterRepeat,
            position,
        }
    }

    /// Creates a [`MultipleIdentifierIncrement`] error.
    ///
    /// [`MultipleIdentifierIncrement`]: Kind::MultipleIdentifierIncrement
    pub(crate) fn multiple_identifier_increment(position: Position) -> Self {
        Self {
            kind: Kind::MultipleIdentifierIncrement,
            position,
        }
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Returns the position in the source that caused the error.
    pub fn position(&self) -> Position {
        self.position
    }
}
