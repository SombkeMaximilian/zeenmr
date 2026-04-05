//! JCAMP-DX grouped block tabulation error types.

use crate::Position;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out
/// `Result<T, zeenmr_parse::jcampdx::tabulation::error::Error>` directly.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during tabulation of a block.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Position in the source.
    position: Position,
}

/// The kind of `Error` that can occur during tabulation.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    /// A literal does not match any token.
    #[default]
    InvalidLiteral,
    /// Value is too large for primitive types.
    ///
    /// This occurs if a value can't fit in an [`i64`] and usually means that
    /// the file is corrupted.
    Overflow,
    /// A group with an unexpected size was encountered.
    MismatchedGroupSize,
    /// A mismatched group delimiter was encountered.
    ///
    /// This can be:
    /// - A semicolon that terminates a group within parentheses
    /// - An unmatched closing parenthesis
    /// - A new opening parenthesis before the previous one was closed, but the
    ///   expected group size was not violated.
    MismatchedGroupDelimiter,
    /// An unmatched opening or closing angle bracket was encountered.
    ///
    /// This specifically occurs if an angle bracket isn't closed before
    /// reaching the end of the input, the next opening angle bracket, or the
    /// next header key, or if a closing angle bracket occurs without an opening
    /// one.
    UnmatchedStringDelimiter,
    /// Two or more consecutive values are not separated or enclosed by angle
    /// brackets.
    NonSeparatedValues,
    /// A group extends past a new line without being enclosed by parentheses.
    CrossLineGroup,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::InvalidLiteral => "invalid literal",
            Kind::Overflow => "overflow",
            Kind::MismatchedGroupSize => "mismatched group size",
            Kind::MismatchedGroupDelimiter => "mismatched group delimiter",
            Kind::UnmatchedStringDelimiter => "unmatched string delimiter",
            Kind::NonSeparatedValues => "non-separated values",
            Kind::CrossLineGroup => "cross-line group",
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

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(position: Position) -> Self {
        Self {
            kind: Kind::Overflow,
            position,
        }
    }

    /// Creates a [`MismatchedGroupSize`] error.
    ///
    /// [`MismatchedGroupSize`]: Kind::MismatchedGroupSize
    pub(crate) fn mismatched_group_size(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedGroupSize,
            position,
        }
    }

    /// Creates a [`MismatchedGroupDelimiter`] error.
    ///
    /// [`MismatchedGroupDelimiter`]: Kind::MismatchedGroupDelimiter
    pub(crate) fn mismatched_group_delimiter(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedGroupDelimiter,
            position,
        }
    }

    /// Creates a [`UnmatchedStringDelimiter`] error.
    ///
    /// [`UnmatchedStringDelimiter`]: Kind::UnmatchedStringDelimiter
    pub(crate) fn unmatched_string_delimiter(position: Position) -> Self {
        Self {
            kind: Kind::UnmatchedStringDelimiter,
            position,
        }
    }

    /// Creates a [`NonSeparatedValues`] error.
    ///
    /// [`NonSeparatedValues`]: Kind::NonSeparatedValues
    pub(crate) fn non_separated_values(position: Position) -> Self {
        Self {
            kind: Kind::NonSeparatedValues,
            position,
        }
    }

    /// Creates a [`CrossLineGroup`] error.
    ///
    /// [`CrossLineGroup`]: Kind::CrossLineGroup
    pub(crate) fn cross_line_group(position: Position) -> Self {
        Self {
            kind: Kind::CrossLineGroup,
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
