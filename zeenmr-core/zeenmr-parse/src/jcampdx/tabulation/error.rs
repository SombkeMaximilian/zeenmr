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
    MismatchedGroupSize,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::InvalidLiteral => "invalid literal",
            Kind::MismatchedGroupSize => "mismatched group size",
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

    /// Creates a [`MismatchedGroupSize`] error.
    ///
    /// [`MismatchedGroupSize`]: Kind::MismatchedGroupSize
    pub(crate) fn mismatched_group_size(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedGroupSize,
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
