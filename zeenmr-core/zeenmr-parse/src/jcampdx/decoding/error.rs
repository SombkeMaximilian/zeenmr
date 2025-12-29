//! JCAMP-DX decoding error types.

use crate::Position;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out `Result<T, zeenmr_parse::Error>`
/// directly.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during decoding of a block.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Position in the source.
    position: Position,
    /// Index of the value in the block.
    index: Option<usize>,
}

/// The kind of `Error` that can occur during decoding.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    /// A literal does not match any token.
    #[default]
    InvalidLiteral,
    /// Value is too large for primitive types.
    ///
    /// This occurs if a value can't fit in an [`i64`] and usually means that
    /// the file is corrupted.
    Overflow,
    /// An integrity check failed.
    ///
    /// In `XYDATA` mode, if the final value in the previous line is `DIF`
    /// encoded, the first value of the next line will be a repeat of this
    /// last value. If they do not match, the integrity check fails.
    IntegrityCheck,
    /// An invalid value was found.
    ///
    /// Missing or corrupted values are replaced by `?`. These are decoded as
    /// [`i64::MIN`].
    InvalidValue,
    /// A `DIF` or `DUP` was encountered directly after a checkpoint.
    ///
    /// `DIF` and `DUP` values apply to the directly preceding value in the same
    /// line. If they're the first value in a line, they can't be resolved. This
    /// applies to the checkpoint value itself and the first actual data value.
    DifDupAfterCheckPoint,
    /// Input contains an unsupported format.
    ///
    /// This currently includes using floating point values in `XYDATA` blocks,
    /// which practically never gets used over `ASDF` encoding.
    UnsupportedFormat,
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::InvalidLiteral => "invalid literal encountered",
            Kind::Overflow => "overflow",
            Kind::IntegrityCheck => "integrity check failed",
            Kind::InvalidValue => "invalid value",
            Kind::DifDupAfterCheckPoint => "DIF or DUP after checkpoint",
            Kind::UnsupportedFormat => "unsupported format",
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
            index: None,
        }
    }

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(position: Position) -> Self {
        Self {
            kind: Kind::Overflow,
            position,
            index: None,
        }
    }

    /// Creates an [`IntegrityCheck`] error.
    ///
    /// [`IntegrityCheck`]: Kind::IntegrityCheck
    pub(crate) fn integrity_check(position: Position, index: usize) -> Self {
        Self {
            kind: Kind::IntegrityCheck,
            position,
            index: Some(index),
        }
    }

    /// Creates an [`InvalidValue`] error.
    ///
    /// [`InvalidValue`]: Kind::InvalidValue
    pub(crate) fn invalid_value(position: Position, index: usize) -> Self {
        Self {
            kind: Kind::InvalidValue,
            position,
            index: Some(index),
        }
    }

    /// Creates a [`DifDupAfterCheckPoint`] error.
    ///
    /// [`DifDupAfterCheckPoint`]: Kind::DifDupAfterCheckPoint
    pub(crate) fn dif_dup_after_check_point(position: Position) -> Self {
        Self {
            kind: Kind::DifDupAfterCheckPoint,
            position,
            index: None,
        }
    }

    /// Creates an [`UnsupportedFormat`] error.
    ///
    /// [`UnsupportedFormat`]: Kind::UnsupportedFormat
    pub(crate) fn unsupported_format(position: Position) -> Self {
        Self {
            kind: Kind::UnsupportedFormat,
            position,
            index: None,
        }
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }
}
