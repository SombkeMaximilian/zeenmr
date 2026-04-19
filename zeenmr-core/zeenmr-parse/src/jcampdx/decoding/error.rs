//! JCAMP-DX decoding error types.

use crate::error::{Annotations, ParseError, Position};

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out
/// `Result<T, zeenmr_parse::jcampdx::decoding::error::Error>` directly.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during decoding of a block.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
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
    /// A `DIF` or `DUP` value caused an overflow error.
    ///
    /// Since these affect not only their own data but also surrounding values,
    /// this error is guaranteed to corrupt the rest of the data, making the
    /// input impossible to decode.
    DifDupOverflow,
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
    /// A checkpoint value was `ASDF` encoded.
    ///
    /// Checkpoints are meant to be plain numeric values as per the standard.
    /// While `SQZ` values could be resolved in this place, they strongly
    /// indicate data corruption.
    AsdfCheckpoint,
    /// A `DIF` or `DUP` was encountered directly after a checkpoint.
    ///
    /// `DIF` and `DUP` values apply to the directly preceding value in the same
    /// line. If they're the first value in a line, they can't be resolved. This
    /// applies to the checkpoint value itself and the first actual data value.
    DifDupAfterCheckPoint,
    /// A `DIF` or `DUP` was encountered after or before floating point values.
    ///
    /// `DIF` encodes the next value as a difference to the previous value. This
    /// accumulates errors with floating point values and was therefore not
    /// intended for the standard. In principle `DUP` could be resolved with
    /// floating point values. However, a `DUP` can also apply to a prior `DIF`
    /// and these encodings should not be mixed with floating point values as
    /// per the standard.
    AsdfWithFloat,
    /// The step size between check points does not match.
    ///
    /// The expected step size between checkpoints is
    /// ```text
    /// x_b - x_a / (n_b - n_a),
    /// ```
    /// where `x_b` and `x_a` are the checkpoint values and `n_b` and `n_a` are
    /// the number of decoded data points at checkpoints `b` and `a`. If the
    /// decoded points in that segment are not spaced by this constant step, the
    /// spacing or count is inconsistent.
    CheckPointStepMismatch,
}

impl ParseError for Error {
    fn primary(&self) -> Position {
        self.position
    }

    fn message(&self) -> String {
        match self.kind {
            Kind::InvalidLiteral => "invalid literal",
            Kind::Overflow => "value magnitude too large to decode",
            Kind::DifDupOverflow => "DIF/DUP value magnitude too large to decode",
            Kind::IntegrityCheck => "integrity check failed",
            Kind::InvalidValue => "invalid value (parsed as `-2^63`)",
            Kind::AsdfCheckpoint => "checkpoint is ASDF-encoded",
            Kind::DifDupAfterCheckPoint => "first value is DIF/DUP",
            Kind::AsdfWithFloat => "DIF/DUP encoding mixed with floating point values",
            Kind::CheckPointStepMismatch => "checkpoint value spacing is inconsistent",
        }
        .into()
    }

    fn highlight_text(&self) -> String {
        match self.kind {
            Kind::InvalidLiteral => "does not match any token",
            Kind::Overflow | Kind::DifDupOverflow => "does not fit into 64-bit signed integer",
            Kind::IntegrityCheck => "not equal to the last value on previous line",
            Kind::InvalidValue => "",
            Kind::AsdfCheckpoint => "should not be ASDF-encoded",
            Kind::DifDupAfterCheckPoint => "should not be DIF/DUP-encoded",
            Kind::AsdfWithFloat => "should not be DIF/DUP-encoded or float",
            Kind::CheckPointStepMismatch => "does not match prior step size",
        }
        .into()
    }

    fn note(&self) -> Annotations {
        match self.kind {
            Kind::InvalidLiteral => "\
             values must be a standard numeric format or ASDF-encoded \
            "
            .into(),
            Kind::DifDupOverflow => "\
             DIF/DUP values affect all subsequent values on the line \
            "
            .into(),
            Kind::IntegrityCheck => "\
             in XYDATA mode, the first value of a line repeats the last \
             value of the previous line if DIF-encoded \
            "
            .into(),
            Kind::InvalidValue => "\
             missing or corrupted values are marked `?` in the source \
            "
            .into(),
            Kind::AsdfCheckpoint => "\
             ASDF encoding is not permitted as a checkpoint value\
            "
            .into(),
            Kind::DifDupAfterCheckPoint => "\
             DIF/DUP encoding is not permitted as the first data value\
            "
            .into(),
            Kind::AsdfWithFloat => "\
             mixing ASDF encoding with floating point is not permitted by \
             the standard \
            "
            .into(),
            Kind::CheckPointStepMismatch => "\
             expected step = (x_b - x_a) / (n_b - n_a) where x is the value \
             and n is the decoded point count at checkpoints a and b \
            "
            .into(),
            _ => Annotations::None,
        }
    }

    fn fix_hint(&self) -> Annotations {
        match self.kind {
            Kind::Overflow | Kind::DifDupOverflow => Annotations::Multiple(vec![
                "adjacent values may have been combined".into(),
                "re-export the file from the software".into(),
            ]),
            Kind::IntegrityCheck => Annotations::Multiple(vec![
                "a line may have been skipped or duplicated".into(),
                "re-export the file from the software".into(),
            ]),
            Kind::AsdfCheckpoint | Kind::DifDupAfterCheckPoint | Kind::AsdfWithFloat => "\
             the software exporting this dataset might not comply with \
             the JCAMP-DX standard \
            "
            .into(),
            Kind::CheckPointStepMismatch => Annotations::Multiple(vec![
                "a line may have been skipped or duplicated".into(),
                "adjacent values may have been combined".into(),
                "re-export the file from the software".into(),
            ]),
            _ => Annotations::None,
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl Error {
    /// Adds the index of the value in the decoded values that caused the error.
    pub(crate) fn with_index(mut self, index: usize) -> Self {
        self.index = Some(index);

        self
    }

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

    /// Creates a [`DifDupOverflow`] error.
    ///
    /// [`DifDupOverflow`]: Kind::DifDupOverflow
    pub(crate) fn dif_dup_overflow(position: Position) -> Self {
        Self {
            kind: Kind::DifDupOverflow,
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
    pub(crate) fn invalid_value(position: Position) -> Self {
        Self {
            kind: Kind::InvalidValue,
            position,
            index: None,
        }
    }

    /// Creates a [`AsdfCheckpoint`] error.
    ///
    /// [`AsdfCheckpoint`]: Kind::AsdfCheckpoint
    pub(crate) fn asdf_checkpoint(position: Position) -> Self {
        Self {
            kind: Kind::AsdfCheckpoint,
            position,
            index: None,
        }
    }

    /// Creates a [`DifDupAfterCheckPoint`] error.
    ///
    /// [`DifDupAfterCheckPoint`]: Kind::DifDupAfterCheckPoint
    pub(crate) fn dif_dup_after_checkpoint(position: Position) -> Self {
        Self {
            kind: Kind::DifDupAfterCheckPoint,
            position,
            index: None,
        }
    }

    /// Creates a [`AsdfWithFloat`] error.
    ///
    /// [`AsdfWithFloat`]: Kind::AsdfWithFloat
    pub(crate) fn asdf_with_float(position: Position) -> Self {
        Self {
            kind: Kind::AsdfWithFloat,
            position,
            index: None,
        }
    }

    /// Creates a [`CheckPointStepMismatch`] error.
    ///
    /// [`CheckPointStepMismatch`]: Kind::CheckPointStepMismatch
    pub(crate) fn checkpoint_step_mismatch(position: Position) -> Self {
        Self {
            kind: Kind::CheckPointStepMismatch,
            position,
            index: None,
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

    /// Returns the index of the value in the decoded sequence which caused the
    /// error.
    ///
    /// If the error occurred in the checkpoint value, this returns [`None`].
    pub fn index(&self) -> Option<usize> {
        self.index
    }
}
