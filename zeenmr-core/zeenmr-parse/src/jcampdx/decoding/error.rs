//! JCAMP-DX decoding error types.

use crate::error::{ByteRange, ParseError, RangeLabel};
use crate::jcampdx::decoding::EncodedToken;
use logos::Logos;
use std::borrow::Cow;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out the full type name directly, and is
/// broadly used across the library.
pub(crate) type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during decoding of a block.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Byte range in the source.
    range: ByteRange,
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
    fn message(&self) -> Cow<'static, str> {
        match self.kind {
            Kind::InvalidLiteral => "invalid literal".into(),
            Kind::Overflow => "value magnitude too large to fit into i64".into(),
            Kind::DifDupOverflow => "DIF/DUP value magnitude too large to fit into i64".into(),
            Kind::IntegrityCheck => "integrity check failed".into(),
            Kind::InvalidValue => "invalid or missing value".into(),
            Kind::AsdfCheckpoint => "checkpoint value is ASDF-encoded".into(),
            Kind::DifDupAfterCheckPoint => "first value is DIF/DUP-encoded".into(),
            Kind::AsdfWithFloat => "DIF/DUP encoding mixed with floating point values".into(),
            Kind::CheckPointStepMismatch => "checkpoint value spacing is inconsistent".into(),
        }
    }

    fn labels(&self, source: &str) -> Vec<RangeLabel> {
        match self.kind {
            Kind::InvalidLiteral => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match any token".into()),
            }],
            Kind::Overflow | Kind::DifDupOverflow => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not fit into 64-bit signed integer".into()),
            }],
            Kind::IntegrityCheck => {
                let previous_end = source[..self.range.end].rfind("\n");
                let previous_start = previous_end.map(|current| {
                    source[..current]
                        .rfind("\n")
                        .map(|start| start + 1)
                        .unwrap_or(0)
                });

                if let (Some(start), Some(end)) = (previous_start, previous_end) {
                    let last_range = EncodedToken::lexer(&source[start..end])
                        .spanned()
                        .skip(1)
                        .filter_map(|(token, span)| token.ok().map(|token| (token, span)))
                        .fold(start..end, |acc, (token, span)| match token {
                            EncodedToken::CheckPoint => (start + acc.start)..(start + acc.end),
                            _ => span,
                        })
                        .into();

                    vec![
                        RangeLabel {
                            range: self.range,
                            is_cause: true,
                            label: Some("integrity check failed here".into()),
                        },
                        RangeLabel {
                            range: last_range,
                            is_cause: false,
                            label: Some(
                                "DIF-encoded value that should be duplicated on the next line"
                                    .into(),
                            ),
                        },
                    ]
                } else {
                    vec![RangeLabel {
                        range: self.range,
                        is_cause: true,
                        label: Some("does not match last value on previous line".into()),
                    }]
                }
            }
            Kind::InvalidValue => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: None,
            }],
            Kind::AsdfCheckpoint => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("should not be ASDF-encoded".into()),
            }],
            Kind::DifDupAfterCheckPoint => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("should not be DIF/DUP-encoded".into()),
            }],
            Kind::AsdfWithFloat => {
                let cause_is_float = source[self.range.as_range()].contains(".");
                let cause_label = RangeLabel {
                    range: self.range,
                    is_cause: true,
                    label: Some(if cause_is_float {
                        "should not be float".into()
                    } else {
                        "should not be ASDF-encoded".into()
                    }),
                };
                let other_text = if cause_is_float {
                    Some("ASDF-encoded value here".into())
                } else {
                    Some("float value here".into())
                };
                let mut start_candidate = source[..self.range.end].rfind("\n");
                let mut line_end = self.range.end;
                let mut other_label = None;
                while let Some(line_start) = start_candidate {
                    if let Some(other_range) = EncodedToken::lexer(&source[line_start..line_end])
                        .spanned()
                        .skip(1)
                        .filter_map(|(token, span)| token.ok().map(|token| (token, span)))
                        .find(|(token, span)| match token {
                            EncodedToken::Compressed
                            | EncodedToken::Difference
                            | EncodedToken::Duplicate
                                if cause_is_float =>
                            {
                                true
                            }
                            EncodedToken::Numeric
                                if !cause_is_float && source[span.clone()].contains(".") =>
                            {
                                true
                            }
                            _ => false,
                        })
                        .map(|(_, span)| {
                            ByteRange::from((line_start + span.start)..(line_start + span.end))
                        })
                    {
                        other_label = Some(RangeLabel {
                            range: other_range,
                            is_cause: false,
                            label: other_text,
                        });
                        break;
                    } else {
                        line_end = line_start;
                        start_candidate = source[..line_end].rfind("\n");
                    }
                }

                if let Some(other_label) = other_label {
                    vec![cause_label, other_label]
                } else {
                    vec![cause_label]
                }
            }
            Kind::CheckPointStepMismatch => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match prior step size".into()),
            }],
        }
    }

    fn notes(&self, _: &str) -> Vec<Cow<'static, str>> {
        match self.kind {
            Kind::InvalidLiteral => {
                vec!["values must be a standard numeric format or ASDF-encoded".into()]
            }
            Kind::Overflow => {
                vec!["a value of this magnitude is almost certainly not expected".into()]
            }
            Kind::DifDupOverflow => vec![
                "a value of this magnitude is almost certainly not expected".into(),
                "overflow in DIF/DUP-encoding corrupts other data".into(),
            ],
            Kind::IntegrityCheck => vec![
                "if the last value on a line is DIF-encoded, it is repeated on the next".into(),
                "a DUP preceded by a DIF also triggers an integrity check".into(),
            ],
            Kind::InvalidValue => {
                vec!["missing or corrupted values are marked as `?` in the source".into()]
            }
            Kind::AsdfCheckpoint => {
                vec!["ASDF encoding is not permitted as a checkpoint value".into()]
            }
            Kind::DifDupAfterCheckPoint => {
                vec!["DIF/DUP encoding is not permitted as the first data value".into()]
            }
            Kind::AsdfWithFloat => vec![
                "mixing ASDF encoding with floating point values is not permitted".into(),
                "differences of floating point values accumulate errors".into(),
            ],
            Kind::CheckPointStepMismatch => vec![
                "x is the value and n the data point count at checkpoints a and b".into(),
                "expected step = (x_b - x_a) / (n_b - n_a - 1)".into(),
            ],
        }
    }

    fn fix_hints(&self, _: &str) -> Vec<Cow<'static, str>> {
        let adjacent = "adjacent values may have been combined".into();
        let reexport = "re-exporting the file from the software might fix it".into();
        let skipped = "a line may have been skipped or duplicated".into();
        let standard = "the software used for exporting might not comply with the standard".into();

        match self.kind {
            Kind::InvalidLiteral
            | Kind::AsdfCheckpoint
            | Kind::DifDupAfterCheckPoint
            | Kind::AsdfWithFloat => vec![standard, reexport],
            Kind::Overflow | Kind::DifDupOverflow => vec![adjacent, reexport],
            Kind::IntegrityCheck => vec![skipped, reexport],
            Kind::CheckPointStepMismatch => vec![skipped, adjacent, reexport],
            _ => Vec::new(),
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
    pub(crate) fn invalid_literal(range: ByteRange) -> Self {
        Self {
            kind: Kind::InvalidLiteral,
            range,
            index: None,
        }
    }

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(range: ByteRange) -> Self {
        Self {
            kind: Kind::Overflow,
            range,
            index: None,
        }
    }

    /// Creates a [`DifDupOverflow`] error.
    ///
    /// [`DifDupOverflow`]: Kind::DifDupOverflow
    pub(crate) fn dif_dup_overflow(range: ByteRange) -> Self {
        Self {
            kind: Kind::DifDupOverflow,
            range,
            index: None,
        }
    }

    /// Creates an [`IntegrityCheck`] error.
    ///
    /// [`IntegrityCheck`]: Kind::IntegrityCheck
    pub(crate) fn integrity_check(range: ByteRange, index: usize) -> Self {
        Self {
            kind: Kind::IntegrityCheck,
            range,
            index: Some(index),
        }
    }

    /// Creates an [`InvalidValue`] error.
    ///
    /// [`InvalidValue`]: Kind::InvalidValue
    pub(crate) fn invalid_value(range: ByteRange) -> Self {
        Self {
            kind: Kind::InvalidValue,
            range,
            index: None,
        }
    }

    /// Creates a [`AsdfCheckpoint`] error.
    ///
    /// [`AsdfCheckpoint`]: Kind::AsdfCheckpoint
    pub(crate) fn asdf_checkpoint(range: ByteRange) -> Self {
        Self {
            kind: Kind::AsdfCheckpoint,
            range,
            index: None,
        }
    }

    /// Creates a [`DifDupAfterCheckPoint`] error.
    ///
    /// [`DifDupAfterCheckPoint`]: Kind::DifDupAfterCheckPoint
    pub(crate) fn dif_dup_after_checkpoint(range: ByteRange) -> Self {
        Self {
            kind: Kind::DifDupAfterCheckPoint,
            range,
            index: None,
        }
    }

    /// Creates a [`AsdfWithFloat`] error.
    ///
    /// [`AsdfWithFloat`]: Kind::AsdfWithFloat
    pub(crate) fn asdf_with_float(range: ByteRange) -> Self {
        Self {
            kind: Kind::AsdfWithFloat,
            range,
            index: None,
        }
    }

    /// Creates a [`CheckPointStepMismatch`] error.
    ///
    /// [`CheckPointStepMismatch`]: Kind::CheckPointStepMismatch
    pub(crate) fn checkpoint_step_mismatch(range: ByteRange) -> Self {
        Self {
            kind: Kind::CheckPointStepMismatch,
            range,
            index: None,
        }
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Returns the range in the source that caused the error.
    pub fn byte_range(&self) -> ByteRange {
        self.range
    }

    /// Returns the index of the value in the decoded sequence which caused the
    /// error.
    ///
    /// If the error occurred in the checkpoint value, this returns [`None`].
    pub fn index(&self) -> Option<usize> {
        self.index
    }
}
