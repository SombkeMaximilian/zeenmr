//! JCAMP-DX block format parsing error types.

use crate::error::{ByteRange, ParseError, RangeLabel};
use std::borrow::Cow;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out
/// `Result<T, zeenmr_parse::jcampdx::block_format::error::Error>` directly.
pub(crate) type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred while parsing the block format.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Byte range in the source.
    range: ByteRange,
}

/// The kind of `Error` that can occur while parsing the block format.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    /// A literal does not match any token.
    #[default]
    InvalidLiteral,
    /// The input ended.
    ///
    /// Since a block format describes how to read data following it, the input
    /// ending means the file is certainly malformed or corrupted.
    EndOfInput,
    /// The format specifier string is not present.
    EmptyFormat,
    /// No identifier was encountered in the increment pattern.
    ///
    /// Example: `++(Y..Y)`
    EmptyIncrement,
    /// No identifier was encountered in the repeat pattern.
    ///
    /// Example: `X++(..)`, `X++(..Y)`
    EmptyRepeat,
    /// No repeat pattern was encountered after an increment pattern.
    ///
    /// Example: `X++`
    MissingRepeat,
    /// Repeating identifiers do not match.
    ///
    /// Example: `X++(Y..R)`
    MismatchedRepeat,
    /// Multiple increment patterns were encountered.
    ///
    /// Example: `X++X++(Y..Y)`
    MultipleIncrement,
    /// Multiple repeat patterns were encountered.
    ///
    /// Example: `X++(R..R..R)`, `X++(R..R)(I..I)`
    MultipleRepeat,
    /// An increment pattern was encountered after a repeat pattern.
    ///
    /// Example: `(R..R)++`
    IncrementAfterRepeat,
    /// Multiple identifiers were encountered in an increment pattern.
    ///
    /// Example: `XY++(Z..Z)`
    MultipleIdentifierIncrement,
}

impl ParseError for Error {
    fn message(&self) -> Cow<'static, str> {
        match self.kind {
            Kind::InvalidLiteral => "invalid literal".into(),
            Kind::EndOfInput => "unexpected end of input".into(),
            Kind::EmptyFormat => "empty format specifier".into(),
            Kind::EmptyIncrement => "increment pattern has no identifier".into(),
            Kind::EmptyRepeat => "repeat pattern has no identifier".into(),
            Kind::MissingRepeat => "missing repeat pattern after increment pattern".into(),
            Kind::MismatchedRepeat => "repeat identifiers do not match prefix".into(),
            Kind::MultipleIncrement => "multiple increment patterns in format specifier".into(),
            Kind::MultipleRepeat => "multiple repeat patterns in format specifier".into(),
            Kind::IncrementAfterRepeat => "increment pattern follows repeat pattern".into(),
            Kind::MultipleIdentifierIncrement => {
                "increment pattern contains multiple identifiers".into()
            }
        }
    }

    fn labels(&self, _: &str) -> Vec<RangeLabel> {
        match self.kind {
            Kind::InvalidLiteral => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match any token".into()),
            }],
            Kind::EndOfInput => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("input source ends here".into()),
            }],
            Kind::EmptyFormat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected format specifier here".into()),
            }],
            Kind::EmptyIncrement => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected an identifier before this".into()),
            }],
            Kind::EmptyRepeat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected identifier(s) before this".into()),
            }],
            Kind::MissingRepeat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected repeat pattern following this increment pattern".into()),
            }],
            Kind::MismatchedRepeat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match prefix".into()),
            }],
            Kind::MultipleIncrement => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("second increment pattern here".into()),
            }],
            Kind::MultipleRepeat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("second repeat pattern here".into()),
            }],
            Kind::IncrementAfterRepeat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("must appear before preceding repeat group".into()),
            }],
            Kind::MultipleIdentifierIncrement => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("only one identifier may precede this".into()),
            }],
        }
    }

    fn notes(&self, _: &str) -> Vec<Cow<'static, str>> {
        match self.kind {
            Kind::InvalidLiteral => {
                vec![
                    "identifiers must be single letters optionally followed by digits".into(),
                    "other, valid tokens are `++` and `..`".into(),
                ]
            }
            Kind::EndOfInput => {
                vec!["the format specifier must be followed by data".into()]
            }
            Kind::EmptyFormat => {
                vec!["a format specifier describes the layout of the following data block".into()]
            }
            Kind::EmptyIncrement => {
                vec!["the increment pattern requires an identifier before `++`".into()]
            }
            Kind::EmptyRepeat => {
                vec!["the repeat pattern requires an identifier(s) on both sides of `..`".into()]
            }
            Kind::MissingRepeat => {
                vec!["a repeat pattern must follow an increment pattern to fill the lines".into()]
            }
            Kind::MismatchedRepeat => {
                vec!["the identifiers on both sides of `..` must be identical".into()]
            }
            Kind::MultipleIncrement | Kind::MultipleIdentifierIncrement => {
                vec!["only one identifier may be marked as incrementing with `++`".into()]
            }
            Kind::MultipleRepeat => {
                vec!["only one repeat group `(..)` is permitted per format specifier".into()]
            }
            Kind::IncrementAfterRepeat => {
                vec!["the incrementing pattern must appear before the repeat pattern".into()]
            }
        }
    }

    fn fix_hints(&self, _: &str) -> Vec<Cow<'static, str>> {
        let standard = "the software used for exporting might not comply with the standard".into();
        let reexport = "re-exporting the file from the software might fix it".into();

        match self.kind {
            Kind::InvalidLiteral
            | Kind::EmptyFormat
            | Kind::EmptyIncrement
            | Kind::EmptyRepeat
            | Kind::MissingRepeat
            | Kind::MultipleIncrement
            | Kind::MultipleRepeat
            | Kind::IncrementAfterRepeat
            | Kind::MultipleIdentifierIncrement => vec![standard, reexport],
            Kind::EndOfInput => {
                vec!["the file may be truncated or corrupted".into(), reexport]
            }
            Kind::MismatchedRepeat => vec![
                "the identifier(s) on the right side of `..` may contain typos".into(),
                standard,
                reexport,
            ],
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
    /// Creates an [`InvalidLiteral`] error.
    ///
    /// [`InvalidLiteral`]: Kind::InvalidLiteral
    pub(crate) fn invalid_literal(range: ByteRange) -> Self {
        Self {
            kind: Kind::InvalidLiteral,
            range,
        }
    }

    /// Creates an [`EndOfInput`] error.
    ///
    /// [`EndOfInput`]: Kind::EndOfInput
    pub(crate) fn end_of_input(range: ByteRange) -> Self {
        Self {
            kind: Kind::EndOfInput,
            range,
        }
    }

    /// Creates an [`EmptyFormat`] error.
    ///
    /// [`EmptyFormat`]: Kind::EmptyFormat
    pub(crate) fn empty_format(range: ByteRange) -> Self {
        Self {
            kind: Kind::EmptyFormat,
            range,
        }
    }

    /// Creates an [`EmptyIncrement`] error.
    ///
    /// [`EmptyIncrement`]: Kind::EmptyIncrement
    pub(crate) fn empty_increment(range: ByteRange) -> Self {
        Self {
            kind: Kind::EmptyIncrement,
            range,
        }
    }

    /// Creates an [`EmptyRepeat`] error.
    ///
    /// [`EmptyRepeat`]: Kind::EmptyRepeat
    pub(crate) fn empty_repeat(range: ByteRange) -> Self {
        Self {
            kind: Kind::EmptyRepeat,
            range,
        }
    }

    /// Creates an [`MissingRepeat`] error.
    ///
    /// [`MissingRepeat`]: Kind::MissingRepeat
    pub(crate) fn missing_repeat(range: ByteRange) -> Self {
        Self {
            kind: Kind::MissingRepeat,
            range,
        }
    }

    /// Creates a [`MismatchedRepeat`] error.
    ///
    /// [`MismatchedRepeat`]: Kind::MismatchedRepeat
    pub(crate) fn mismatched_repeat(range: ByteRange) -> Self {
        Self {
            kind: Kind::MismatchedRepeat,
            range,
        }
    }

    /// Creates a [`MultipleIncrement`] error.
    ///
    /// [`MultipleIncrement`]: Kind::MultipleIncrement
    pub(crate) fn multiple_increment(range: ByteRange) -> Self {
        Self {
            kind: Kind::MultipleIncrement,
            range,
        }
    }

    /// Creates a [`MultipleRepeat`] error.
    ///
    /// [`MultipleRepeat`]: Kind::MultipleRepeat
    pub(crate) fn multiple_repeat(range: ByteRange) -> Self {
        Self {
            kind: Kind::MultipleRepeat,
            range,
        }
    }

    /// Creates an [`IncrementAfterRepeat`] error.
    ///
    /// [`IncrementAfterRepeat`]: Kind::IncrementAfterRepeat
    pub(crate) fn increment_after_repeat(range: ByteRange) -> Self {
        Self {
            kind: Kind::IncrementAfterRepeat,
            range,
        }
    }

    /// Creates a [`MultipleIdentifierIncrement`] error.
    ///
    /// [`MultipleIdentifierIncrement`]: Kind::MultipleIdentifierIncrement
    pub(crate) fn multiple_identifier_increment(range: ByteRange) -> Self {
        Self {
            kind: Kind::MultipleIdentifierIncrement,
            range,
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
}
