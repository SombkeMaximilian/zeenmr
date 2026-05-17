//! JCAMP-DX parsing error types.

use crate::error::{ByteRange, ParseError, RangeLabel};
use std::borrow::Cow;

pub use crate::jcampdx::block_format::error as format_error;
pub use crate::jcampdx::decoding::error as decode_error;
pub use crate::jcampdx::tabulation::error as table_error;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out `Result<T, zeenmr_parse::Error>`
/// directly.
pub type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred while parsing a file.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Byte range in the source.
    range: ByteRange,
}

/// The kind of `Error` that can occur while parsing a file.
///
/// Marked as non-exhaustive to allow for new variants to be added in the future
/// without breaking compatibility.
#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    /// A literal does not match any token.
    #[default]
    InvalidLiteral,
    /// The input ended unexpectedly early.
    EndOfInput,
    /// An entry point could not be found for the input.
    ///
    /// In the JCAMP-DX standard, a dataset is started with `##TITLE=` and ended
    /// with `##END=`.
    NoEntryPoint,
    /// A key prefix was encountered, but the label was empty.
    ///
    /// Keys start with `##`, `##.` or `##$` and are separated from their value
    /// by `=`. If there is no token between the prefix and the key-value
    /// separator, the key is empty and cannot be parsed.
    EmptyKey,
    /// Another key token was encountered before fully parsing the current one.
    ///
    /// If another key token is encountered between an earlier key token and its
    /// key-value separator, the file is likely malformed or corrupted.
    MultipleKeyTokens,
    /// A closing delimiter which does not match the most recent opening
    /// delimiter was encountered.
    MismatchedDelimiter,
    /// An opening delimiter's matching closing delimiter was not encountered
    /// before the next key token.
    UnclosedDelimiter,
    /// Page tokens are only valid inside an `NTUPLES` block.
    UnexpectedPage,
    /// Further nesting was encountered within an `NTUPLES` block.
    NestedTuples,
    /// A block format which does not represent the correct kind of data block
    /// was parsed.
    ///
    /// For example, the `XYDATA` format requires a format specifier of the form
    /// `X++(Y..Y)`, while the `XYPOINTS` format requires a format specifier of
    /// the form `(XY..XY)` or `(XY)`.
    MismatchedBlockFormat,
    /// Value is too large for primitive types.
    ///
    /// This occurs if a value can't fit in an [`i64`] and usually means that
    /// the file is corrupted.
    Overflow,
    /// An error was encountered while parsing the format specifier of a data
    /// block.
    Format(format_error::Error),
    /// An error was encountered while decoding an encoded block.
    Decode(decode_error::Error),
    /// An error was encountered while parsing a data table.
    Table(table_error::Error),
}

impl From<format_error::Error> for Error {
    fn from(value: format_error::Error) -> Self {
        Self {
            kind: Kind::Format(value),
            range: value.byte_range(),
        }
    }
}

impl From<decode_error::Error> for Error {
    fn from(value: decode_error::Error) -> Self {
        Self {
            kind: Kind::Decode(value),
            range: value.byte_range(),
        }
    }
}

impl From<table_error::Error> for Error {
    fn from(value: table_error::Error) -> Self {
        Self {
            kind: Kind::Table(value),
            range: value.byte_range(),
        }
    }
}

impl ParseError for Error {
    fn message(&self) -> Cow<'static, str> {
        match self.kind {
            Kind::InvalidLiteral => "invalid literal".into(),
            Kind::EndOfInput => "unexpected end of input".into(),
            Kind::NoEntryPoint => "no entry point".into(),
            Kind::EmptyKey => "empty key".into(),
            Kind::MultipleKeyTokens => "malformed key".into(),
            Kind::MismatchedDelimiter => "mismatched delimiter".into(),
            Kind::UnclosedDelimiter => "unclosed delimiter".into(),
            Kind::UnexpectedPage => "page token outside `NTUPLES` block".into(),
            Kind::NestedTuples => "nested `NTUPLES` block".into(),
            Kind::MismatchedBlockFormat => "block format does not match block kind".into(),
            Kind::Overflow => "value magnitude too large to fit into i64".into(),
            Kind::Format(inner) => inner.message(),
            Kind::Decode(inner) => inner.message(),
            Kind::Table(inner) => inner.message(),
        }
    }

    fn labels(&self, source: &str) -> Vec<RangeLabel> {
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
            Kind::NoEntryPoint => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected `##TITLE=` here".into()),
            }],
            Kind::EmptyKey => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("no label between `##` and `=`".into()),
            }],
            Kind::MultipleKeyTokens => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("second `##` before `=`".into()),
            }],
            Kind::MismatchedDelimiter => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match the most recent opening delimiter".into()),
            }],
            Kind::UnclosedDelimiter => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("not closed before the next key".into()),
            }],
            Kind::UnexpectedPage => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("only valid inside an NTUPLES block".into()),
            }],
            Kind::NestedTuples => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("NTUPLES blocks may not be nested".into()),
            }],
            Kind::MismatchedBlockFormat => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("format specifier does not match this block kind".into()),
            }],
            Kind::Overflow => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not fit into 64-bit signed integer".into()),
            }],
            Kind::Format(inner) => inner.labels(source),
            Kind::Decode(inner) => inner.labels(source),
            Kind::Table(inner) => inner.labels(source),
        }
    }

    fn notes(&self, source: &str) -> Vec<Cow<'static, str>> {
        match self.kind {
            Kind::InvalidLiteral => vec![
                "values must be numeric, strings without whitespace, or enclosed in `<>`".into(),
            ],
            Kind::EndOfInput => {
                vec!["input source ended before the current key/dataset could be processed".into()]
            }
            Kind::NoEntryPoint => {
                vec!["a JCAMP-DX dataset starts with `##TITLE=` and ends with `##END=`".into()]
            }
            Kind::EmptyKey => vec![
                "keys start with `##`, `##.`, or `##$` and are followed by a label and `=`".into(),
            ],
            Kind::MultipleKeyTokens => vec!["a key label may not itself contain `##`".into()],
            Kind::MismatchedDelimiter => {
                vec!["parentheses and angle brackets must be correctly nested".into()]
            }
            Kind::UnclosedDelimiter => vec![
                "parentheses and angle brackets must be closed within the same key-value pair"
                    .into(),
            ],
            Kind::UnexpectedPage => {
                vec!["`##PAGE=` is only valid inside an `##NTUPLES=` block".into()]
            }
            Kind::NestedTuples => vec![
                "`##NTUPLES=` blocks may not contain further `##TITLE=` or `##NTUPLES=` blocks"
                    .into(),
            ],
            Kind::MismatchedBlockFormat => vec![
                "XYDATA requires a format specifier of the form `X++(Y..Y)`".into(),
                "others a format specifier of the form `(XY..XY)` or `(XY)`".into(),
            ],
            Kind::Overflow => {
                vec!["a value of this magnitude is almost certainly not expected".into()]
            }
            Kind::Format(inner) => inner.notes(source),
            Kind::Decode(inner) => inner.notes(source),
            Kind::Table(inner) => inner.notes(source),
        }
    }

    fn fix_hints(&self, source: &str) -> Vec<Cow<'static, str>> {
        let standard = "the software used for exporting might not comply with the standard".into();
        let reexport = "re-exporting the file from the software might fix it".into();

        match self.kind {
            Kind::InvalidLiteral
            | Kind::EmptyKey
            | Kind::MultipleKeyTokens
            | Kind::MismatchedDelimiter
            | Kind::UnclosedDelimiter
            | Kind::UnexpectedPage
            | Kind::NestedTuples
            | Kind::MismatchedBlockFormat => vec![standard, reexport],
            Kind::EndOfInput => vec!["the file may be truncated or corrupted".into(), reexport],
            Kind::NoEntryPoint => vec![
                "the file may not be a JCAMP-DX file".into(),
                "an entry point may be further into the file".into(),
            ],
            Kind::Overflow => vec!["adjacent values may have been combined".into(), reexport],
            Kind::Format(inner) => inner.fix_hints(source),
            Kind::Decode(inner) => inner.fix_hints(source),
            Kind::Table(inner) => inner.fix_hints(source),
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
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

    /// Creates an [`EmptyKey`] error.
    ///
    /// [`EmptyKey`]: Kind::EmptyKey
    pub(crate) fn empty_key(range: ByteRange) -> Self {
        Self {
            kind: Kind::EmptyKey,
            range,
        }
    }

    /// Creates a [`NoEntryPoint`] error.
    ///
    /// [`NoEntryPoint`]: Kind::NoEntryPoint
    pub(crate) fn no_entry_point(range: ByteRange) -> Self {
        Self {
            kind: Kind::NoEntryPoint,
            range,
        }
    }

    /// Creates a [`MultipleKeyTokens`] error.
    ///
    /// [`MultipleKeyTokens`]: Kind::MultipleKeyTokens
    pub(crate) fn multiple_key_tokens(range: ByteRange) -> Self {
        Self {
            kind: Kind::MultipleKeyTokens,
            range,
        }
    }

    /// Creates a [`MismatchedDelimiter`] error.
    ///
    /// [`MismatchedDelimiter`]: Kind::MismatchedDelimiter
    pub(crate) fn mismatched_delimiter(range: ByteRange) -> Self {
        Self {
            kind: Kind::MismatchedDelimiter,
            range,
        }
    }

    /// Creates an [`UnclosedDelimiter`] error.
    ///
    /// [`UnclosedDelimiter`]: Kind::UnclosedDelimiter
    pub(crate) fn unclosed_delimiter(range: ByteRange) -> Self {
        Self {
            kind: Kind::UnclosedDelimiter,
            range,
        }
    }

    /// Creates a [`UnexpectedPage`] error.
    ///
    /// [`UnexpectedPage`]: Kind::UnexpectedPage
    pub(crate) fn unexpected_page(range: ByteRange) -> Self {
        Self {
            kind: Kind::UnexpectedPage,
            range,
        }
    }

    /// Creates a [`NestedTuples`] error.
    ///
    /// [`NestedTuples`]: Kind::NestedTuples
    pub(crate) fn nested_tuples(range: ByteRange) -> Self {
        Self {
            kind: Kind::NestedTuples,
            range,
        }
    }

    /// Creates a [`MismatchedBlockFormat`] error.
    ///
    /// [`MismatchedBlockFormat`]: Kind::MismatchedBlockFormat
    pub(crate) fn mismatched_block_format(range: ByteRange) -> Self {
        Self {
            kind: Kind::MismatchedBlockFormat,
            range,
        }
    }

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(range: ByteRange) -> Self {
        Self {
            kind: Kind::Overflow,
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
