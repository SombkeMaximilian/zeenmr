//! JCAMP-DX parsing error types.

use crate::Position;
use std::num::ParseIntError;
use std::sync::Arc;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out `Result<T, zeenmr_parse::Error>`
/// directly.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Default)]
pub struct Error {
    kind: Kind,
    position: Position,
    source: Option<Arc<dyn std::error::Error + Send + Sync>>,
}

#[non_exhaustive]
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum Kind {
    #[default]
    InvalidLiteral,
    NoEntryPoint,
    EmptyKey,
    MultipleKeyTokens,
    MismatchedDelimiter,
    UnclosedDelimiter,
    UnexpectedPage,
    NestedTuples,
    MismatchedBlockFormat,
    Overflow,
    BlockFormat,
    Decode,
    Tabulate,
    EndOfInput,
}

impl From<crate::jcampdx::block_format::error::Error> for Error {
    fn from(value: crate::jcampdx::block_format::error::Error) -> Self {
        Self {
            kind: Kind::BlockFormat,
            position: value.position(),
            source: Some(Arc::new(value)),
        }
    }
}

impl From<crate::jcampdx::decoding::error::Error> for Error {
    fn from(value: crate::jcampdx::decoding::error::Error) -> Self {
        Self {
            kind: Kind::Decode,
            position: value.position(),
            source: Some(Arc::new(value)),
        }
    }
}

impl From<crate::jcampdx::tabulation::error::Error> for Error {
    fn from(value: crate::jcampdx::tabulation::error::Error) -> Self {
        Self {
            kind: Kind::Tabulate,
            position: value.position(),
            source: Some(Arc::new(value)),
        }
    }
}

impl Eq for Error {}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.position == other.position
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as _)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let description = match self.kind() {
            Kind::InvalidLiteral => "invalid literal",
            Kind::NoEntryPoint => "no entry point",
            Kind::EmptyKey => "empty key",
            Kind::MultipleKeyTokens => "multiple key tokens",
            Kind::MismatchedDelimiter => "mismatched delimiter",
            Kind::UnclosedDelimiter => "unclosed delimiter",
            Kind::UnexpectedPage => "page outside of tuples",
            Kind::NestedTuples => "nested tuples",
            Kind::MismatchedBlockFormat => "mismatched block format",
            Kind::Overflow => "overflow",
            Kind::BlockFormat => "block format",
            Kind::Decode => "decode error",
            Kind::Tabulate => "tabulate error",
            Kind::EndOfInput => "end of input",
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
            source: None,
        }
    }

    /// Creates an [`EmptyKey`] error.
    ///
    /// [`EmptyKey`]: Kind::EmptyKey
    pub(crate) fn empty_key(position: Position) -> Self {
        Self {
            kind: Kind::EmptyKey,
            position,
            source: None,
        }
    }

    /// Creates a [`NoEntryPoint`] error.
    ///
    /// [`NoEntryPoint`]: Kind::NoEntryPoint
    pub(crate) fn no_entry_point(position: Position) -> Self {
        Self {
            kind: Kind::NoEntryPoint,
            position,
            source: None,
        }
    }

    /// Creates a [`MultipleKeyTokens`] error.
    ///
    /// [`MultipleKeyTokens`]: Kind::MultipleKeyTokens
    pub(crate) fn multiple_key_tokens(position: Position) -> Self {
        Self {
            kind: Kind::MultipleKeyTokens,
            position,
            source: None,
        }
    }

    /// Creates a [`MismatchedDelimiter`] error.
    ///
    /// [`MismatchedDelimiter`]: Kind::MismatchedDelimiter
    pub(crate) fn mismatched_delimiter(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedDelimiter,
            position,
            source: None,
        }
    }

    /// Creates an [`UnclosedDelimiter`] error.
    ///
    /// [`UnclosedDelimiter`]: Kind::UnclosedDelimiter
    pub(crate) fn unclosed_delimiter(position: Position) -> Self {
        Self {
            kind: Kind::UnclosedDelimiter,
            position,
            source: None,
        }
    }

    /// Creates a [`UnexpectedPage`] error.
    ///
    /// [`UnexpectedPage`]: Kind::UnexpectedPage
    pub(crate) fn unexpected_page(position: Position) -> Self {
        Self {
            kind: Kind::UnexpectedPage,
            position,
            source: None,
        }
    }

    /// Creates a [`NestedTuples`] error.
    ///
    /// [`NestedTuples`]: Kind::NestedTuples
    pub(crate) fn nested_tuples(position: Position) -> Self {
        Self {
            kind: Kind::NestedTuples,
            position,
            source: None,
        }
    }

    /// Creates a [`MismatchedBlockFormat`] error.
    ///
    /// [`MismatchedBlockFormat`]: Kind::MismatchedBlockFormat
    pub(crate) fn mismatched_block_format(position: Position) -> Self {
        Self {
            kind: Kind::MismatchedBlockFormat,
            position,
            source: None,
        }
    }

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(position: Position, source: ParseIntError) -> Self {
        Self {
            kind: Kind::Overflow,
            position,
            source: Some(Arc::new(source)),
        }
    }

    /// Creates an [`EndOfInput`] error.
    ///
    /// [`EndOfInput`]: Kind::EndOfInput
    pub(crate) fn end_of_input(position: Position) -> Self {
        Self {
            kind: Kind::EndOfInput,
            position,
            source: None,
        }
    }

    /// Returns the `Kind` of error that occurred.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// Returns the position in the source that caused the error.
    pub fn position(&self) -> Position {
        self.position
    }

    /// Returns the source of the error, if any.
    pub fn source(&self) -> Option<&(dyn std::error::Error + Send + Sync)> {
        self.source.as_deref()
    }
}
