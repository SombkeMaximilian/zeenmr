//! JCAMP-DX grouped block tabulation error types.

use crate::error::{ByteRange, ParseError, RangeLabel};
use std::borrow::Cow;

/// A specialized [`Result`] type.
///
/// [`Result`]: std::result::Result
///
/// This type alias avoids writing out the full type name directly, and is
/// broadly used across the library.
pub(crate) type Result<T> = std::result::Result<T, Error>;

/// An `Error` that occurred during tabulation of a block.
///
/// See the [`Kind`] enum for the different kinds of errors that can occur.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Error {
    /// The `Kind` of error that occurred.
    kind: Kind,
    /// Byte range in the source.
    range: ByteRange,
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
    /// A group with too many elements was encountered.
    GroupTooLarge,
    /// A group with too few elements was encountered.
    GroupTooSmall,
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

impl ParseError for Error {
    fn message(&self) -> Cow<'static, str> {
        match self.kind {
            Kind::InvalidLiteral => "invalid literal".into(),
            Kind::Overflow => "value magnitude too large to fit into i64".into(),
            Kind::GroupTooLarge => "group has too many elements".into(),
            Kind::GroupTooSmall => "group has too few elements".into(),
            Kind::MismatchedGroupDelimiter => "mismatched group delimiter".into(),
            Kind::UnmatchedStringDelimiter => "unmatched string delimiter".into(),
            Kind::NonSeparatedValues => "consecutive values are not separated".into(),
            Kind::CrossLineGroup => "group extends past end of line".into(),
        }
    }

    fn labels(&self, _: &str) -> Vec<RangeLabel> {
        match self.kind {
            Kind::InvalidLiteral => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not match any token".into()),
            }],
            Kind::Overflow => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("does not fit into 64-bit signed integer".into()),
            }],
            Kind::GroupTooLarge => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("value(s) following this comma exceeds group size".into()),
            }],
            Kind::GroupTooSmall => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("current group ends too early here".into()),
            }],
            Kind::MismatchedGroupDelimiter => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("unexpected delimiter here".into()),
            }],
            Kind::UnmatchedStringDelimiter => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("no matching delimiter found".into()),
            }],
            Kind::NonSeparatedValues => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("expected `,` before this value".into()),
            }],
            Kind::CrossLineGroup => vec![RangeLabel {
                range: self.range,
                is_cause: true,
                label: Some("group not closed before end of line".into()),
            }],
        }
    }

    fn notes(&self, source: &str) -> Vec<Cow<'static, str>> {
        match self.kind {
            Kind::InvalidLiteral => vec![
                "values must be numeric, strings without whitespace, or enclosed in `<>`".into(),
            ],
            Kind::Overflow => {
                vec!["a value of this magnitude is almost certainly not expected".into()]
            }
            Kind::GroupTooLarge | Kind::GroupTooSmall => vec![
                "groups are the rows in the resulting table".into(),
                "mismatched group size corrupts the whole table".into(),
            ],
            Kind::MismatchedGroupDelimiter => {
                let semicolon = "semicolons may not terminate a group enclosed in `()`".into();
                let parentheses = "each `(` must be closed by a matching `)`".into();
                let auto_start = "a value may also start a new group".into();

                match &source[self.range.as_range()] {
                    ";" => vec![semicolon],
                    "(" => vec![parentheses],
                    ")" => vec![parentheses, auto_start],
                    _ => vec![semicolon, parentheses, auto_start],
                }
            }
            Kind::UnmatchedStringDelimiter => {
                vec!["strings with whitespace must be enclosed in matching `<` and `>`".into()]
            }
            Kind::NonSeparatedValues => {
                vec!["values inside a group must be separated by `,`".into()]
            }
            Kind::CrossLineGroup => {
                vec!["groups spanning multiple lines must be enclosed in `(` and `)`".into()]
            }
        }
    }

    fn fix_hints(&self, _: &str) -> Vec<Cow<'static, str>> {
        let adjacent = "adjacent values may have been combined".into();
        let split = "a value may have been split into two, such as due to missing `<>`".into();
        let reexport = "re-exporting the file from the software might fix it".into();
        let standard = "the software used for exporting might not comply with the standard".into();

        match self.kind {
            Kind::InvalidLiteral
            | Kind::MismatchedGroupDelimiter
            | Kind::UnmatchedStringDelimiter => vec![standard, reexport],
            Kind::Overflow => vec![adjacent, reexport],
            Kind::GroupTooLarge => vec![split, reexport],
            Kind::GroupTooSmall => vec![adjacent, reexport],
            Kind::NonSeparatedValues => vec![split, standard, reexport],
            Kind::CrossLineGroup => {
                vec!["value(s) may have been skipped".into(), standard, reexport]
            }
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

    /// Creates an [`Overflow`] error.
    ///
    /// [`Overflow`]: Kind::Overflow
    pub(crate) fn overflow(range: ByteRange) -> Self {
        Self {
            kind: Kind::Overflow,
            range,
        }
    }

    /// Creates a [`GroupTooLarge`] error.
    ///
    /// [`GroupTooLarge`]: Kind::GroupTooLarge
    pub(crate) fn group_too_large(range: ByteRange) -> Self {
        Self {
            kind: Kind::GroupTooLarge,
            range,
        }
    }

    /// Creates a [`GroupTooSmall`] error.
    ///
    /// [`GroupTooSmall`]: Kind::GroupTooSmall
    pub(crate) fn group_too_small(range: ByteRange) -> Self {
        Self {
            kind: Kind::GroupTooSmall,
            range,
        }
    }

    /// Creates a [`MismatchedGroupDelimiter`] error.
    ///
    /// [`MismatchedGroupDelimiter`]: Kind::MismatchedGroupDelimiter
    pub(crate) fn mismatched_group_delimiter(range: ByteRange) -> Self {
        Self {
            kind: Kind::MismatchedGroupDelimiter,
            range,
        }
    }

    /// Creates a [`UnmatchedStringDelimiter`] error.
    ///
    /// [`UnmatchedStringDelimiter`]: Kind::UnmatchedStringDelimiter
    pub(crate) fn unmatched_string_delimiter(range: ByteRange) -> Self {
        Self {
            kind: Kind::UnmatchedStringDelimiter,
            range,
        }
    }

    /// Creates a [`NonSeparatedValues`] error.
    ///
    /// [`NonSeparatedValues`]: Kind::NonSeparatedValues
    pub(crate) fn non_separated_values(range: ByteRange) -> Self {
        Self {
            kind: Kind::NonSeparatedValues,
            range,
        }
    }

    /// Creates a [`CrossLineGroup`] error.
    ///
    /// [`CrossLineGroup`]: Kind::CrossLineGroup
    pub(crate) fn cross_line_group(range: ByteRange) -> Self {
        Self {
            kind: Kind::CrossLineGroup,
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
