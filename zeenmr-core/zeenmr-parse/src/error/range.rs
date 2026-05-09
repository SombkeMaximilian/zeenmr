use std::borrow::Cow;
use std::ops::Range;

/// Byte range in the source.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
pub struct ByteRange {
    /// Start of the range within the source.
    pub start: usize,
    /// End of the range within the source.
    pub end: usize,
}

impl From<Range<usize>> for ByteRange {
    fn from(value: Range<usize>) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

impl From<ByteRange> for Range<usize> {
    fn from(value: ByteRange) -> Self {
        value.start..value.end
    }
}

impl ByteRange {
    /// Constructs a new `ByteRange`.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the byte range as `Range<usize>`.
    pub fn as_range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Returns the length of the range in bytes.
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns `true` if the byte range is of length 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if two byte ranges overlap.
    pub fn overlaps(&self, other: &ByteRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// Labeled byte range for display in error messages.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
pub struct RangeLabel {
    /// Byte range in the source.
    pub range: ByteRange,
    /// Whether this is where the error occurred.
    pub is_cause: bool,
    /// Optional label (if omitted, only highlights).
    pub label: Option<Cow<'static, str>>,
}
