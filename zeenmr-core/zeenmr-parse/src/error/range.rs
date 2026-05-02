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
        Range::<usize> {
            start: value.start,
            end: value.end,
        }
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
}
