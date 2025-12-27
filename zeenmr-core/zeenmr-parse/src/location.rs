/// File cursor for reporting locations in errors.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Cursor {
    /// Line in the source.
    pub(crate) line: usize,
    /// Character index in the source.
    pub(crate) index: usize,
}

/// Position of a token in the source.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Position {
    /// Line in the source.
    pub(crate) line: usize,
    /// Column in the source.
    pub(crate) column: usize,
}
