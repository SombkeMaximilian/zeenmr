use crate::error::Position;

/// Stack frame for values of type `V` enclosed by delimiters of type `D`.
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct Frame<D, V> {
    /// Delimiter that opened the stack frame.
    pub(crate) delimiter: D,
    /// Position of the opening token in the source.
    pub(crate) start: Position,
    /// Values after the opening token.
    pub(crate) values: Vec<V>,
}

/// LIFO stack for tracking bounded structures
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub(crate) struct Stack<D, V> {
    /// Stack frames corresponding to units enclosed by delimiters.
    frames: Vec<Frame<D, V>>,
}

impl<D, V> Stack<D, V> {
    /// Creates a new `Stack`.
    pub(crate) fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Returns the `Frame` at the top of the stack, or `None` if it is empty.
    pub(crate) fn top(&self) -> Option<&Frame<D, V>> {
        self.frames.last()
    }

    /// Returns a mutable reference to the `Frame` at the top of the stack, or
    /// `None` if it is empty.
    pub(crate) fn top_mut(&mut self) -> Option<&mut Frame<D, V>> {
        self.frames.last_mut()
    }

    /// Returns the delimiter of the `Frame` at the top of the stack, or `None`
    /// if it is empty.
    pub(crate) fn top_delimiter(&self) -> Option<&D> {
        self.frames.last().map(|last| &last.delimiter)
    }

    /// Pushes a new `Frame` onto the stack.
    pub(crate) fn push(&mut self, delimiter: D, start: Position) {
        self.frames.push(Frame {
            delimiter,
            start,
            values: Vec::new(),
        });
    }

    /// Removes and returns the `Frame` at the top of the stack, or `None` if it
    /// is empty.
    pub(crate) fn pop(&mut self) -> Option<Frame<D, V>> {
        self.frames.pop()
    }
}
