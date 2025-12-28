use crate::jcampdx::decoding::EncodedToken;
use crate::jcampdx::decoding::error::Error;
use logos::Lexer;

/// Checkpoints in a decoded data block.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) struct CheckPoint {
    /// Index in the decoded data.
    index: usize,
    /// Value of the checkpoint.
    value: i64,
}

impl CheckPoint {
    /// Creates a new `CheckPoint` with the given index and value.
    pub(crate) fn new(index: usize, value: i64) -> Self {
        Self { index, value }
    }

    /// Returns the index of the `CheckPoint`.
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    /// Returns the value of the `CheckPoint`.
    pub(crate) fn value(&self) -> i64 {
        self.value
    }
}

/// Decoded data block
#[derive(Clone, Eq, PartialEq, Debug)]
pub(crate) struct DecodedBlock<T> {
    /// Decoded values.
    values: Vec<T>,
    /// Checkpoints in the block.
    check_points: Vec<CheckPoint>,
    /// Non-fatal errors during decoding.
    errors: Vec<Error>,
}

impl<T> DecodedBlock<T> {
    /// Creates a new `DecodedBlock` from the decoded values, checkpoints and
    /// errors encountered during decoding, if any.
    pub(crate) fn new(values: Vec<T>, check_points: Vec<CheckPoint>, errors: Vec<Error>) -> Self {
        Self {
            values,
            check_points,
            errors,
        }
    }

    /// Returns the decoded values.
    pub(crate) fn values(&self) -> &[T] {
        &self.values
    }

    /// Returns the checkpoints.
    pub(crate) fn check_points(&self) -> &[CheckPoint] {
        &self.check_points
    }

    /// Returns the errors that occurred during decoding.
    pub(crate) fn errors(&self) -> &[Error] {
        &self.errors
    }
}

#[derive(Clone, Debug)]
pub(crate) enum DecodeExit<'source, T> {
    EndOfInput(DecodedBlock<T>),
    HeaderKey(DecodedBlock<T>, Lexer<'source, EncodedToken>),
}

impl<'source, T> DecodeExit<'source, T> {
    pub(crate) fn block(&self) -> &DecodedBlock<T> {
        match self {
            DecodeExit::EndOfInput(block) => block,
            DecodeExit::HeaderKey(block, ..) => block,
        }
    }

    pub(crate) fn lexer(&self) -> Option<&Lexer<'source, EncodedToken>> {
        match self {
            DecodeExit::EndOfInput(_) => None,
            DecodeExit::HeaderKey(.., lexer) => Some(lexer),
        }
    }
}
