use crate::jcampdx::decoding::error::Error;
use crate::jcampdx::{RawColumn, Table};

/// Decoded data block
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct DecodedBlock {
    /// Decoded data.
    pub(crate) table: Table,
    /// Non-fatal errors during decoding.
    pub(crate) errors: Vec<Error>,
}

impl DecodedBlock {
    /// Builder pattern for a new `DecodedBlock`.
    pub(crate) fn builder() -> DecodedBlockBuilder {
        DecodedBlockBuilder::default()
    }
}

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct DecodedBlockBuilder {
    /// Identifier of the incrementing variable.
    incrementing: String,
    /// Identifier of the repeating variable.
    repeating: String,
    /// Decoded data points.
    decoded: Vec<i64>,
    /// Checkpoint indices.
    check_points: Vec<usize>,
    /// Checkpoint values
    check_point_values: Vec<f64>,
    /// Non-fatal errors encountered.
    errors: Vec<Error>,
}

impl Default for DecodedBlockBuilder {
    fn default() -> Self {
        Self {
            incrementing: "Unknown".to_string(),
            repeating: "Unknown".to_string(),
            decoded: Vec::new(),
            check_points: vec![0],
            check_point_values: Vec::new(),
            errors: Vec::new(),
        }
    }
}

impl DecodedBlockBuilder {
    /// Finalizes the `DecodedBlock`.
    pub(crate) fn finalize(mut self) -> DecodedBlock {
        let mut table = Table::new();
        table.push(RawColumn::<i64> {
            id: self.repeating,
            values: self.decoded,
        });

        DecodedBlock {
            table,
            errors: self.errors,
        }
    }

    /// Returns the number of elements in the decoded values stack.
    pub(crate) fn decoded_len(&self) -> usize {
        self.decoded.len()
    }

    /// Returns the top of the decoded value stack, or `None` if it is empty.
    pub(crate) fn decoded_top(&self) -> Option<&i64> {
        self.decoded.last()
    }

    /// Returns a mutable reference to the top of the decoded value stack, or
    /// `None` if it is empty.
    pub(crate) fn decoded_top_mut(&mut self) -> Option<&mut i64> {
        self.decoded.last_mut()
    }

    /// Sets the identifier of the incrementing variable.
    pub(crate) fn set_incrementing<T: Into<String>>(&mut self, incrementing: T) {
        self.incrementing = incrementing.into();
    }

    /// Sets the identifier of the repeating variable.
    pub(crate) fn set_repeating<T: Into<String>>(&mut self, repeating: T) {
        self.repeating = repeating.into();
    }

    /// Pushes a checkpoint onto the stack.
    pub(crate) fn checkpoint(&mut self) {
        self.check_points.push(self.decoded.len());
    }

    /// Pushes a checkpoint onto the stack, accounting for an upcoming integrity
    /// check.
    pub(crate) fn checkpoint_integrity_check(&mut self) {
        self.check_points.push(self.decoded.len() - 1);
    }

    /// Pushes a decoded value onto the stack.
    pub(crate) fn push_decoded_value(&mut self, value: i64) {
        self.decoded.push(value);
    }

    /// Extends the decoded value stack with the contents of an iterator.
    pub(crate) fn extend_decoded<I: IntoIterator<Item = i64>>(&mut self, iter: I) {
        self.decoded.extend(iter);
    }

    /// Pushes a checkpoint value onto the stack.
    pub(crate) fn push_checkpoint_value(&mut self, value: f64) {
        self.check_point_values.push(value);
    }

    /// Pushes an error onto the stack.
    pub(crate) fn push_error(&mut self, error: Error) {
        self.errors.push(error);
    }
}
