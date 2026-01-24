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

    /// Performs an integrity check against the most recently decoded value.
    ///
    /// If the check fails, the top value of the stack is overwritten with
    /// `check`.
    ///
    /// Returns
    /// - `Some(true)` if the stack is non-empty and `check` is equal to the top
    ///   value.
    /// - `Some(false)` if the stack is non-empty and `check` differs from the
    ///   top value. In this case, the top value is replaced with `check`.
    /// - `None` if the stack is empty.
    pub(crate) fn integrity_check(&mut self, check: i64) -> Option<bool> {
        let result = self.decoded.last().map(|value| *value == check);

        match result {
            Some(false) => {
                *(self.decoded.last_mut().unwrap()) = check;

                Some(false)
            }
            _ => result,
        }
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

    /// Computes the next value as a difference to the most recent decoded value
    /// and pushes it onto the stack.
    ///
    /// # Panics
    ///
    /// Panics if the decoded values stack is empty.
    pub(crate) fn push_difference(&mut self, difference: i64) {
        self.decoded
            .push(*(self.decoded.last().unwrap()) + difference);
    }

    /// Extends the duplicate stack with a number of repetitions of the most
    /// recent decoded value.
    ///
    /// If difference is not `None`, the contained difference is repeatedly
    /// applied.
    ///
    /// # Panics
    ///
    /// Panics if the decoded values stack is empty.
    pub(crate) fn push_duplicate(&mut self, num: usize, difference: Option<i64>) {
        let previous = *(self.decoded.last().unwrap());
        match difference {
            Some(difference) => {
                let values = (1..num).map(|i| previous + (difference * i as i64));
                self.decoded.extend(values);
            }
            None => {
                self.decoded
                    .extend(std::iter::repeat(previous).take(num - 1));
            }
        }
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
