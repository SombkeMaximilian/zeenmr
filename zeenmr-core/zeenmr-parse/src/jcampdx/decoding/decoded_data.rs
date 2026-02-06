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

/// Stack for processing decoded values.
#[derive(Clone, PartialEq, Debug)]
enum DecodedStack {
    /// Standard integer mode.
    I64(Vec<i64>),
    /// Upgraded float mode upon encountering an `f64` value.
    F64(Vec<f64>),
}

impl DecodedStack {
    /// Creates a new, empty `DecodedStack`.
    fn new() -> Self {
        Self::I64(Vec::new())
    }

    /// Returns `true` if the stack is in the [`DecodedStack::I64`] variant.
    fn is_i64(&self) -> bool {
        match self {
            Self::I64(_) => true,
            Self::F64(_) => false,
        }
    }

    /// Returns the number of elements in the `DecodedStack`.
    fn len(&self) -> usize {
        match self {
            Self::I64(buffer) => buffer.len(),
            Self::F64(buffer) => buffer.len(),
        }
    }

    /// Returns the `i64` at the top of the decoded value stack, or `None` if
    /// it is empty.
    ///
    /// # Panics
    ///
    /// Panics if the stack is in the [`DecodedStack::F64`] variant.
    fn unwrap_i64_top(&self) -> Option<&i64> {
        match self {
            Self::I64(buffer) => buffer.last(),
            Self::F64(_) => unreachable!(),
        }
    }

    /// Returns a mutable reference to the `i64` at the top of the decoded value
    /// stack, or `None` if it is empty.
    ///
    /// # Panics
    ///
    /// Panics if the stack is in the [`DecodedStack::F64`] variant.
    fn unwrap_i64_top_mut(&mut self) -> Option<&mut i64> {
        match self {
            Self::I64(buffer) => buffer.last_mut(),
            Self::F64(_) => unreachable!(),
        }
    }

    /// Pushes a decoded `i64` value onto the stack.
    ///
    /// Converts to `f64` if in the [`DecodedStack::F64`] variant.
    fn push_i64(&mut self, value: i64) {
        match self {
            Self::I64(buffer) => buffer.push(value),
            Self::F64(buffer) => buffer.push(value as f64),
        }
    }

    /// Pushes a decoded `f64` value onto the stack.
    ///
    /// Upgrades the buffer to `Vec<f64>` if in the [`DecodedStack::I64`]
    /// variant.
    fn push_f64(&mut self, value: f64) {
        match self {
            Self::I64(buffer) => {
                let mut buffer = std::mem::take(buffer)
                    .into_iter()
                    .map(|v| v as f64)
                    .collect::<Vec<f64>>();
                buffer.push(value);
                *self = Self::F64(buffer);
            }
            Self::F64(buffer) => buffer.push(value),
        }
    }

    /// Extends the decoded value stack with the contents of an iterator.
    ///
    /// Converts to `f64` if in the [`DecodedStack::F64`] variant.
    fn extend<I: IntoIterator<Item = i64>>(&mut self, iter: I) {
        match self {
            Self::I64(buffer) => buffer.extend(iter),
            Self::F64(buffer) => buffer.extend(iter.into_iter().map(|i| i as f64)),
        }
    }
}

/// Builder pattern for [`DecodedBlock`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct DecodedBlockBuilder {
    /// Identifier of the incrementing variable.
    incrementing: String,
    /// Identifier of the repeating variable.
    repeating: String,
    /// Decoded data points.
    decoded: DecodedStack,
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
            decoded: DecodedStack::new(),
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
        match self.decoded {
            DecodedStack::I64(buffer) => {
                table.push(RawColumn::<i64> {
                    id: self.repeating,
                    values: buffer,
                });
            }
            DecodedStack::F64(buffer) => {
                table.push(RawColumn::<f64> {
                    id: self.repeating,
                    values: buffer,
                })
            }
        }

        DecodedBlock {
            table,
            errors: self.errors,
        }
    }

    /// Returns `true` if the stack is in the [`DecodedStack::I64`] variant.
    pub(crate) fn decoded_is_i64(&self) -> bool {
        self.decoded.is_i64()
    }

    /// Returns the number of elements in the decoded values stack.
    pub(crate) fn decoded_len(&self) -> usize {
        self.decoded.len()
    }

    /// Returns the top `i64` of the decoded value stack, or `None` if it is
    /// empty.
    ///
    /// # Panics
    ///
    /// Panics if the stack is in the [`DecodedStack::F64`] variant.
    pub(crate) fn decoded_top(&self) -> Option<&i64> {
        self.decoded.unwrap_i64_top()
    }

    /// Returns a mutable reference to the top `i64` of the decoded value stack,
    /// or `None` if it is empty.
    ///
    /// # Panics
    ///
    /// Panics if the stack is in the [`DecodedStack::F64`] variant.
    pub(crate) fn decoded_top_mut(&mut self) -> Option<&mut i64> {
        self.decoded.unwrap_i64_top_mut()
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

    /// Pushes a decoded `i64` value onto the stack.
    pub(crate) fn push_decoded_i64(&mut self, value: i64) {
        self.decoded.push_i64(value);
    }

    /// Pushes a decoded `f64` value onto the stack.
    pub(crate) fn push_decoded_f64(&mut self, value: f64) {
        self.decoded.push_f64(value);
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
