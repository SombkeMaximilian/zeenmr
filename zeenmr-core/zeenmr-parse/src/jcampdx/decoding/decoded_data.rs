use crate::data::DataTable;
use crate::jcampdx::decoding::error::{Error, Kind};
use crate::jcampdx::parser::ChildParserExit;
use std::borrow::Cow;

/// Decoded data block.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct DecodedBlock<'source> {
    /// Exit status of the decoder (end of input or header key).
    pub(crate) exit: ChildParserExit,
    /// Decoded data.
    pub(crate) table: DataTable<'source>,
    /// Non-fatal errors during decoding.
    pub(crate) errors: Vec<Error>,
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

/// Sequence of checkpoints consisting of count and value pairs.
#[derive(Clone, PartialEq, Debug)]
struct CheckPointSequence {
    /// Number of decoded values before the checkpoint.
    counts: Vec<usize>,
    /// Value of the checkpoints.
    values: Vec<f64>,
}

impl CheckPointSequence {
    /// Creates a new `CheckPointSequence`.
    fn new() -> Self {
        Self {
            counts: vec![0],
            values: Vec::new(),
        }
    }

    /// Returns the inferred step size from the last two complete checkpoints,
    /// or `None` if there are fewer than two.
    fn step(&self) -> Option<f64> {
        if self.counts.len() < 2 || self.values.len() < 2 {
            return None;
        }

        let last_index = self.counts.len().min(self.values.len()) - 1;
        let curr_value = self.values[last_index];
        let prev_value = self.values[last_index - 1];
        let curr_count = self.counts[last_index];
        let prev_count = self.counts[last_index - 1];

        Some((curr_value - prev_value) / ((curr_count - prev_count) as f64))
    }

    /// Returns the step size calculated from the first and last checkpoints,
    /// or `None` if there are fewer than two.
    fn max_precision_step(&self) -> Option<f64> {
        if self.counts.len() < 2 || self.values.len() < 2 {
            return None;
        }

        let last_index = self.counts.len().min(self.values.len() - 1);
        let first_value = self.values[0];
        let last_value = self.values[last_index];
        let first_count = self.counts[0];
        let last_count = self.counts[last_index];

        Some((last_value - first_value) / ((last_count - first_count) as f64))
    }
}

/// Builder pattern for [`DecodedBlock`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct DecodedBlockBuilder<'source> {
    /// Exit status of the decoder (end of input or header key).
    exit: ChildParserExit,
    /// Title of the dataset.
    title: Cow<'source, str>,
    /// Identifier of the incrementing variable.
    incrementing: Cow<'source, str>,
    /// Identifier of the repeating variable.
    repeating: Cow<'source, str>,
    /// Decoded data points.
    decoded: DecodedStack,
    /// Recorded checkpoints.
    check_points: CheckPointSequence,
    /// Non-fatal errors encountered.
    errors: Vec<Error>,
}

impl<'source> Default for DecodedBlockBuilder<'source> {
    fn default() -> Self {
        Self {
            exit: ChildParserExit::default(),
            title: "XYDATA".into(),
            incrementing: "Unknown".into(),
            repeating: "Unknown".into(),
            decoded: DecodedStack::new(),
            check_points: CheckPointSequence::new(),
            errors: Vec::new(),
        }
    }
}

impl<'source> DecodedBlockBuilder<'source> {
    /// Finalizes the `DecodedBlock`.
    pub(crate) fn finalize(self) -> DecodedBlock<'source> {
        let mut table = DataTable::new_with_id(self.title);
        let is_valid = !self
            .errors
            .iter()
            .any(|e| e.kind() == Kind::CheckPointValue || e.kind() == Kind::CheckPointStepMismatch);
        if is_valid && let Some(step) = self.check_points.max_precision_step() {
            let first = *(self.check_points.values.first().unwrap());
            let count = self.decoded.len();
            table.insert(
                self.incrementing,
                (0_usize..count)
                    .map(|i| first + step * i as f64)
                    .collect(),
            );
        }
        match self.decoded {
            DecodedStack::I64(buffer) => table.insert(self.repeating, buffer.into()),
            DecodedStack::F64(buffer) => table.insert(self.repeating, buffer.into()),
        };

        DecodedBlock {
            exit: self.exit,
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

    /// Returns the inferred step size from the last two complete checkpoints,
    /// or `None` if there are fewer than two.
    pub(crate) fn checkpoint_step(&self) -> Option<f64> {
        self.check_points.step()
    }

    /// Updates the exit status to having encountered the [`End`] token.
    ///
    /// [`End`]: crate::jcampdx::decoding::EncodedToken::End
    pub(crate) fn header_key_exit(&mut self) {
        self.exit = ChildParserExit::EndToken;
    }

    /// Sets the title of the dataset.
    pub(crate) fn set_title<T: Into<Cow<'source, str>>>(&mut self, title: T) {
        self.title = title.into();
    }

    /// Sets the identifier of the incrementing variable.
    pub(crate) fn set_incrementing<T: Into<Cow<'source, str>>>(&mut self, incrementing: T) {
        self.incrementing = incrementing.into();
    }

    /// Sets the identifier of the repeating variable.
    pub(crate) fn set_repeating<T: Into<Cow<'source, str>>>(&mut self, repeating: T) {
        self.repeating = repeating.into();
    }

    /// Pushes a checkpoint onto the stack.
    pub(crate) fn checkpoint(&mut self) {
        self.check_points.counts.push(self.decoded.len());
    }

    /// Pushes a checkpoint onto the stack, accounting for an upcoming integrity
    /// check.
    pub(crate) fn checkpoint_integrity_check(&mut self) {
        self.check_points
            .counts
            .push(self.decoded.len() - 1);
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
        self.check_points.values.push(value);
    }

    /// Pushes an error onto the stack.
    pub(crate) fn push_error(&mut self, error: Error) {
        self.errors.push(error);
    }
}
