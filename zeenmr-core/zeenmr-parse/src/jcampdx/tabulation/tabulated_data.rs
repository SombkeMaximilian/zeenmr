use crate::data::{Column, DataTable, Value};
use crate::jcampdx::parser::ChildParserExit;
use crate::jcampdx::tabulation::error::Error;
use std::borrow::Cow;

/// Tabulated data block.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlock<'source> {
    /// Exit status of the parser (end of input or header key).
    pub(crate) exit: ChildParserExit,
    /// Tabulated data.
    pub(crate) table: DataTable<'source>,
    /// Non-fatal errors during tabulation.
    pub(crate) errors: Vec<Error>,
}

/// Dynamically upgrading buffer that accepts `i64`, `f64`, `String` and `Value`
/// elements.
///
/// # Worst Case Guarantee
///
/// A buffer that starts out in the `Integer` state can at most be upgraded
/// twice: to `Float` and then to `Mixed`. A buffer that starts out as `Float`
/// or `String` can at most be upgraded once. A `Mixed` buffer never gets
/// upgraded.
#[derive(Clone, PartialEq, Debug)]
enum UpgradingBuffer<'source> {
    /// Only integers.
    Integer(Vec<i64>),
    /// Only floats.
    Float(Vec<f64>),
    /// Only strings.
    String(Vec<Cow<'source, str>>),
    /// Potentially mixed values.
    Mixed(Vec<Value<'source>>),
}

impl<'source> From<UpgradingBuffer<'source>> for Column<'source> {
    fn from(value: UpgradingBuffer<'source>) -> Self {
        match value {
            UpgradingBuffer::Integer(values) => values.into(),
            UpgradingBuffer::Float(values) => values.into(),
            UpgradingBuffer::String(values) => values.into(),
            UpgradingBuffer::Mixed(values) => values.into(),
        }
    }
}

impl<'source> UpgradingBuffer<'source> {
    /// Creates a new, empty `Integer` buffer.
    fn new_integer() -> Self {
        Self::Integer(Vec::new())
    }

    /// Creates a new, empty `Float` buffer.
    fn new_float() -> Self {
        Self::Float(Vec::new())
    }

    /// Creates a new, empty `String` buffer.
    fn new_string() -> Self {
        Self::String(Vec::new())
    }

    /// Creates a new, empty `Mixed` buffer.
    fn new_mixed() -> Self {
        Self::Mixed(Vec::new())
    }

    /// Helper for converting inner `Vec<i64>` to `Vec<f64>`, appending a new
    /// `f64` to it and returning a `Float` buffer.
    fn convert_to_float_and_push(inner: Vec<i64>, value: f64) -> Self {
        let mut converted = inner
            .into_iter()
            .map(|value| value as f64)
            .collect::<Vec<f64>>();
        converted.push(value);

        Self::Float(converted)
    }

    /// Helper for converting inner `Vec<T>` to `Vec<Value>`, appending a new
    /// value to it and returning a `Mixed` buffer.
    fn convert_to_mixed_and_push<T, U>(inner: Vec<T>, value: U) -> Self
    where
        T: Into<Value<'source>>,
        U: Into<Value<'source>>,
    {
        let mut converted = inner
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Value<'source>>>();
        converted.push(value.into());

        Self::Mixed(converted)
    }

    /// Appends an `i64` to the column, converting the input if necessary.
    ///
    /// # Upgrading Rules
    ///
    /// | **Buffer**  | **Convert Input**  | **Upgrade Buffer** |
    /// | ----------- | ------------------ | ------------------ |
    /// | Integer     | No                 | No                 |
    /// | Float       | `f64`              | No                 |
    /// | String      | [`Value::Integer`] | [`Column::Mixed`]  |
    /// | Mixed       | [`Value::Integer`] | No                 |
    fn push_i64(&mut self, value: i64) {
        match self {
            Self::Integer(inner) => inner.push(value),
            Self::Float(inner) => inner.push(value as f64),
            Self::String(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Mixed(inner) => inner.push(value.into()),
        }
    }

    /// Appends an `f64` to the buffer, converting the input and/or upgrading
    /// the buffer if necessary.
    ///
    /// # Upgrading Rules
    ///
    /// | **Buffer**  | **Convert Input**   | **Upgrade Buffer**   |
    /// | ----------- | ------------------- | -------------------- |
    /// | `Integer`   | `i64` if input ∈ ℤ  | `Float` if input ∉ ℤ |
    /// | `Float`     | No                  | No                   |
    /// | `String`    | [`Value::Float`]    | `Mixed`              |
    /// | `Mixed`     | [`Value::Float`]    | No                   |
    fn push_f64(&mut self, value: f64) {
        match self {
            Self::Integer(inner) if value.fract() == 0.0 => inner.push(value as i64),
            Self::Integer(inner) => {
                *self = Self::convert_to_float_and_push(std::mem::take(inner), value);
            }
            Self::Float(inner) => inner.push(value),
            Self::String(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Mixed(inner) => inner.push(value.into()),
        }
    }

    /// Appends a `String` to the buffer, converting the input and/or upgrading
    /// the buffer if necessary.
    ///
    /// # Upgrading Rules
    ///
    /// | **Buffer**  | **Convert Input** | **Upgrade Buffer** |
    /// | ----------- | ----------------- | ------------------ |
    /// | `Integer`   | ---               | `Mixed`            |
    /// | `Float`     | ---               | `Mixed`            |
    /// | `String`    | ---               | ---                |
    /// | `Mixed`     | [`Value::String`] | ---                |
    fn push_string<T: Into<Cow<'source, str>>>(&mut self, value: T) {
        let value = value.into();

        match self {
            Self::Integer(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Float(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::String(inner) => inner.push(value),
            Self::Mixed(inner) => inner.push(value.into()),
        }
    }

    /// Appends a `Value` to the buffer, upgrading it if necessary.
    ///
    /// # Upgrading Rules
    ///
    /// | **Buffer**  | **Convert Input** | **Upgrade Buffer** |
    /// | ----------- | ----------------- | ------------------ |
    /// | `Integer`   | ---               | `Mixed`            |
    /// | `Float`     | ---               | `Mixed`            |
    /// | `String`    | ---               | `Mixed`            |
    /// | `Mixed`     | ---               | ---                |
    fn push_value<T: Into<Value<'source>>>(&mut self, value: T) {
        match self {
            Self::Integer(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Float(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::String(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Mixed(inner) => inner.push(value.into()),
        }
    }
}

/// Constructs a collection of buffers by pushing to them in a cycle.
#[derive(Clone, PartialEq, Debug, Default)]
struct BufferCycle<'source> {
    /// Cycling buffers.
    buffers: Vec<UpgradingBuffer<'source>>,
    /// Current buffer to push to.
    current: usize,
}

impl<'source> IntoIterator for BufferCycle<'source> {
    type Item = UpgradingBuffer<'source>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.buffers.into_iter()
    }
}

impl<'source> BufferCycle<'source> {
    /// Creates a new, empty `BufferCycle`.
    fn new() -> Self {
        Self {
            buffers: Vec::new(),
            current: 0,
        }
    }

    /// Returns a mutable reference to the current `UpgradingBuffer`.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn current_mut(&mut self) -> &mut UpgradingBuffer<'source> {
        &mut self.buffers[self.current]
    }

    /// Clears the cycle, removing all buffers and the values they contain.
    fn clear(&mut self) {
        self.buffers.clear();
        self.current = 0;
    }

    /// Advances to the next buffer in the cycle.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn advance(&mut self) {
        self.current = (self.current + 1) % self.buffers.len();
    }

    /// Adds an `Integer` buffer to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_integer_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_integer());
    }

    /// Adds a `Float` buffer to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_float_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_float());
    }

    /// Adds a `String` buffer to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_string_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_string());
    }

    /// Adds a `Mixed` buffer to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_mixed_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_mixed());
    }

    /// Skips the current buffer in the cycle by inserting an empty value
    /// and advances to the next.
    fn skip_current(&mut self) {
        self.current_mut().push_value(Value::Empty);
        self.advance();
    }

    /// Appends an `i64` to the current buffer in the cycle and advances to the
    /// next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_i64(&mut self, value: i64) {
        self.current_mut().push_i64(value);
        self.advance();
    }

    /// Appends an `f64` to the current buffer in the cycle and advances to the
    /// next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_f64(&mut self, value: f64) {
        self.current_mut().push_f64(value);
        self.advance();
    }

    /// Appends a `String` to the current buffer in the cycle and advances to
    /// the next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_string<T: Into<Cow<'source, str>>>(&mut self, value: T) {
        self.current_mut().push_string(value);
        self.advance();
    }

    /// Appends a `Value` to the current buffer in the cycle and advances to the
    /// next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_value<T: Into<Value<'source>>>(&mut self, value: T) {
        self.current_mut().push_value(value);
        self.advance();
    }
}

/// Builder pattern for [`TabulatedBlock`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlockBuilder<'source> {
    /// Exit status of the parser (end of input or header key).
    exit: ChildParserExit,
    /// Tabulated data.
    title: Cow<'source, str>,
    /// Identifiers of the variables in the table.
    identifiers: Vec<Cow<'source, str>>,
    /// Columns being constructed.
    buffer_cycle: BufferCycle<'source>,
    /// Non-fatal errors during tabulation.
    errors: Vec<Error>,
}

impl<'source> Default for TabulatedBlockBuilder<'source> {
    fn default() -> Self {
        Self {
            exit: ChildParserExit::default(),
            title: "XYPOINTS".into(),
            identifiers: Vec::new(),
            buffer_cycle: BufferCycle::default(),
            errors: Vec::new(),
        }
    }
}

impl<'source> TabulatedBlockBuilder<'source> {
    /// Finalizes the `TabulatedBlock`.
    pub(crate) fn finalize(self) -> TabulatedBlock<'source> {
        let columns = self
            .buffer_cycle
            .into_iter()
            .zip(self.identifiers)
            .map(|(buffer, id)| (id, Column::from(buffer)));
        let mut table = DataTable::from_iter(columns);
        table.set_id(self.title);

        TabulatedBlock {
            exit: self.exit,
            table,
            errors: self.errors,
        }
    }

    /// Returns the index of the current column.
    pub(crate) fn current_column_index(&self) -> usize {
        self.buffer_cycle.current
    }

    /// Returns the number of elements in a row.
    pub(crate) fn row_len(&self) -> usize {
        self.buffer_cycle.buffers.len()
    }

    /// Updates the exit status to having encountered the [`End`] token.
    ///
    /// [`End`]: crate::jcampdx::tabulation::GroupToken::End
    pub(crate) fn header_key_exit(&mut self) {
        self.exit = ChildParserExit::EndToken;
    }

    /// Sets the title of the dataset.
    pub(crate) fn set_title<T: Into<Cow<'source, str>>>(&mut self, title: T) {
        self.title = title.into();
    }

    /// Sets columns by inferring the type from the identifiers.
    ///
    /// Using this method clears the current cycle and any inserted values.
    pub(crate) fn set_columns(&mut self, identifiers: Vec<&'source str>) {
        self.buffer_cycle.clear();
        for id in identifiers.iter() {
            match *id {
                "X" | "Y" | "R" | "I" | "M" | "W" => self.buffer_cycle.add_integer_buffer(),
                "F" | "F1" | "F2" => self.buffer_cycle.add_float_buffer(),
                "A" => self.buffer_cycle.add_string_buffer(),
                _ => self.buffer_cycle.add_mixed_buffer(),
            }
        }
        self.identifiers = identifiers.into_iter().map(Into::into).collect();
    }

    /// Skips the current column by inserting a [`Value::Empty`] and advances
    /// to the next.
    pub(crate) fn skip_current(&mut self) {
        self.buffer_cycle.skip_current();
    }

    /// Appends an `i64` to the current column and advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no columns have been added.
    pub(crate) fn push_i64(&mut self, value: i64) {
        self.buffer_cycle.push_i64(value);
    }

    /// Appends an `f64` to the current column and advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no columns have been added.
    pub(crate) fn push_f64(&mut self, value: f64) {
        self.buffer_cycle.push_f64(value);
    }

    /// Appends a `String` to the current column and advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no columns have been added.
    pub(crate) fn push_string<T: Into<String>>(&mut self, value: T) {
        self.buffer_cycle.push_string(value.into());
    }

    /// Appends a `Value` to the current column and advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no columns have been added.
    pub(crate) fn push_value<T: Into<Value<'source>>>(&mut self, value: T) {
        self.buffer_cycle.push_value(value.into());
    }

    /// Pushes an error onto the stack.
    pub(crate) fn push_error(&mut self, error: Error) {
        self.errors.push(error);
    }
}
