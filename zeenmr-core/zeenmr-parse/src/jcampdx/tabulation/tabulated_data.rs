use crate::jcampdx::tabulation::error::Error;
use crate::jcampdx::{ChildParserExit, Column, Table, Value};

/// Tabulated data block.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlock {
    /// Exit status of the parser (end of input or header key).
    pub(crate) exit: ChildParserExit,
    /// Tabulated data.
    pub(crate) table: Table,
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
enum UpgradingBuffer {
    /// Only integers.
    Integer(Vec<i64>),
    /// Only floats.
    Float(Vec<f64>),
    /// Only strings.
    String(Vec<String>),
    /// Potentially mixed values.
    Mixed(Vec<Value>),
}

impl UpgradingBuffer {
    /// Creates a new, empty [`UpgradingBuffer::Integer`].
    fn new_integer() -> Self {
        Self::Integer(Vec::new())
    }

    /// Creates a new, empty [`UpgradingBuffer::Float`].
    fn new_float() -> Self {
        Self::Float(Vec::new())
    }

    /// Creates a new, empty [`UpgradingBuffer::String`].
    fn new_string() -> Self {
        Self::String(Vec::new())
    }

    /// Creates a new, empty [`UpgradingBuffer::Mixed`].
    fn new_mixed() -> Self {
        Self::Mixed(Vec::new())
    }

    /// Helper for converting inner `Vec<i64>` to `Vec<f64>`, appending a new
    /// `f64` to it and returning an [`UpgradingBuffer::Float`].
    fn convert_to_float_and_push(inner: Vec<i64>, value: f64) -> Self {
        let mut converted = inner
            .into_iter()
            .map(|value| value as f64)
            .collect::<Vec<f64>>();
        converted.push(value);

        Self::Float(converted)
    }

    /// Helper for converting inner `Vec<T>` to `Vec<Value>`, appending a new
    /// value to it and returning an [`UpgradingBuffer::Mixed`].
    fn convert_to_mixed_and_push<T, U>(inner: Vec<T>, value: U) -> Self
    where
        T: Into<Value>,
        U: Into<Value>,
    {
        let mut converted = inner
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Value>>();
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
    fn push_string<T: Into<String>>(&mut self, value: T) {
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
    fn push_value<T: Into<Value>>(&mut self, value: T) {
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
struct BufferCycle {
    /// Cycling buffers.
    buffers: Vec<UpgradingBuffer>,
    /// Current column to push to.
    current: usize,
}

impl BufferCycle {
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
    fn current_mut(&mut self) -> &mut UpgradingBuffer {
        &mut self.buffers[self.current]
    }

    /// Advances to the next [`UpgradingBuffer`] in the cycle.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn advance(&mut self) {
        self.current = (self.current + 1) % self.buffers.len();
    }

    /// Adds an [`UpgradingBuffer::Integer`] to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_integer_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_integer());
    }

    /// Adds an [`UpgradingBuffer::Float`] to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_float_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_float());
    }

    /// Adds an [`UpgradingBuffer::String`] to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_string_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_string());
    }

    /// Adds an [`UpgradingBuffer::Mixed`] to the cycle.
    ///
    /// Using this method after pushing values potentially breaks the cycle.
    fn add_mixed_buffer(&mut self) {
        self.buffers.push(UpgradingBuffer::new_mixed());
    }

    /// Appends an `i64` to the current [`UpgradingBuffer`] in the cycle and
    /// advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_i64(&mut self, value: i64) {
        self.current_mut().push_i64(value);
        self.advance();
    }

    /// Appends an `f64` to the current [`UpgradingBuffer`] in the cycle and
    /// advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_f64(&mut self, value: f64) {
        self.current_mut().push_f64(value);
        self.advance();
    }

    /// Appends a `String` to the current [`UpgradingBuffer`] in the cycle and
    /// advances to the next.
    ///
    /// # Panics
    ///
    /// Panics if no buffers have been added.
    fn push_string<T: Into<String>>(&mut self, value: T) {
        self.current_mut().push_string(value);
        self.advance();
    }
}

/// Builder pattern for [`TabulatedBlock`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlockBuilder {
    exit: ChildParserExit,
    title: String,
    errors: Vec<Error>,
}
