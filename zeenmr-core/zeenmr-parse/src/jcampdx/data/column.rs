use crate::jcampdx::Value;

/// Column in a JCAMP-DX data table.
#[derive(Clone, PartialEq, Debug)]
pub enum Column {
    /// Only Integers.
    Integer(RawColumn<i64>),
    /// Only Floats.
    Float(RawColumn<f64>),
    /// Only Strings.
    String(RawColumn<String>),
    /// Potentially mixed values.
    Mixed(RawColumn<Value>),
}

impl From<RawColumn<i64>> for Column {
    fn from(value: RawColumn<i64>) -> Self {
        Self::Integer(value)
    }
}

impl From<RawColumn<f64>> for Column {
    fn from(value: RawColumn<f64>) -> Self {
        Self::Float(value)
    }
}

impl From<RawColumn<String>> for Column {
    fn from(value: RawColumn<String>) -> Self {
        Self::String(value)
    }
}

impl From<RawColumn<Value>> for Column {
    fn from(value: RawColumn<Value>) -> Self {
        Self::Mixed(value)
    }
}

impl Column {
    /// Returns the id of the inner [`RawColumn`].
    pub fn id(&self) -> &str {
        match self {
            Column::Integer(inner) => inner.id(),
            Column::Float(inner) => inner.id(),
            Column::String(inner) => inner.id(),
            Column::Mixed(inner) => inner.id(),
        }
    }

    /// Sets the id of the inner [`RawColumn`].
    pub fn set_id<T: Into<String>>(&mut self, id: T) {
        match self {
            Column::Integer(inner) => inner.id = id.into(),
            Column::Float(inner) => inner.id = id.into(),
            Column::String(inner) => inner.id = id.into(),
            Column::Mixed(inner) => inner.id = id.into(),
        }
    }

    /// Appends an `i64` to the column, converting the input and/or `Column` if
    /// necessary.
    ///
    /// # Conversion Rules
    ///
    /// | **Column**  | **Convert Input**  | **Convert Column** |
    /// | ----------- | ------------------ | ------------------ |
    /// | Integer     | No                 | No                 |
    /// | Float       | `f64`              | No                 |
    /// | String      | [`Value::Integer`] | [`Column::Mixed`]  |
    /// | Mixed       | [`Value::Integer`] | No                 |
    pub(crate) fn push_i64(&mut self, value: i64) {
        match self {
            Self::Integer(inner) => inner.values.push(value),
            Self::Float(inner) => inner.values.push(value as f64),
            Self::String(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Mixed(inner) => inner.values.push(value.into()),
        }
    }

    /// Appends `f64` to the column, converting the input and/or `Column` if
    /// necessary.
    ///
    /// # Conversion Rules
    ///
    /// | **Column**  | **Convert Input** | **Convert Column** |
    /// | ----------- | ----------------- | ------------------ |
    /// | Integer     | No                | [`Column::Float`]  |
    /// | Float       | No                | No                 |
    /// | String      | [`Value::Float`]  | [`Column::Mixed`]  |
    /// | Mixed       | [`Value::Float`]  | No                 |
    pub(crate) fn push_f64(&mut self, value: f64) {
        match self {
            Self::Integer(inner) => {
                *self = Self::convert_to_float_and_push(std::mem::take(inner), value);
            }
            Self::Float(inner) => inner.values.push(value),
            Self::String(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Mixed(inner) => inner.values.push(value.into()),
        }
    }

    /// Appends a `String` to the column, converting the input and/or `Column`
    /// if necessary.
    ///
    /// # Conversion Rules
    ///
    /// | **Column**  | **Convert Input** | **Convert Column** |
    /// | ----------- | ----------------- | ------------------ |
    /// | Integer     | No                | [`Column::Mixed`]  |
    /// | Float       | No                | [`Column::Mixed`]  |
    /// | String      | No                | No                 |
    /// | Mixed       | [`Value::String`] | No                 |
    pub(crate) fn push_string<T: Into<String>>(&mut self, value: T) {
        let value = value.into();

        match self {
            Self::Integer(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::Float(inner) => {
                *self = Self::convert_to_mixed_and_push(std::mem::take(inner), value);
            }
            Self::String(inner) => inner.values.push(value),
            Self::Mixed(inner) => inner.values.push(value.into()),
        }
    }

    /// Appends a `Value` to the column, converting the `Column` if necessary.
    ///
    /// # Conversion Rules
    ///
    /// | **Column**  | **Convert Input** | **Convert Column** |
    /// | ----------- | ----------------- | ------------------ |
    /// | Integer     | No                | [`Column::Mixed`]  |
    /// | Float       | No                | [`Column::Mixed`]  |
    /// | String      | No                | [`Column::Mixed`]  |
    /// | Mixed       | No                | No                 |
    pub(crate) fn push_value<T: Into<Value>>(&mut self, value: T) {
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
            Self::Mixed(inner) => inner.values.push(value.into()),
        }
    }

    /// Converts raw to `RawColumn<f64>`, pushes the value and returns a
    /// [`Column::Float`].
    fn convert_to_float_and_push(raw: RawColumn<i64>, value: f64) -> Self {
        let mut converted = raw.into_float_column();
        converted.values.push(value);

        Self::Float(converted)
    }

    /// Converts raw to `RawColumn<Value>`, pushes the value and returns a
    /// [`Column::Mixed`].
    fn convert_to_mixed_and_push<T, U>(raw: RawColumn<T>, value: U) -> Self
    where
        T: Into<Value>,
        U: Into<Value>,
    {
        let mut converted = raw.into_value_column();
        converted.values.push(value.into());

        Self::Mixed(converted)
    }
}

/// Underlying data structure for [`Column`].
///
/// Wrapper around [`Vec<T>`] that includes a column label as a [`String`].
#[derive(Clone, Eq, PartialEq, Debug, Default)]
pub struct RawColumn<T> {
    /// Identifier of the column.
    pub id: String,
    /// Raw data.
    pub values: Vec<T>,
}

impl<T> From<RawColumn<T>> for Vec<T> {
    fn from(value: RawColumn<T>) -> Self {
        value.values
    }
}

impl<T> AsRef<[T]> for RawColumn<T> {
    fn as_ref(&self) -> &[T] {
        &self.values
    }
}

impl<T> AsMut<[T]> for RawColumn<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

impl<T> std::ops::Deref for RawColumn<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T> std::ops::DerefMut for RawColumn<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<T> FromIterator<T> for RawColumn<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            id: String::new(),
            values: iter.into_iter().collect(),
        }
    }
}

impl<T> IntoIterator for RawColumn<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a RawColumn<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut RawColumn<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

impl<T> RawColumn<T>
where
    T: Into<Value>,
{
    /// Converts all elements in this column into [`Value`].
    pub(crate) fn into_value_column(self) -> RawColumn<Value> {
        RawColumn::<Value> {
            id: self.id,
            values: self.values.into_iter().map(Into::into).collect(),
        }
    }
}

impl RawColumn<i64> {
    /// Converts the `i64` in this column into `f64`.
    pub(crate) fn into_float_column(self) -> RawColumn<f64> {
        RawColumn::<f64> {
            id: self.id,
            values: self
                .values
                .into_iter()
                .map(|value| value as f64)
                .collect(),
        }
    }
}

impl<T> RawColumn<T> {
    /// Returns the id of the `RawColumn`.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns an [`Iterator`] over the slice.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.values.iter()
    }
}
