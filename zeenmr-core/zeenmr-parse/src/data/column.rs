use crate::data::Value;
use std::borrow::Cow;

/// Column in a JCAMP-DX data table.
#[derive(Clone, PartialEq, Debug)]
pub enum Column<'source> {
    /// Only Integers.
    Integer(Vec<i64>),
    /// Only Floats.
    Float(Vec<f64>),
    /// Only Strings.
    String(Vec<Cow<'source, str>>),
    /// Potentially mixed values.
    Mixed(Vec<Value<'source>>),
}

impl From<Vec<i64>> for Column<'_> {
    fn from(value: Vec<i64>) -> Self {
        Self::Integer(value)
    }
}

impl From<Vec<f64>> for Column<'_> {
    fn from(value: Vec<f64>) -> Self {
        Self::Float(value)
    }
}

impl<'source> From<Vec<Cow<'source, str>>> for Column<'source> {
    fn from(value: Vec<Cow<'source, str>>) -> Self {
        Self::String(value)
    }
}

impl<'source> From<Vec<Value<'source>>> for Column<'source> {
    fn from(value: Vec<Value<'source>>) -> Self {
        Self::Mixed(value)
    }
}

impl FromIterator<i64> for Column<'_> {
    fn from_iter<T: IntoIterator<Item = i64>>(iter: T) -> Self {
        Self::Integer(iter.into_iter().collect())
    }
}

impl FromIterator<f64> for Column<'_> {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        Self::Float(iter.into_iter().collect())
    }
}

impl FromIterator<String> for Column<'_> {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        Self::String(iter.into_iter().map(Into::into).collect())
    }
}

impl<'source> FromIterator<&'source str> for Column<'source> {
    fn from_iter<T: IntoIterator<Item = &'source str>>(iter: T) -> Self {
        Self::String(iter.into_iter().map(Into::into).collect())
    }
}

impl<'source> FromIterator<Cow<'source, str>> for Column<'source> {
    fn from_iter<T: IntoIterator<Item = Cow<'source, str>>>(iter: T) -> Self {
        Self::String(iter.into_iter().collect())
    }
}

impl<'source> Column<'source> {
    /// Returns the number of elements in the column.
    pub fn len(&self) -> usize {
        match self {
            Self::Integer(inner) => inner.len(),
            Self::Float(inner) => inner.len(),
            Self::String(inner) => inner.len(),
            Self::Mixed(inner) => inner.len(),
        }
    }

    /// Returns `true` if the column contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if in the `Integer` variant.
    pub fn is_integer(&self) -> bool {
        match self {
            Self::Integer(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if in the `Float` variant.
    pub fn is_float(&self) -> bool {
        match self {
            Self::Float(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if in the `String` variant.
    pub fn is_string(&self) -> bool {
        match self {
            Self::String(_) => true,
            _ => false,
        }
    }

    /// Returns `true` if in the `Mixed` variant.
    pub fn is_mixed(&self) -> bool {
        match self {
            Self::Mixed(_) => true,
            _ => false,
        }
    }

    /// Converts this column into an owned form with a `'static` lifetime.
    pub fn into_owned(self) -> Column<'static> {
        match self {
            Self::Integer(inner) => Column::Integer(inner),
            Self::Float(inner) => Column::Float(inner),
            Self::String(inner) => Column::String(
                inner
                    .into_iter()
                    .map(|s| Cow::Owned(s.into_owned()))
                    .collect(),
            ),
            Self::Mixed(inner) => Column::Mixed(inner.into_iter().map(Value::into_owned).collect()),
        }
    }
}
