use crate::jcampdx::Value;

/// Column in a JCAMP-DX data table.
#[derive(Clone, PartialEq, Debug)]
pub enum Column {
    /// Only Integer values ([`i64`]).
    Integer(RawColumn<i64>),
    /// Only Float values ([`f64`]).
    Float(RawColumn<f64>),
    /// Only String values ([`String`]).
    String(RawColumn<String>),
    /// Potentially mixed values.
    Mixed(RawColumn<Value>),
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
