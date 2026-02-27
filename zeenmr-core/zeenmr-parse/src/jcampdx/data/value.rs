use std::borrow::Cow;

/// Values in JCAMP-DX files.
#[derive(Clone, PartialEq, Debug, Default)]
pub enum Value {
    /// Empty values for unset parameters.
    #[default]
    Empty,
    /// Integer values.
    Integer(i64),
    /// Float values.
    Float(f64),
    /// Everything else is treated as a string.
    String(String),
    /// Arrays of values.
    Array(Vec<Self>),
}

impl From<()> for Value {
    fn from(_: ()) -> Self {
        Self::Empty
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}

impl From<&mut str> for Value {
    fn from(value: &mut str) -> Self {
        Self::String(value.into())
    }
}

impl<'a> From<Cow<'a, str>> for Value {
    fn from(value: Cow<'a, str>) -> Self {
        Self::String(value.into())
    }
}

impl<T> From<Vec<T>> for Value
where
    T: Into<Value>,
{
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<T, const N: usize> From<[T; N]> for Value
where
    T: Into<Value>,
{
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<T> From<&[T]> for Value
where
    T: Clone + Into<Value>,
{
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().cloned().map(Into::into).collect())
    }
}

impl<T> FromIterator<T> for Value
where
    T: Into<Value>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::Array(iter.into_iter().map(Into::into).collect())
    }
}

impl Value {
    /// Returns `true` if in the `Empty` variant.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Empty => true,
            _ => false,
        }
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

    /// Returns `true` if in the `Array` variant.
    pub fn is_array(&self) -> bool {
        match self {
            Self::Array(_) => true,
            _ => false,
        }
    }

    /// Returns the inner `i64`, or `None` if not in the `Integer` variant.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns the inner `f64`, or `None` if not in the `Float` variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns a string slice of the inner `String`, or `None` if not in the
    /// `String` variant.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Returns a slice of the inner `Vec<Value>`, or `None` if not in the
    /// `Array` variant.
    pub fn as_slice(&self) -> Option<&[Value]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}
