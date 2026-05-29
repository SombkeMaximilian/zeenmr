use std::borrow::Cow;

/// Heterogeneous value parsed from spectroscopy data files.
#[derive(Clone, PartialEq, Debug, Default)]
pub enum Value<'source> {
    /// Represents an unset parameter.
    #[default]
    Empty,
    /// Integer value.
    Integer(i64),
    /// IEEE float value.
    Float(f64),
    /// String value.
    String(Cow<'source, str>),
    /// Arrays of values.
    Array(Vec<Self>),
}

impl From<()> for Value<'_> {
    fn from(_: ()) -> Self {
        Self::Empty
    }
}

impl From<i64> for Value<'_> {
    fn from(value: i64) -> Self {
        Self::Integer(value)
    }
}

impl From<f64> for Value<'_> {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<String> for Value<'_> {
    fn from(value: String) -> Self {
        Self::String(value.into())
    }
}

impl<'source> From<&'source str> for Value<'source> {
    fn from(value: &'source str) -> Self {
        Self::String(value.into())
    }
}

impl<'source> From<Cow<'source, str>> for Value<'source> {
    fn from(value: Cow<'source, str>) -> Self {
        Self::String(value)
    }
}

impl<'source, T> From<Vec<T>> for Value<'source>
where
    T: Into<Value<'source>>,
{
    fn from(value: Vec<T>) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<'source, T, const N: usize> From<[T; N]> for Value<'source>
where
    T: Into<Value<'source>>,
{
    fn from(value: [T; N]) -> Self {
        Self::Array(value.into_iter().map(Into::into).collect())
    }
}

impl<'source, T> From<&[T]> for Value<'source>
where
    T: Clone + Into<Value<'source>>,
{
    fn from(value: &[T]) -> Self {
        Self::Array(value.iter().cloned().map(Into::into).collect())
    }
}

impl<'source, T> FromIterator<T> for Value<'source>
where
    T: Into<Value<'source>>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::Array(iter.into_iter().map(Into::into).collect())
    }
}

impl<'source> Value<'source> {
    /// Returns `true` if the value is `Empty`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::Empty;
    ///
    /// assert!(value.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Empty)
    }

    /// Returns `true` if the value is an `Integer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::Integer(42);
    ///
    /// assert!(value.is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Integer(_))
    }

    /// Returns the contained `i64`, or `None` if the value is not an `Integer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let integer_value = Value::Integer(42);
    /// let string_value = Value::from("not an integer");
    ///
    /// assert_eq!(integer_value.as_i64(), Some(42));
    /// assert!(string_value.as_i64().is_none());
    /// ```
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns `true` if the value is a `Float`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::Float(69.420);
    ///
    /// assert!(value.is_float());
    /// ```
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float(_))
    }

    /// Returns the contained `f64`, or `None` if the value is not a `Float`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let float_value = Value::Float(69.420);
    /// let string_value = Value::from("not a float");
    ///
    /// assert_eq!(float_value.as_f64(), Some(69.420));
    /// assert!(string_value.as_f64().is_none());
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    /// Returns `true` if the value is a `String` (note: not the stdlib type).
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::from("Hello, world!");
    ///
    /// assert!(value.is_string());
    /// ```
    pub fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    /// Returns a string slice containing the string, or `None` if the value is
    /// not a `String`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let string_value = Value::from("Hello, world!");
    /// let integer_value = Value::Integer(42);
    ///
    /// assert_eq!(string_value.as_str(), Some("Hello, world!"));
    /// assert!(integer_value.as_str().is_none());
    /// ```
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value.as_ref()),
            _ => None,
        }
    }

    /// Returns `true` if the value is an `Array`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::from([
    ///     Value::Integer(42),
    ///     Value::Float(69.420),
    ///     Value::from("Hello, world!"),
    /// ]);
    ///
    /// assert!(value.is_array());
    /// ```
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Returns a slice of the contained `Vec<Value>`, or `None` if the value is
    /// not an `Array`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let array_value = Value::from([Value::Integer(42)]);
    /// let string_value = Value::from("Hello, world!");
    ///
    /// assert_eq!(array_value.as_slice(), Some([Value::Integer(42)].as_ref()));
    /// assert!(string_value.as_slice().is_none());
    /// ```
    pub fn as_slice(&self) -> Option<&[Self]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Returns a mutable slice of the contained `Vec<Value>`, or `None` if the
    /// value is not an `Array`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_parse::data::Value;
    ///
    /// let mut array_value = Value::from([Value::Integer(42)]);
    ///
    /// if let Some(slice) = array_value.as_mut_slice() {
    ///     slice[0] = Value::Float(42.0)
    /// }
    ///
    /// assert_eq!(array_value.as_slice(), Some([Value::Float(42.0)].as_ref()));
    /// ```
    pub fn as_mut_slice(&mut self) -> Option<&mut [Self]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }

    /// Converts this `Value` into an owned form with a `'static` lifetime.
    ///
    /// This is useful when you need to store a parsed `Value` beyond the
    /// lifetime of the input buffer. Borrowed string data is cloned into
    /// `Cow::Owned`s. Arrays are converted recursively. All other variants are
    /// moved.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::borrow::Cow;
    /// use zeenmr_parse::data::Value;
    ///
    /// let value = Value::from("nmr");
    /// let owned = value.into_owned();
    ///
    /// assert!(matches!(owned, Value::String(Cow::Owned(_))));
    /// assert_eq!(owned.as_str(), Some("nmr"));
    /// ```
    pub fn into_owned(self) -> Value<'static> {
        match self {
            Self::Empty => Value::Empty,
            Self::Integer(value) => Value::Integer(value),
            Self::Float(value) => Value::Float(value),
            Self::String(value) => Value::String(Cow::Owned(value.into_owned())),
            Self::Array(value) => Value::Array(value.into_iter().map(Value::into_owned).collect()),
        }
    }
}
