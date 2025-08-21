#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the referencing method used in NMR experiments.
///
/// While the referencing method is not critical for correctly interpreting the
/// data, it can be useful, additional information. There are two main variants:
/// - Internal: Reference is within the same sample.
/// - External: Reference is supplied from an external source.
///
/// # Conversion
///
/// `ReferencingMethod` implements [`FromStr`] to allow for easy conversion
/// from string representations of the variants. This conversion is
/// case-insensitive, meaning that any variation in capitalization (e.g.,
/// "internal", "InternAL", or "INTERNAL") will correctly parse to the
/// respective variant. Note that there aren't any checks for the validity of
/// the string representation, meaning any string can be converted to a
/// `ReferencingMethod::Other` variant. This is done to prevent data entry
/// errors, localization or similar issues from causing parsing failures of
/// otherwise valid data.
///
/// [`FromStr`]: std::str::FromStr
///
/// ## Example
///
/// ```
/// use std::str::FromStr;
/// use zeenmr_spectrum::ReferencingMethod;
///
/// let internal = "internal".parse::<ReferencingMethod>();
/// let external = ReferencingMethod::from_str("external");
/// let other = "extern".parse::<ReferencingMethod>();
///
/// assert_eq!(internal, Ok(ReferencingMethod::Internal));
/// assert_eq!(external, Ok(ReferencingMethod::External));
/// assert_eq!(other, Ok(ReferencingMethod::Other("extern".to_string())));
/// ```
///
/// # Display
///
/// The `Display` implementation for `ReferencingMethod` returns the lowercase
/// string representation for the two main variants, or the string value for the
/// `Other` variant.
///
/// ## Example
///
/// ```
/// use zeenmr_spectrum::ReferencingMethod;
///
/// let internal = ReferencingMethod::Internal.to_string();
/// let external = ReferencingMethod::External.to_string();
///
/// assert_eq!(internal, "internal");
/// assert_eq!(external, "external");
/// ```
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// If the `serde` feature is enabled, `ReferencingMethod` implements
/// [`Serialize`] and [`Deserialize`].
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Clone, Eq, PartialEq, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub enum ReferencingMethod {
    /// Reference was added to the sample.
    Internal,
    /// Reference was supplied from an external source.
    External,
    /// A fallback variant for referencing method descriptions that do not fit
    /// into the known categories, and data entry errors. Useful for handling
    /// unexpected or custom values.
    Other(String),
}

impl<T: AsRef<str>> From<T> for ReferencingMethod {
    fn from(value: T) -> Self {
        let method = value.as_ref().trim().to_uppercase();

        match method.as_str() {
            "INTERNAL" => Self::Internal,
            "EXTERNAL" => Self::External,

            // Graceful fallback for unrecognized referencing methods.
            _ => Self::Other(value.as_ref().to_string()),
        }
    }
}

impl std::str::FromStr for ReferencingMethod {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl std::fmt::Display for ReferencingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let method = match self {
            Self::Internal => "internal",
            Self::External => "external",

            // Graceful fallback for unrecognized referencing methods.
            Self::Other(value) => value.as_str(),
        };

        write!(f, "{method}")
    }
}

/// Represents a reference used for calibrating chemical shifts in the NMR
/// spectrum.
///
/// # Conversion
///
/// `ShiftReference` implements [`From<f64>`] and [`From<(f64, usize)>`] to
/// allow for easy conversion from a chemical shift or chemical shift and index
/// pair. In the former case, the index is set to 0, meaning that the leftmost
/// chemical shift in the [`Spectrum`] is the reference.
///
/// [`From<f64>`]: From
/// [`From<(f64, usize)>`]: From
/// [`Spectrum`]: crate::spectrum::Spectrum
///
/// ## Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use zeenmr_spectrum::ShiftReference;
///
/// let left = ShiftReference::from(10.0);
///
/// assert_approx_eq!(f64, left.chemical_shift(), 10.0);
/// assert_eq!(left.index(), 0);
/// assert!(left.name().is_none());
/// assert!(left.method().is_none());
///
/// let water = ShiftReference::from((4.8, 2_usize.pow(14) - 1));
///
/// assert_approx_eq!(f64, water.chemical_shift(), 4.8);
/// assert_eq!(water.index(), 2_usize.pow(14) - 1);
/// assert!(water.name().is_none());
/// assert!(water.method().is_none());
/// ```
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// If the `serde` feature is enabled, `ShiftReference` implements
/// [`Serialize`] and [`Deserialize`].
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Clone, Debug, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct ShiftReference {
    /// Chemical shift of the reference in parts per million (ppm).
    chemical_shift: f64,
    /// Index within the Spectrum that corresponds to the reference position.
    index: usize,
    /// Optional name for the reference.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    name: Option<String>,
    /// Optional information about the method used for referencing.
    #[cfg_attr(
        feature = "serde",
        serde(default, skip_serializing_if = "Option::is_none")
    )]
    method: Option<ReferencingMethod>,
}

impl From<f64> for ShiftReference {
    fn from(value: f64) -> Self {
        Self {
            chemical_shift: value,
            ..Default::default()
        }
    }
}

impl From<(f64, usize)> for ShiftReference {
    fn from(value: (f64, usize)) -> Self {
        Self {
            chemical_shift: value.0,
            index: value.1,
            ..Default::default()
        }
    }
}

impl ShiftReference {
    /// Constructs a new `ShiftReference` with all available metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let reference = ShiftReference::new(4.8, 2_usize.pow(14) - 1, Some("H2O"), Some("internal"));
    /// assert_approx_eq!(f64, reference.chemical_shift(), 4.8);
    /// assert_eq!(reference.index(), 2_usize.pow(14) - 1);
    /// assert_eq!(reference.name(), Some("H2O"));
    /// assert_eq!(reference.method(), Some(&ReferencingMethod::Internal));
    /// ```
    pub fn new<T, U>(chemical_shift: f64, index: usize, name: Option<T>, method: Option<U>) -> Self
    where
        T: Into<String>,
        U: Into<ReferencingMethod>,
    {
        Self {
            chemical_shift,
            index,
            name: name.map(Into::into),
            method: method.map(Into::into),
        }
    }

    /// Returns the chemical shift of the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(10.0);
    /// assert_approx_eq!(f64, reference.chemical_shift(), 10.0);
    /// ```
    pub fn chemical_shift(&self) -> f64 {
        self.chemical_shift
    }

    /// Returns the position of the reference in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(10.0);
    /// assert_eq!(reference.index(), 0);
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the name of the reference, if available.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(10.0);
    /// assert!(reference.name().is_none());
    /// ```
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the referencing method, if available.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(10.0);
    /// assert!(reference.method().is_none());
    /// ```
    pub fn method(&self) -> Option<&ReferencingMethod> {
        self.method.as_ref()
    }

    /// Sets a new chemical shift for the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(10.0);
    /// reference.set_chemical_shift(9.5);
    /// assert_approx_eq!(f64, reference.chemical_shift(), 9.5);
    /// ```
    pub fn set_chemical_shift(&mut self, chemical_shift: f64) {
        self.chemical_shift = chemical_shift;
    }

    /// Sets a new index of the reference in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(10.0);
    /// reference.set_index(5);
    /// assert_eq!(reference.index(), 5);
    /// ```
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
    }

    /// Sets the name of the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(10.0);
    /// reference.set_name(Some("TMS"));
    ///
    /// assert_eq!(reference.name(), Some("TMS"));
    /// ```
    pub fn set_name<T: Into<String>>(&mut self, name: Option<T>) {
        self.name = name.map(|name| name.into());
    }

    /// Sets the referencing method used in the NMR experiment.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let mut reference = ShiftReference::from(10.0);
    /// reference.set_method(Some(ReferencingMethod::Internal));
    ///
    /// assert_eq!(reference.method(), Some(&ReferencingMethod::Internal));
    /// ```
    pub fn set_method(&mut self, method: Option<ReferencingMethod>) {
        self.method = method;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "serde")]
    use float_cmp::assert_approx_eq;
    use static_assertions::assert_impl_all;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ReferencingMethod: Send, Sync);
        assert_impl_all!(ShiftReference: Send, Sync);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let references = [
            14_f64.into(),
            (4.8, 2_usize.pow(14)).into(),
            ShiftReference::new(0.0, 12000, Some("TMS"), Some(ReferencingMethod::Internal)),
        ];
        let serialized = references
            .clone()
            .map(|reference| serde_json::to_string(&reference).unwrap());
        let deserialized = serialized
            .clone()
            .map(|serialized| serde_json::from_str::<ShiftReference>(&serialized).unwrap());
        references
            .into_iter()
            .zip(deserialized)
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, init.chemical_shift(), rec.chemical_shift());
                assert_eq!(init.index(), rec.index());
                assert_eq!(init.name(), rec.name());
                assert_eq!(init.method(), rec.method());
            })
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_missing_fields() {
        let serialized = [
            "{\"chemicalShift\": 14.0, \"index\": 0}",
            "{\"chemicalShift\": 4.8, \"index\": 16384, \"name\": \"H2O\"}",
            "{\"chemicalShift\": 0.0, \"index\": 12000, \"method\": \"internal\"}",
        ];
        let expected = [
            14_f64.into(),
            ShiftReference::new(4.8, 2_usize.pow(14), Some("H2O"), None::<&str>),
            ShiftReference::new(0.0, 12000, None::<&str>, Some(ReferencingMethod::Internal)),
        ];
        let deserialized =
            serialized.map(|reference| serde_json::from_str::<ShiftReference>(reference).unwrap());
        expected
            .into_iter()
            .zip(deserialized)
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, init.chemical_shift(), rec.chemical_shift());
                assert_eq!(init.index(), rec.index());
                assert_eq!(init.name(), rec.name());
                assert_eq!(init.method(), rec.method());
            });
    }
}
