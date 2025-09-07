use uom::si::f64::Ratio;

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
    serde(into = "String", from = "String", rename_all = "camelCase")
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
            "INT" | "INTERNAL" => Self::Internal,
            "EXT" | "EXTERNAL" => Self::External,

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

impl From<ReferencingMethod> for String {
    fn from(value: ReferencingMethod) -> Self {
        value.to_string()
    }
}

/// Represents a reference used for calibrating chemical shifts in the NMR
/// spectrum.
///
/// # Construction
///
/// `ShiftReference` implements [`From<T>`] to allow for easy construction from
/// just a chemical shift, an index, or both, whenever metadata is not needed.
/// The following conversions are supported:
///
/// - [`From<Ratio>`]: first point in the [`Spectrum`], shift set to the
///   provided value.
/// - [`From<usize>`]: provided point in the [`Spectrum`], shift set to 0.
/// - [`From<(Ratio, usize)>`] and [`From<(usize, Ratio)>`]: provided shift and
///   index.
///
/// [`From<(Ratio, usize)>`]: From
/// [`From<(usize, Ratio)>`]: From
/// [`Spectrum`]: crate::spectrum::Spectrum
///
/// ## Example
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use uom::si::f64::Ratio;
/// use uom::si::ratio::part_per_million as ppm;
/// use zeenmr_spectrum::ShiftReference;
///
/// // first data point is the reference at 10 ppm
/// let left = ShiftReference::from(Ratio::new::<ppm>(10.0));
/// assert_approx_eq!(f64, left.shift().get::<ppm>(), 10.0);
/// assert_eq!(left.index(), 0);
/// assert!(left.name().is_none());
/// assert!(left.method().is_none());
///
/// // point 24576 is the reference at 0 ppm
/// let tms = ShiftReference::from(24576_usize);
/// assert_approx_eq!(f64, tms.shift().get::<ppm>(), 0.0);
/// assert_eq!(tms.index(), 24576);
/// assert!(tms.name().is_none());
/// assert!(tms.method().is_none());
///
/// // point 8192 is the reference at 4.8 ppm
/// let water = ShiftReference::from((Ratio::new::<ppm>(4.8), 8192));
/// assert_approx_eq!(f64, water.shift().get::<ppm>(), 4.8);
/// assert_eq!(water.index(), 8192);
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
    /// Chemical shift of the reference.
    #[cfg_attr(feature = "serde", serde(with = "serialize_shift"))]
    shift: Ratio,
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

// There is probably a better way within uom, but I couldn't find it.
#[cfg(feature = "serde")]
mod serialize_shift {
    use serde::{Deserialize, Deserializer, Serializer};
    use uom::si::f64::Ratio;
    use uom::si::ratio::part_per_million as ppm;

    pub(crate) fn serialize<S>(value: &Ratio, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_f64(value.get::<ppm>())
    }

    pub(crate) fn deserialize<'de, D>(deserializer: D) -> Result<Ratio, D::Error>
    where
        D: Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(Ratio::new::<ppm>)
    }
}

impl From<Ratio> for ShiftReference {
    fn from(value: Ratio) -> Self {
        Self {
            shift: value,
            ..Default::default()
        }
    }
}

impl From<usize> for ShiftReference {
    fn from(value: usize) -> Self {
        Self {
            index: value,
            ..Default::default()
        }
    }
}

impl From<(Ratio, usize)> for ShiftReference {
    fn from(value: (Ratio, usize)) -> Self {
        Self {
            shift: value.0,
            index: value.1,
            ..Default::default()
        }
    }
}

impl From<(usize, Ratio)> for ShiftReference {
    fn from(value: (usize, Ratio)) -> Self {
        (value.1, value.0).into()
    }
}

impl ShiftReference {
    /// Constructs a new [`ShiftReference`].
    ///
    /// Equivalent to the [`From<(Ratio, usize)>`] and [`From<(usize, Ratio)>`]
    /// implementations.
    ///
    /// [`From<(Ratio, usize)>`]: From
    /// [`From<(usize, Ratio)>`]: From
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let reference = ShiftReference::new(Ratio::new::<ppm>(4.8), 8192);
    /// assert_approx_eq!(f64, reference.shift().get::<ppm>(), 4.8);
    /// assert_eq!(reference.index(), 8192);
    /// assert_eq!(reference.name(), None);
    /// assert_eq!(reference.method(), None);
    /// ```
    pub fn new(shift: Ratio, index: usize) -> Self {
        Self {
            shift,
            index,
            name: None,
            method: None,
        }
    }

    /// Constructs a new `ShiftReference` with a name.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let reference = ShiftReference::new_with_name(Ratio::new::<ppm>(4.8), 8192, "H2O");
    /// assert_approx_eq!(f64, reference.shift().get::<ppm>(), 4.8);
    /// assert_eq!(reference.index(), 8192);
    /// assert_eq!(reference.name(), Some("H2O"));
    /// assert_eq!(reference.method(), None);
    /// ```
    pub fn new_with_name<T>(shift: Ratio, index: usize, name: T) -> Self
    where
        T: Into<String>,
    {
        Self {
            shift,
            index,
            name: Some(name.into()),
            method: None,
        }
    }

    /// Constructs a new `ShiftReference` with metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let reference = ShiftReference::new_with_meta(Ratio::new::<ppm>(4.8), 8192, "H2O", "internal");
    /// assert_approx_eq!(f64, reference.shift().get::<ppm>(), 4.8);
    /// assert_eq!(reference.index(), 8192);
    /// assert_eq!(reference.name(), Some("H2O"));
    /// assert_eq!(reference.method(), Some(&ReferencingMethod::Internal));
    /// ```
    pub fn new_with_meta<T, U>(shift: Ratio, index: usize, name: T, method: U) -> Self
    where
        T: Into<String>,
        U: Into<ReferencingMethod>,
    {
        Self {
            shift,
            index,
            name: Some(name.into()),
            method: Some(method.into()),
        }
    }

    /// Returns the chemical shift of the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// assert_approx_eq!(f64, reference.shift().get::<ppm>(), 10.0);
    /// ```
    pub fn shift(&self) -> Ratio {
        self.shift
    }

    /// Returns the position of the reference in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
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
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
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
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
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
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// reference.set_shift(Ratio::new::<ppm>(9.5));
    /// assert_approx_eq!(f64, reference.shift().get::<ppm>(), 9.5);
    /// ```
    pub fn set_shift(&mut self, shift: Ratio) {
        self.shift = shift;
    }

    /// Sets a new index of the reference in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
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
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// reference.set_name("TMS");
    /// assert_eq!(reference.name(), Some("TMS"));
    /// ```
    pub fn set_name<T: Into<String>>(&mut self, name: T) {
        self.name = Some(name.into());
    }

    /// Clears the name of the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::ShiftReference;
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// reference.set_name("TMS");
    /// reference.clear_name();
    /// assert!(reference.name().is_none());
    /// ```
    pub fn clear_name(&mut self) {
        self.name = None;
    }

    /// Sets the referencing method used in the NMR experiment.
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// reference.set_method(ReferencingMethod::Internal);
    /// assert_eq!(reference.method(), Some(&ReferencingMethod::Internal));
    /// ```
    pub fn set_method<T: Into<ReferencingMethod>>(&mut self, method: T) {
        self.method = Some(method.into());
    }

    /// Clears the referencing method of the reference.
    ///
    /// # Example
    ///
    /// ```
    /// use uom::si::f64::Ratio;
    /// use uom::si::ratio::part_per_million as ppm;
    /// use zeenmr_spectrum::{ReferencingMethod, ShiftReference};
    ///
    /// let mut reference = ShiftReference::from(Ratio::new::<ppm>(10.0));
    /// reference.set_method(ReferencingMethod::Internal);
    /// reference.clear_method();
    /// assert!(reference.method().is_none());
    /// ```
    pub fn clear_method(&mut self) {
        self.method = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "serde")]
    use float_cmp::assert_approx_eq;
    use num_traits::Zero;
    use static_assertions::assert_impl_all;
    use uom::si::ratio::part_per_million as ppm;

    #[test]
    fn thread_safety() {
        assert_impl_all!(ReferencingMethod: Send, Sync);
        assert_impl_all!(ShiftReference: Send, Sync);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let references = [
            Ratio::new::<ppm>(14.0).into(),
            (Ratio::new::<ppm>(4.8), 2_usize.pow(13)).into(),
            ShiftReference::new_with_meta(Ratio::zero(), 12000, "TMS", ReferencingMethod::Internal),
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
                assert_approx_eq!(f64, init.shift().get::<ppm>(), rec.shift().get::<ppm>());
                assert_eq!(init.index(), rec.index());
                assert_eq!(init.name(), rec.name());
                assert_eq!(init.method(), rec.method());
            })
    }

    #[cfg(feature = "serde")]
    #[test]
    fn deserialization_missing_fields() {
        let serialized = [
            "{\"shift\": 14.0, \"index\": 0}",
            "{\"shift\": 4.8, \"index\": 16384, \"name\": \"H2O\"}",
            "{\"shift\": 0.0, \"index\": 12000, \"method\": \"internal\"}",
        ];
        let expected = [
            Ratio::new::<ppm>(14_f64).into(),
            ShiftReference {
                shift: Ratio::new::<ppm>(4.8),
                index: 2_usize.pow(14),
                name: Some("H2O".into()),
                method: None,
            },
            ShiftReference {
                shift: Ratio::zero(),
                index: 12000,
                name: None,
                method: Some(ReferencingMethod::Internal),
            },
        ];
        let deserialized =
            serialized.map(|reference| serde_json::from_str::<ShiftReference>(reference).unwrap());
        expected
            .into_iter()
            .zip(deserialized)
            .for_each(|(init, rec)| {
                assert_approx_eq!(f64, init.shift().get::<ppm>(), rec.shift().get::<ppm>());
                assert_eq!(init.index(), rec.index());
                assert_eq!(init.name(), rec.name());
                assert_eq!(init.method(), rec.method());
            });
    }
}
