#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Represents the referencing method used in NMR experiments.
///
/// While the referencing method is not critical for correctly interpreting the
/// data, it can be useful, additional information. There are two variants:
/// - Internal: Reference is within the same sample.
/// - External: Reference is supplied from an external source.
///
/// # Conversion
///
/// `ReferencingMethod` implements [`FromStr`] to allow for easy conversion
/// from string representations of the variants. This conversion is
/// case-insensitive, meaning that any variation in capitalization (e.g.,
/// "internal", "Internal", or "INTERNAL") will correctly parse to the
/// respective variant.
///
/// [`FromStr`]: std::str::FromStr
///
/// ## Example
///
/// ```
/// use metabodecon::spectrum::meta::ReferencingMethod;
/// use std::str::FromStr;
///
/// let internal = "internal".parse::<ReferencingMethod>();
/// let external = ReferencingMethod::from_str("external");
/// let invalid = "extraterrestrial".parse::<ReferencingMethod>();
///
/// assert_eq!(internal, Ok(ReferencingMethod::Internal));
/// assert_eq!(external, Ok(ReferencingMethod::External));
/// assert!(invalid.is_err());
/// ```
///
/// # Display
///
/// The `Display` implementation for `ReferencingMethod` returns the lowercase
/// string representation of the variant.
///
/// ## Example
///
/// ```
/// use metabodecon::spectrum::meta::ReferencingMethod;
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
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
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
}

impl TryFrom<&str> for ReferencingMethod {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let method = value.trim().to_uppercase();

        match method.as_str() {
            "INTERNAL" => Ok(Self::Internal),
            "EXTERNAL" => Ok(Self::External),
            _ => Err(format!("invalid referencing method: {}", value)),
        }
    }
}

impl std::str::FromStr for ReferencingMethod {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.try_into()
    }
}

impl std::fmt::Display for ReferencingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let method = match self {
            Self::Internal => "internal",
            Self::External => "external",
        };
        write!(f, "{}", method)
    }
}

/// Represents a reference compound used for calibrating chemical shifts in an
/// NMR spectrum.
///
/// # Conversion
///
/// `ReferenceCompound` implements [`From<f64>`] and [`From<(f64, usize)>`] to
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
/// use metabodecon::spectrum::meta::ReferenceCompound;
///
/// let left = ReferenceCompound::from(10.0);
///
/// assert_approx_eq!(f64, left.chemical_shift(), 10.0);
/// assert_eq!(left.index(), 0);
/// assert!(left.name().is_none());
/// assert!(left.method().is_none());
///
/// let water = ReferenceCompound::from((4.8, 2_usize.pow(14) - 1));
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
/// If the `serde` feature is enabled, `ReferenceCompound` implements
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
pub struct ReferenceCompound {
    /// Chemical shift of the reference compound in parts per million (ppm).
    chemical_shift: f64,
    /// Index within the Spectrum that corresponds to the reference position.
    index: usize,
    /// Optional name for the reference compound.
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

impl From<f64> for ReferenceCompound {
    fn from(value: f64) -> Self {
        Self {
            chemical_shift: value,
            ..Default::default()
        }
    }
}

impl From<(f64, usize)> for ReferenceCompound {
    fn from(value: (f64, usize)) -> Self {
        Self {
            chemical_shift: value.0,
            index: value.1,
            ..Default::default()
        }
    }
}

impl ReferenceCompound {
    /// Constructs a new `ReferenceCompound` with all available metadata.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::spectrum::meta::{ReferenceCompound, ReferencingMethod};
    ///
    /// let reference = ReferenceCompound::new(
    ///     4.8,
    ///     2_usize.pow(14) - 1,
    ///     Some("H2O"),
    ///     Some(ReferencingMethod::Internal),
    /// );
    ///
    /// assert_approx_eq!(f64, reference.chemical_shift(), 4.8);
    /// assert_eq!(reference.index(), 2_usize.pow(14) - 1);
    /// assert_eq!(reference.name(), Some("H2O"));
    /// assert_eq!(reference.method(), Some(ReferencingMethod::Internal));
    /// ```
    pub fn new<T: Into<String>>(
        chemical_shift: f64,
        index: usize,
        name: Option<T>,
        method: Option<ReferencingMethod>,
    ) -> Self {
        Self {
            chemical_shift,
            index,
            name: name.map(|name| name.into()),
            method,
        }
    }

    /// Returns the chemical shift of the reference compound.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let reference = ReferenceCompound::from(10.0);
    ///
    /// assert_approx_eq!(f64, reference.chemical_shift(), 10.0);
    /// ```
    pub fn chemical_shift(&self) -> f64 {
        self.chemical_shift
    }

    /// Returns the position of the reference compound in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let reference = ReferenceCompound::from(10.0);
    ///
    /// assert_eq!(reference.index(), 0);
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the name of the reference compound, if available.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let reference = ReferenceCompound::from(10.0);
    ///
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
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let compound = ReferenceCompound::from(10.0);
    ///
    /// assert!(compound.method().is_none());
    /// ```
    pub fn method(&self) -> Option<ReferencingMethod> {
        self.method
    }

    /// Sets a new chemical shift for the reference compound.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let mut compound = ReferenceCompound::from(10.0);
    /// compound.set_chemical_shift(9.5);
    ///
    /// assert_approx_eq!(f64, compound.chemical_shift(), 9.5);
    /// ```
    pub fn set_chemical_shift(&mut self, chemical_shift: f64) {
        self.chemical_shift = chemical_shift;
    }

    /// Sets a new position of the reference compound in the [`Spectrum`].
    ///
    /// [`Spectrum`]: crate::spectrum::Spectrum
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let mut compound = ReferenceCompound::from(10.0);
    /// compound.set_index(5);
    ///
    /// assert_eq!(compound.index(), 5);
    /// ```
    pub fn set_index(&mut self, index: usize) {
        self.index = index;
    }

    /// Sets the name of the reference compound.
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::meta::ReferenceCompound;
    ///
    /// let mut reference = ReferenceCompound::from(10.0);
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
    /// use metabodecon::spectrum::meta::{ReferenceCompound, ReferencingMethod};
    ///
    /// let mut reference = ReferenceCompound::from(10.0);
    /// reference.set_method(Some(ReferencingMethod::Internal));
    ///
    /// assert_eq!(reference.method(), Some(ReferencingMethod::Internal));
    /// ```
    pub fn set_method(&mut self, method: Option<ReferencingMethod>) {
        self.method = method;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};
    use float_cmp::assert_approx_eq;

    #[test]
    fn thread_safety() {
        assert_send!(ReferencingMethod, ReferenceCompound);
        assert_sync!(ReferencingMethod, ReferenceCompound);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let references = [
            14_f64.into(),
            (4.8, 2_usize.pow(14)).into(),
            ReferenceCompound::new(0.0, 12000, Some("TMS"), Some(ReferencingMethod::Internal)),
        ];
        let serialized = references
            .clone()
            .map(|reference| serde_json::to_string(&reference).unwrap());
        let deserialized = serialized
            .clone()
            .map(|serialized| serde_json::from_str::<ReferenceCompound>(&serialized).unwrap());
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
            ReferenceCompound::new(4.8, 2_usize.pow(14), Some("H2O"), None),
            ReferenceCompound::new::<&str>(0.0, 12000, None, Some(ReferencingMethod::Internal)),
        ];
        let deserialized = serialized
            .map(|reference| serde_json::from_str::<ReferenceCompound>(reference).unwrap());
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
