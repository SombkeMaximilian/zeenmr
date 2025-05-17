#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Representing the nucleus observed in an NMR experiment.
///
/// # Conversion
///
/// `Nucleus` implements [`From<T: AsRef<str>>`] and [`FromStr`] to allow
/// for easy conversion from string representations of nuclei. Whitespace,
/// dashes, underscores and carets are removed from the input string before
/// case-insensitively converting from a string to a `Nucleus` enum variant.
/// Note that there aren't any checks for the validity of a nucleus, so any
/// string can be converted to a `Nucleus::Other` variant, even if it doesn't
/// correspond to a real nucleus. As such, [`from_str`] will never fail.
///
/// [`From<T: AsRef<str>>`]: From
/// [`FromStr`]: std::str::FromStr
/// [`from_str`]: std::str::FromStr::from_str
///
/// ## Example
///
/// ```
/// use metabodecon::spectrum::meta::Nucleus;
///
/// let proton = Nucleus::from("proton");
/// let carbon13 = "Carbon-13".parse::<Nucleus>();
/// let nitrogen15 = Nucleus::from("   15n  ");
///
/// assert_eq!(proton, Nucleus::Hydrogen1);
/// assert_eq!(carbon13, Ok(Nucleus::Carbon13));
/// assert_eq!(nitrogen15, Nucleus::Nitrogen15);
/// ```
///
/// # Display
///
/// The `Display` implementation formats standard nuclei using IUPAC notation
/// (e.g., `1H`, `13C`). Other nuclei are displayed as-is. Therefore, it may be
/// desirable to use IUPAC notation when creating `Nucleus::Other` variants
/// in order to ensure consistent formatting.
///
/// ## Example
///
/// ```
/// use metabodecon::spectrum::meta::Nucleus;
///
/// let fluorine19 = Nucleus::Fluorine19;
/// let deuterium = Nucleus::Other("Deuterium".to_string());
/// let lead207 = Nucleus::Other("207Pb".to_string());
///
/// assert_eq!(fluorine19.to_string(), "19F");
/// assert_eq!(deuterium.to_string(), "Deuterium");
/// assert_eq!(lead207.to_string(), "207Pb");
/// ```
///
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// If the `serde` feature is enabled, `Nucleus` implements [`Serialize`] and
/// [`Deserialize`]. Serialization always uses the IUPAC atomic element (AE)
/// notation, while deserialization accepts the full name as well (e.g. Proton,
/// Carbon-13).
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Clone, Debug, Eq, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "String", into = "String",)
)]
pub enum Nucleus {
    /// Proton NMR.
    #[default]
    Hydrogen1,
    /// Boron-11 NMR.
    Boron11,
    /// Carbon-13 NMR.
    Carbon13,
    /// Nitrogen-15 NMR.
    Nitrogen15,
    /// Fluorine-19 NMR.
    Fluorine19,
    /// Silicon-29 NMR.
    Silicon29,
    /// Phosphorus-31 NMR.
    Phosphorus31,
    /// Other NMR nuclei.
    Other(String),
}

impl<T: AsRef<str>> From<T> for Nucleus {
    fn from(value: T) -> Self {
        let nucleus = value
            .as_ref()
            .trim()
            .replace(" ", "")
            .replace("^", "")
            .replace("-", "")
            .replace("_", "")
            .as_str()
            .to_uppercase();

        match nucleus.as_str() {
            "1H" | "PROTON" | "HYDROGEN1" => Self::Hydrogen1,
            "11B" | "BORON11" => Self::Boron11,
            "13C" | "CARBON13" => Self::Carbon13,
            "15N" | "NITROGEN15" => Self::Nitrogen15,
            "19F" | "FLUORINE19" => Self::Fluorine19,
            "29SI" | "SILICON29" => Self::Silicon29,
            "31P" | "PHOSPHORUS31" => Self::Phosphorus31,
            _ => Self::Other(value.as_ref().to_string()),
        }
    }
}

impl std::str::FromStr for Nucleus {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(s.into())
    }
}

impl std::fmt::Display for Nucleus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nucleus = match self {
            Self::Hydrogen1 => "1H",
            Self::Boron11 => "11B",
            Self::Carbon13 => "13C",
            Self::Nitrogen15 => "15N",
            Self::Fluorine19 => "19F",
            Self::Silicon29 => "29Si",
            Self::Phosphorus31 => "31P",
            Self::Other(value) => value.as_str(),
        };
        write!(f, "{}", nucleus)
    }
}

impl From<Nucleus> for String {
    fn from(value: Nucleus) -> Self {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(Nucleus);
        assert_sync!(Nucleus);
    }

    #[test]
    fn from_str() {
        let nuclei = [
            "1H".into(),
            "Proton".into(),
            "Hydrogen1".into(),
            "^1H".into(),
            "Boron-11".into(),
            "FluoRine_19".into(),
            "29Si".into(),
        ];
        let expected = [
            Nucleus::Hydrogen1,
            Nucleus::Hydrogen1,
            Nucleus::Hydrogen1,
            Nucleus::Hydrogen1,
            Nucleus::Boron11,
            Nucleus::Fluorine19,
            Nucleus::Silicon29,
        ];
        assert_eq!(expected, nuclei);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let nuclei = [
            Nucleus::Hydrogen1,
            Nucleus::Boron11,
            Nucleus::Carbon13,
            Nucleus::Nitrogen15,
            Nucleus::Fluorine19,
            Nucleus::Silicon29,
            Nucleus::Phosphorus31,
            Nucleus::Other("207Pb".to_string()),
        ];
        let serialized = nuclei
            .clone()
            .map(|nucleus| serde_json::to_string(&nucleus).unwrap());
        let deserialized = serialized
            .clone()
            .map(|serialized| serde_json::from_str::<Nucleus>(&serialized).unwrap());
        assert_eq!(nuclei, deserialized);
    }
}
