use uom::si::action::joule_second;
use uom::si::f64::{Action, GyromagneticRatio, MagneticMoment, Ratio};
use uom::si::gyromagnetic_ratio::megahertz_per_tesla;

#[cfg(test)]
use strum::EnumIter;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Planck constant
const PLANCK: f64 = 6.626_070_15e-34;

/// Compile-time created spin (as [`Ratio`]).
macro_rules! spin {
    ($value:expr) => {
        ::uom::si::f64::Ratio::new::<::uom::si::ratio::ratio>($value)
    };
}

/// Compile-time created magnetic moment (as [`MagneticMoment`]).
macro_rules! mag_mom {
    ($value:expr) => {
        ::uom::si::f64::MagneticMoment::new::<::uom::si::magnetic_moment::nuclear_magneton>($value)
    };
}

/// Representing the nucleus observed in an NMR experiment.
///
/// # NMR Active Nuclei
///
/// The `Nucleus `enum includes all stable isotopes that are NMR active, meaning
/// they possess a non-zero nuclear spin and are thus detectable via nuclear
/// magnetic resonance spectroscopy (e.g., Deuterium, Carbon-13).
///
/// # Conversion
///
/// `Nucleus` implements [`From<T: AsRef<str>>`] and [`FromStr`] to allow
/// for easy conversion from string representations of nuclei. Whitespace,
/// dashes, underscores and carets are removed from the input string before
/// case-insensitively converting from a string to a `Nucleus` enum variant.
/// Note that there aren't any checks for the validity of a nucleus, so any
/// string can be converted to a `Nucleus::Other` variant, even if it doesn't
/// correspond to a real nucleus. This is done to prevent data entry errors
/// from invalidating otherwise valid data. As such, [`from_str`] will never
/// fail.
///
/// [`From<T: AsRef<str>>`]: From
/// [`FromStr`]: std::str::FromStr
/// [`from_str`]: std::str::FromStr::from_str
///
/// ## Example
///
/// ```
/// use zeenmr_spectrum::Nucleus;
///
/// let proton = Nucleus::from("proton");
/// let carbon13 = "Carbon-13".parse::<Nucleus>();
/// let nitrogen15 = Nucleus::from("   15n  ");
///
/// assert_eq!(proton, Nucleus::Hydrogen);
/// assert_eq!(carbon13, Ok(Nucleus::Carbon13));
/// assert_eq!(nitrogen15, Nucleus::Nitrogen15);
/// ```
///
/// # Display
///
/// The `Display` implementation formats standard nuclei using IUPAC atomic
/// element (AE) notation (e.g., `1H`, `13C`).
///
/// ## Example
///
/// ```
/// use zeenmr_spectrum::Nucleus;
///
/// let fluorine19 = Nucleus::Fluorine19;
/// let deuterium = Nucleus::Deuterium;
///
/// assert_eq!(fluorine19.to_string(), "19F");
/// assert_eq!(deuterium.to_string(), "2H");
/// ```
///
/// # Serialization with [Serde]
///
/// [Serde]: https://serde.rs/
///
/// If the `serde` feature is enabled, `Nucleus` implements [`Serialize`] and
/// [`Deserialize`]. Serialization always uses the IUPAC atomic element (AE)
/// notation, while deserialization accepts the full name as well (e.g., Proton,
/// Deuterium, Carbon-13).
///
/// [`Serialize`]: serde::Serialize
/// [`Deserialize`]: serde::Deserialize
#[derive(Clone, Debug, Eq, PartialEq, Default)]
#[cfg_attr(test, derive(EnumIter))]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(from = "String", into = "String",)
)]
#[allow(missing_docs)]
pub enum Nucleus {
    // Main group and transition elements
    #[default]
    Hydrogen,
    Deuterium,
    Tritium,
    Helium3,
    Lithium6,
    Lithium7,
    Beryllium9,
    Boron10,
    Boron11,
    Carbon13,
    Nitrogen14,
    Nitrogen15,
    Oxygen17,
    Fluorine19,
    Neon21,
    Sodium23,
    Magnesium25,
    Aluminium27,
    Silicon29,
    Phosphorus31,
    Sulphur33,
    Chlorine35,
    Chlorine37,
    Potassium39,
    Potassium41,
    Calcium43,
    Scandium45,
    Titanium47,
    Titanium49,
    Vanadium50,
    Vanadium51,
    Chromium53,
    Manganese55,
    Iron57,
    Cobalt59,
    Nickel61,
    Copper63,
    Copper65,
    Zinc67,
    Gallium69,
    Gallium71,
    Germanium73,
    Arsenic75,
    Selenium77,
    Bromine79,
    Bromine81,
    Krypton83,
    Rubidium85,
    Rubidium87,
    Strontium87,
    Yttrium89,
    Zirconium91,
    Niobium93,
    Molybdenum95,
    Molybdenum97,
    Technetium99,
    Ruthenium99,
    Ruthenium101,
    Rhodium103,
    Palladium105,
    Silver107,
    Silver109,
    Cadmium111,
    Cadmium113,
    Indium113,
    Indium115,
    Tin115,
    Tin117,
    Tin119,
    Antimony121,
    Antimony123,
    Tellurium123,
    Tellurium125,
    Iodine127,
    Xenon129,
    Xenon131,
    Caesium133,
    Barium135,
    Barium137,
    Lanthanum139,
    Hafnium177,
    Hafnium179,
    Tantalum181,
    Tungsten183,
    Rhenium185,
    Rhenium187,
    Osmium187,
    Osmium189,
    Iridium191,
    Iridium193,
    Platinum195,
    Gold197,
    Mercury199,
    Mercury201,
    Thallium203,
    Thallium205,
    Lead207,
    Bismuth209,

    // Lanthanides
    Praseodymium141,
    Neodymium143,
    Neodymium145,
    Samarium147,
    Samarium149,
    Europium151,
    Europium153,
    Gadolinium155,
    Gadolinium157,
    Terbium159,
    Dysprosium161,
    Dysprosium163,
    Holmium165,
    Erbium167,
    Thulium169,
    Ytterbium171,
    Ytterbium173,
    Lutetium175,
    Lutetium176,

    /// A fallback variant for nuclei that are not recognized in the standard
    /// list. This includes unrecognized names, incorrect input, or names
    /// provided in languages other than English. Useful for handling unexpected
    /// or localized input gracefully.
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
            // Main group and transition elements
            "1H" | "H" | "HYDROGEN" | "PROTON" => Self::Hydrogen,
            "2H" | "D" | "DEUTERIUM" => Self::Deuterium,
            "3H" | "T" | "TRITIUM" => Self::Tritium,
            "3HE" | "HELIUM3" => Self::Helium3,
            "6LI" | "LITHIUM6" => Self::Lithium6,
            "7LI" | "LITHIUM7" => Self::Lithium7,
            "9BE" | "BERYLLIUM9" => Self::Beryllium9,
            "10B" | "BORON10" => Self::Boron10,
            "11B" | "BORON11" => Self::Boron11,
            "13C" | "CARBON13" => Self::Carbon13,
            "14N" | "NITROGEN14" => Self::Nitrogen14,
            "15N" | "NITROGEN15" => Self::Nitrogen15,
            "17O" | "OXYGEN17" => Self::Oxygen17,
            "19F" | "FLUORINE19" => Self::Fluorine19,
            "21NE" | "NEON21" => Self::Neon21,
            "23NA" | "SODIUM23" => Self::Sodium23,
            "25MG" | "MAGNESIUM25" => Self::Magnesium25,
            "27AL" | "ALUMINIUM27" | "ALUMINUM27" => Self::Aluminium27,
            "29SI" | "SILICON29" => Self::Silicon29,
            "31P" | "PHOSPHORUS31" => Self::Phosphorus31,
            "33S" | "SULPHUR33" | "SULFUR33" => Self::Sulphur33,
            "35CL" | "CHLORINE35" => Self::Chlorine35,
            "37CL" | "CHLORINE37" => Self::Chlorine37,
            "39K" | "POTASSIUM39" => Self::Potassium39,
            "41K" | "POTASSIUM41" => Self::Potassium41,
            "43CA" | "CALCIUM43" => Self::Calcium43,
            "45SC" | "SCANDIUM45" => Self::Scandium45,
            "47TI" | "TITANIUM47" => Self::Titanium47,
            "49TI" | "TITANIUM49" => Self::Titanium49,
            "50V" | "VANADIUM50" => Self::Vanadium50,
            "51V" | "VANADIUM51" => Self::Vanadium51,
            "53CR" | "CHROMIUM53" => Self::Chromium53,
            "55MN" | "MANGANESE55" => Self::Manganese55,
            "57FE" | "IRON57" => Self::Iron57,
            "59CO" | "COBALT59" => Self::Cobalt59,
            "61NI" | "NICKEL61" => Self::Nickel61,
            "63CU" | "COPPER63" => Self::Copper63,
            "65CU" | "COPPER65" => Self::Copper65,
            "67ZN" | "ZINC67" => Self::Zinc67,
            "69GA" | "GALLIUM69" => Self::Gallium69,
            "71GA" | "GALLIUM71" => Self::Gallium71,
            "73GE" | "GERMANIUM73" => Self::Germanium73,
            "75AS" | "ARSENIC75" => Self::Arsenic75,
            "77SE" | "SELENIUM77" => Self::Selenium77,
            "79BR" | "BROMINE79" => Self::Bromine79,
            "81BR" | "BROMINE81" => Self::Bromine81,
            "83KR" | "KRYPTON83" => Self::Krypton83,
            "85RB" | "RUBIDIUM85" => Self::Rubidium85,
            "87RB" | "RUBIDIUM87" => Self::Rubidium87,
            "87SR" | "STRONTIUM87" => Self::Strontium87,
            "89Y" | "YTTRIUM89" => Self::Yttrium89,
            "91ZR" | "ZIRCONIUM91" => Self::Zirconium91,
            "93NB" | "NIOBIUM93" => Self::Niobium93,
            "95MO" | "MOLYBDENUM95" => Self::Molybdenum95,
            "97MO" | "MOLYBDENUM97" => Self::Molybdenum97,
            "99TC" | "TECHNETIUM99" => Self::Technetium99,
            "99RU" | "RUTHENIUM99" => Self::Ruthenium99,
            "101RU" | "RUTHENIUM101" => Self::Ruthenium101,
            "103RH" | "RHODIUM103" => Self::Rhodium103,
            "105PD" | "PALLADIUM105" => Self::Palladium105,
            "107AG" | "SILVER107" => Self::Silver107,
            "109AG" | "SILVER109" => Self::Silver109,
            "111CD" | "CADMIUM111" => Self::Cadmium111,
            "113CD" | "CADMIUM113" => Self::Cadmium113,
            "113IN" | "INDIUM113" => Self::Indium113,
            "115IN" | "INDIUM115" => Self::Indium115,
            "115SN" | "TIN115" => Self::Tin115,
            "117SN" | "TIN117" => Self::Tin117,
            "119SN" | "TIN119" => Self::Tin119,
            "121SB" | "ANTIMONY121" => Self::Antimony121,
            "123SB" | "ANTIMONY123" => Self::Antimony123,
            "123TE" | "TELLURIUM123" => Self::Tellurium123,
            "125TE" | "TELLURIUM125" => Self::Tellurium125,
            "127I" | "IODINE127" => Self::Iodine127,
            "129XE" | "XENON129" => Self::Xenon129,
            "131XE" | "XENON131" => Self::Xenon131,
            "133CS" | "CAESIUM133" | "CESIUM133" => Self::Caesium133,
            "135BA" | "BARIUM135" => Self::Barium135,
            "137BA" | "BARIUM137" => Self::Barium137,
            "139LA" | "LANTHANUM139" => Self::Lanthanum139,
            "177HF" | "HAFNIUM177" => Self::Hafnium177,
            "179HF" | "HAFNIUM179" => Self::Hafnium179,
            "181TA" | "TANTALUM181" => Self::Tantalum181,
            "183W" | "TUNGSTEN183" => Self::Tungsten183,
            "185RE" | "RHENIUM185" => Self::Rhenium185,
            "187RE" | "RHENIUM187" => Self::Rhenium187,
            "187OS" | "OSMIUM187" => Self::Osmium187,
            "189OS" | "OSMIUM189" => Self::Osmium189,
            "191IR" | "IRIDIUM191" => Self::Iridium191,
            "193IR" | "IRIDIUM193" => Self::Iridium193,
            "195PT" | "PLATINUM195" => Self::Platinum195,
            "197AU" | "GOLD197" => Self::Gold197,
            "199HG" | "MERCURY199" => Self::Mercury199,
            "201HG" | "MERCURY201" => Self::Mercury201,
            "203TL" | "THALLIUM203" => Self::Thallium203,
            "205TL" | "THALLIUM205" => Self::Thallium205,
            "207PB" | "LEAD207" => Self::Lead207,
            "209BI" | "BISMUTH209" => Self::Bismuth209,

            // Lanthanides
            "141PR" | "PRASEODYMIUM141" => Self::Praseodymium141,
            "143ND" | "NEODYMIUM143" => Self::Neodymium143,
            "145ND" | "NEODYMIUM145" => Self::Neodymium145,
            "147SM" | "SAMARIUM147" => Self::Samarium147,
            "149SM" | "SAMARIUM149" => Self::Samarium149,
            "151EU" | "EUROPIUM151" => Self::Europium151,
            "153EU" | "EUROPIUM153" => Self::Europium153,
            "155GD" | "GADOLINIUM155" => Self::Gadolinium155,
            "157GD" | "GADOLINIUM157" => Self::Gadolinium157,
            "159TB" | "TERBIUM159" => Self::Terbium159,
            "161DY" | "DYSPROSIUM161" => Self::Dysprosium161,
            "163DY" | "DYSPROSIUM163" => Self::Dysprosium163,
            "165HO" | "HOLMIUM165" => Self::Holmium165,
            "167ER" | "ERBIUM167" => Self::Erbium167,
            "169TM" | "THULIUM169" => Self::Thulium169,
            "171YB" | "YTTERBIUM171" => Self::Ytterbium171,
            "173YB" | "YTTERBIUM173" => Self::Ytterbium173,
            "175LU" | "LUTETIUM175" => Self::Lutetium175,
            "176LU" | "LUTETIUM176" => Self::Lutetium176,

            // Graceful fallback for unrecognized nuclei
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
            // Main group and transition elements
            Self::Hydrogen => "1H",
            Self::Deuterium => "2H",
            Self::Tritium => "3H",
            Self::Helium3 => "3He",
            Self::Lithium6 => "6Li",
            Self::Lithium7 => "7Li",
            Self::Beryllium9 => "9Be",
            Self::Boron10 => "10B",
            Self::Boron11 => "11B",
            Self::Carbon13 => "13C",
            Self::Nitrogen14 => "14N",
            Self::Nitrogen15 => "15N",
            Self::Oxygen17 => "17O",
            Self::Fluorine19 => "19F",
            Self::Neon21 => "21Ne",
            Self::Sodium23 => "23Na",
            Self::Magnesium25 => "25Mg",
            Self::Aluminium27 => "27Al",
            Self::Silicon29 => "29Si",
            Self::Phosphorus31 => "31P",
            Self::Sulphur33 => "33S",
            Self::Chlorine35 => "35Cl",
            Self::Chlorine37 => "37Cl",
            Self::Potassium39 => "39K",
            Self::Potassium41 => "41K",
            Self::Calcium43 => "43Ca",
            Self::Scandium45 => "45Sc",
            Self::Titanium47 => "47Ti",
            Self::Titanium49 => "49Ti",
            Self::Vanadium50 => "50V",
            Self::Vanadium51 => "51V",
            Self::Chromium53 => "53Cr",
            Self::Manganese55 => "55Mn",
            Self::Iron57 => "57Fe",
            Self::Cobalt59 => "59Co",
            Self::Nickel61 => "61Ni",
            Self::Copper63 => "63Cu",
            Self::Copper65 => "65Cu",
            Self::Zinc67 => "67Zn",
            Self::Gallium69 => "69Ga",
            Self::Gallium71 => "71Ga",
            Self::Germanium73 => "73Ge",
            Self::Arsenic75 => "75As",
            Self::Selenium77 => "77Se",
            Self::Bromine79 => "79Br",
            Self::Bromine81 => "81Br",
            Self::Krypton83 => "83Kr",
            Self::Rubidium85 => "85Rb",
            Self::Rubidium87 => "87Rb",
            Self::Strontium87 => "87Sr",
            Self::Yttrium89 => "89Y",
            Self::Zirconium91 => "91Zr",
            Self::Niobium93 => "93Nb",
            Self::Molybdenum95 => "95Mo",
            Self::Molybdenum97 => "97Mo",
            Self::Technetium99 => "99Tc",
            Self::Ruthenium99 => "99Ru",
            Self::Ruthenium101 => "101Ru",
            Self::Rhodium103 => "103Rh",
            Self::Palladium105 => "105Pd",
            Self::Silver107 => "107Ag",
            Self::Silver109 => "109Ag",
            Self::Cadmium111 => "111Cd",
            Self::Cadmium113 => "113Cd",
            Self::Indium113 => "113In",
            Self::Indium115 => "115In",
            Self::Tin115 => "115Sn",
            Self::Tin117 => "117Sn",
            Self::Tin119 => "119Sn",
            Self::Antimony121 => "121Sb",
            Self::Antimony123 => "123Sb",
            Self::Tellurium123 => "123Te",
            Self::Tellurium125 => "125Te",
            Self::Iodine127 => "127I",
            Self::Xenon129 => "129Xe",
            Self::Xenon131 => "131Xe",
            Self::Caesium133 => "133Cs",
            Self::Barium135 => "135Ba",
            Self::Barium137 => "137Ba",
            Self::Lanthanum139 => "139La",
            Self::Hafnium177 => "177Hf",
            Self::Hafnium179 => "179Hf",
            Self::Tantalum181 => "181Ta",
            Self::Tungsten183 => "183W",
            Self::Rhenium185 => "185Re",
            Self::Rhenium187 => "187Re",
            Self::Osmium187 => "187Os",
            Self::Osmium189 => "189Os",
            Self::Iridium191 => "191Ir",
            Self::Iridium193 => "193Ir",
            Self::Platinum195 => "195Pt",
            Self::Gold197 => "197Au",
            Self::Mercury199 => "199Hg",
            Self::Mercury201 => "201Hg",
            Self::Thallium203 => "203Tl",
            Self::Thallium205 => "205Tl",
            Self::Lead207 => "207Pb",
            Self::Bismuth209 => "209Bi",

            // Lanthanides
            Self::Praseodymium141 => "141Pr",
            Self::Neodymium143 => "143Nd",
            Self::Neodymium145 => "145Nd",
            Self::Samarium147 => "147Sm",
            Self::Samarium149 => "149Sm",
            Self::Europium151 => "151Eu",
            Self::Europium153 => "153Eu",
            Self::Gadolinium155 => "155Gd",
            Self::Gadolinium157 => "157Gd",
            Self::Terbium159 => "159Tb",
            Self::Dysprosium161 => "161Dy",
            Self::Dysprosium163 => "163Dy",
            Self::Holmium165 => "165Ho",
            Self::Erbium167 => "167Er",
            Self::Thulium169 => "169Tm",
            Self::Ytterbium171 => "171Yb",
            Self::Ytterbium173 => "173Yb",
            Self::Lutetium175 => "175Lu",
            Self::Lutetium176 => "176Lu",

            // Graceful fallback for unrecognized nuclei
            Self::Other(value) => value.as_str(),
        };

        write!(f, "{nucleus}")
    }
}

impl From<Nucleus> for String {
    fn from(value: Nucleus) -> Self {
        value.to_string()
    }
}

impl Nucleus {
    /// Returns the nuclear spin quantum number `I`.
    ///
    /// In the case of the fallback variant [`Nucleus::Other`], `None` is
    /// returned.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::ratio::ratio;
    /// use zeenmr_spectrum::Nucleus;
    ///
    /// assert_approx_eq!(f64, Nucleus::Hydrogen.spin().unwrap().get::<ratio>(), 0.5);
    /// assert_approx_eq!(f64, Nucleus::Carbon13.spin().unwrap().get::<ratio>(), 0.5);
    /// ```
    pub fn spin(&self) -> Option<Ratio> {
        match self {
            // Main group and transition elements
            Self::Hydrogen => Some(spin!(0.5)),
            Self::Deuterium => Some(spin!(1.0)),
            Self::Tritium => Some(spin!(0.5)),
            Self::Helium3 => Some(spin!(0.5)),
            Self::Lithium6 => Some(spin!(1.0)),
            Self::Lithium7 => Some(spin!(1.5)),
            Self::Beryllium9 => Some(spin!(1.5)),
            Self::Boron10 => Some(spin!(3.0)),
            Self::Boron11 => Some(spin!(1.5)),
            Self::Carbon13 => Some(spin!(0.5)),
            Self::Nitrogen14 => Some(spin!(1.0)),
            Self::Nitrogen15 => Some(spin!(0.5)),
            Self::Oxygen17 => Some(spin!(2.5)),
            Self::Fluorine19 => Some(spin!(0.5)),
            Self::Neon21 => Some(spin!(1.5)),
            Self::Sodium23 => Some(spin!(1.5)),
            Self::Magnesium25 => Some(spin!(2.5)),
            Self::Aluminium27 => Some(spin!(2.5)),
            Self::Silicon29 => Some(spin!(0.5)),
            Self::Phosphorus31 => Some(spin!(0.5)),
            Self::Sulphur33 => Some(spin!(1.5)),
            Self::Chlorine35 => Some(spin!(1.5)),
            Self::Chlorine37 => Some(spin!(1.5)),
            Self::Potassium39 => Some(spin!(1.5)),
            Self::Potassium41 => Some(spin!(1.5)),
            Self::Calcium43 => Some(spin!(3.5)),
            Self::Scandium45 => Some(spin!(3.5)),
            Self::Titanium47 => Some(spin!(2.5)),
            Self::Titanium49 => Some(spin!(3.5)),
            Self::Vanadium50 => Some(spin!(6.0)),
            Self::Vanadium51 => Some(spin!(3.5)),
            Self::Chromium53 => Some(spin!(1.5)),
            Self::Manganese55 => Some(spin!(2.5)),
            Self::Iron57 => Some(spin!(0.5)),
            Self::Cobalt59 => Some(spin!(3.5)),
            Self::Nickel61 => Some(spin!(1.5)),
            Self::Copper63 => Some(spin!(1.5)),
            Self::Copper65 => Some(spin!(1.5)),
            Self::Zinc67 => Some(spin!(2.5)),
            Self::Gallium69 => Some(spin!(1.5)),
            Self::Gallium71 => Some(spin!(1.5)),
            Self::Germanium73 => Some(spin!(4.5)),
            Self::Arsenic75 => Some(spin!(1.5)),
            Self::Selenium77 => Some(spin!(0.5)),
            Self::Bromine79 => Some(spin!(1.5)),
            Self::Bromine81 => Some(spin!(1.5)),
            Self::Krypton83 => Some(spin!(4.5)),
            Self::Rubidium85 => Some(spin!(2.5)),
            Self::Rubidium87 => Some(spin!(1.5)),
            Self::Strontium87 => Some(spin!(4.5)),
            Self::Yttrium89 => Some(spin!(0.5)),
            Self::Zirconium91 => Some(spin!(2.5)),
            Self::Niobium93 => Some(spin!(4.5)),
            Self::Molybdenum95 => Some(spin!(2.5)),
            Self::Molybdenum97 => Some(spin!(2.5)),
            Self::Technetium99 => Some(spin!(4.5)),
            Self::Ruthenium99 => Some(spin!(2.5)),
            Self::Ruthenium101 => Some(spin!(2.5)),
            Self::Rhodium103 => Some(spin!(0.5)),
            Self::Palladium105 => Some(spin!(2.5)),
            Self::Silver107 => Some(spin!(0.5)),
            Self::Silver109 => Some(spin!(0.5)),
            Self::Cadmium111 => Some(spin!(0.5)),
            Self::Cadmium113 => Some(spin!(0.5)),
            Self::Indium113 => Some(spin!(4.5)),
            Self::Indium115 => Some(spin!(4.5)),
            Self::Tin115 => Some(spin!(0.5)),
            Self::Tin117 => Some(spin!(0.5)),
            Self::Tin119 => Some(spin!(0.5)),
            Self::Antimony121 => Some(spin!(2.5)),
            Self::Antimony123 => Some(spin!(3.5)),
            Self::Tellurium123 => Some(spin!(0.5)),
            Self::Tellurium125 => Some(spin!(0.5)),
            Self::Iodine127 => Some(spin!(2.5)),
            Self::Xenon129 => Some(spin!(0.5)),
            Self::Xenon131 => Some(spin!(1.5)),
            Self::Caesium133 => Some(spin!(3.5)),
            Self::Barium135 => Some(spin!(1.5)),
            Self::Barium137 => Some(spin!(1.5)),
            Self::Lanthanum139 => Some(spin!(3.5)),
            Self::Hafnium177 => Some(spin!(3.5)),
            Self::Hafnium179 => Some(spin!(4.5)),
            Self::Tantalum181 => Some(spin!(3.5)),
            Self::Tungsten183 => Some(spin!(0.5)),
            Self::Rhenium185 => Some(spin!(2.5)),
            Self::Rhenium187 => Some(spin!(2.5)),
            Self::Osmium187 => Some(spin!(0.5)),
            Self::Osmium189 => Some(spin!(1.5)),
            Self::Iridium191 => Some(spin!(1.5)),
            Self::Iridium193 => Some(spin!(1.5)),
            Self::Platinum195 => Some(spin!(0.5)),
            Self::Gold197 => Some(spin!(1.5)),
            Self::Mercury199 => Some(spin!(0.5)),
            Self::Mercury201 => Some(spin!(1.5)),
            Self::Thallium203 => Some(spin!(0.5)),
            Self::Thallium205 => Some(spin!(0.5)),
            Self::Lead207 => Some(spin!(0.5)),
            Self::Bismuth209 => Some(spin!(4.5)),

            // Lanthanides
            Self::Praseodymium141 => Some(spin!(2.5)),
            Self::Neodymium143 => Some(spin!(3.5)),
            Self::Neodymium145 => Some(spin!(3.5)),
            Self::Samarium147 => Some(spin!(3.5)),
            Self::Samarium149 => Some(spin!(3.5)),
            Self::Europium151 => Some(spin!(2.5)),
            Self::Europium153 => Some(spin!(2.5)),
            Self::Gadolinium155 => Some(spin!(1.5)),
            Self::Gadolinium157 => Some(spin!(1.5)),
            Self::Terbium159 => Some(spin!(1.5)),
            Self::Dysprosium161 => Some(spin!(2.5)),
            Self::Dysprosium163 => Some(spin!(2.5)),
            Self::Holmium165 => Some(spin!(3.5)),
            Self::Erbium167 => Some(spin!(3.5)),
            Self::Thulium169 => Some(spin!(0.5)),
            Self::Ytterbium171 => Some(spin!(0.5)),
            Self::Ytterbium173 => Some(spin!(2.5)),
            Self::Lutetium175 => Some(spin!(3.5)),
            Self::Lutetium176 => Some(spin!(7.0)),

            // Graceful fallback for unrecognized nuclei
            Self::Other(_) => None,
        }
    }

    /// Returns the nuclear magnetic moment.
    ///
    /// In the case of the fallback variant [`Nucleus::Other`], `None` is
    /// returned.
    ///
    /// The first value is the nuclear magnetic moment, the second value is the
    /// one-sigma uncertainty.
    ///
    /// Values taken from CODATA ([1], [2], [3]) and [IAEA].
    ///
    /// [1]: https://physics.nist.gov/cgi-bin/cuu/Value?mupsmun
    /// [2]: https://physics.nist.gov/cgi-bin/cuu/Value?mudsmun
    /// [3]: https://physics.nist.gov/cgi-bin/cuu/Value?mutsmun
    /// [IAEA]: https://www-nds.iaea.org/publications/indc/indc-nds-0794.pdf
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::magnetic_moment::nuclear_magneton;
    /// use zeenmr_spectrum::Nucleus;
    ///
    /// let (mu, uncertainty) = Nucleus::Hydrogen.magnetic_moment().unwrap();
    /// assert_approx_eq!(f64, mu.get::<nuclear_magneton>(), 2.792_847_344_63);
    /// assert_approx_eq!(f64, uncertainty.get::<nuclear_magneton>(), 8.2e-10);
    /// ```
    pub fn magnetic_moment(&self) -> Option<(MagneticMoment, MagneticMoment)> {
        match self {
            // Main group and transition elements
            Self::Hydrogen => Some((mag_mom!(2.792_847_344_63), mag_mom!(8.2e-10))),
            Self::Deuterium => Some((mag_mom!(0.857_438_233_5), mag_mom!(2.2e-9))),
            Self::Tritium => Some((mag_mom!(2.978_962_465_0), mag_mom!(5.9e-09))),
            Self::Helium3 => Some((mag_mom!(-2.127_625_31), mag_mom!(3e-08))),
            Self::Lithium6 => Some((mag_mom!(0.822_043), mag_mom!(3e-06))),
            Self::Lithium7 => Some((mag_mom!(3.256_407), mag_mom!(1.2e-05))),
            Self::Beryllium9 => Some((mag_mom!(-1.177_43), mag_mom!(5e-06))),
            Self::Boron10 => Some((mag_mom!(1.800_463_6), mag_mom!(8e-07))),
            Self::Boron11 => Some((mag_mom!(2.688_378), mag_mom!(1e-06))),
            Self::Carbon13 => Some((mag_mom!(0.702_369), mag_mom!(4e-06))),
            Self::Nitrogen14 => Some((mag_mom!(0.403_573), mag_mom!(2e-06))),
            Self::Nitrogen15 => Some((mag_mom!(-0.283_056_9), mag_mom!(1.4e-06))),
            Self::Oxygen17 => Some((mag_mom!(-1.893_543), mag_mom!(1e-05))),
            Self::Fluorine19 => Some((mag_mom!(2.628_321), mag_mom!(4e-06))),
            Self::Neon21 => Some((mag_mom!(-0.661_7), mag_mom!(3e-05))),
            Self::Sodium23 => Some((mag_mom!(2.217_5), mag_mom!(3e-05))),
            Self::Magnesium25 => Some((mag_mom!(-0.855_33), mag_mom!(3e-05))),
            Self::Aluminium27 => Some((mag_mom!(3.640_7), mag_mom!(2e-05))),
            Self::Silicon29 => Some((mag_mom!(-0.555_052), mag_mom!(3e-06))),
            Self::Phosphorus31 => Some((mag_mom!(1.130_925), mag_mom!(5e-06))),
            Self::Sulphur33 => Some((mag_mom!(0.643_25), mag_mom!(2e-05))),
            Self::Chlorine35 => Some((mag_mom!(0.821_7), mag_mom!(2e-05))),
            Self::Chlorine37 => Some((mag_mom!(0.684), mag_mom!(1e-05))),
            Self::Potassium39 => Some((mag_mom!(0.391_47), mag_mom!(8e-06))),
            Self::Potassium41 => Some((mag_mom!(0.214_872), mag_mom!(5e-06))),
            Self::Calcium43 => Some((mag_mom!(-1.317_33), mag_mom!(6e-05))),
            Self::Scandium45 => Some((mag_mom!(4.754), mag_mom!(8e-05))),
            Self::Titanium47 => Some((mag_mom!(-0.788_14), mag_mom!(0.00011))),
            Self::Titanium49 => Some((mag_mom!(-1.103_7), mag_mom!(0.00014))),
            Self::Vanadium50 => Some((mag_mom!(3.344_2), mag_mom!(0.0004))),
            Self::Vanadium51 => Some((mag_mom!(5.146_4), mag_mom!(0.0007))),
            Self::Chromium53 => Some((mag_mom!(-0.474_31), mag_mom!(7e-05))),
            Self::Manganese55 => Some((mag_mom!(3.466_9), mag_mom!(0.0006))),
            Self::Iron57 => Some((mag_mom!(0.0906_4), mag_mom!(7e-05))),
            Self::Cobalt59 => Some((mag_mom!(4.615), mag_mom!(0.025))),
            Self::Nickel61 => Some((mag_mom!(-0.749_65), mag_mom!(5e-05))),
            Self::Copper63 => Some((mag_mom!(2.225_9), mag_mom!(0.0004))),
            Self::Copper65 => Some((mag_mom!(2.384_4), mag_mom!(0.0004))),
            Self::Zinc67 => Some((mag_mom!(0.874_85), mag_mom!(0.00016))),
            Self::Gallium69 => Some((mag_mom!(2.015_02), mag_mom!(6e-05))),
            Self::Gallium71 => Some((mag_mom!(2.560_33), mag_mom!(9e-05))),
            Self::Germanium73 => Some((mag_mom!(-0.878_24), mag_mom!(5e-05))),
            Self::Arsenic75 => Some((mag_mom!(1.438_3), mag_mom!(0.0003))),
            Self::Selenium77 => Some((mag_mom!(0.533_56), mag_mom!(5e-05))),
            Self::Bromine79 => Some((mag_mom!(2.104_6), mag_mom!(0.0006))),
            Self::Bromine81 => Some((mag_mom!(2.268_6), mag_mom!(0.0006))),
            Self::Krypton83 => Some((mag_mom!(-0.970_73), mag_mom!(3e-06))),
            Self::Rubidium85 => Some((mag_mom!(1.353_06), mag_mom!(4e-05))),
            Self::Rubidium87 => Some((mag_mom!(2.751_29), mag_mom!(8e-05))),
            Self::Strontium87 => Some((mag_mom!(-1.093_16), mag_mom!(0.00011))),
            Self::Yttrium89 => Some((mag_mom!(-0.137_298), mag_mom!(5e-06))),
            Self::Zirconium91 => Some((mag_mom!(-1.302_2), mag_mom!(0.0004))),
            Self::Niobium93 => Some((mag_mom!(6.163), mag_mom!(0.002))),
            Self::Molybdenum95 => Some((mag_mom!(-0.913_2), mag_mom!(0.0003))),
            Self::Molybdenum97 => Some((mag_mom!(-0.932_4), mag_mom!(0.0003))),
            Self::Technetium99 => Some((mag_mom!(5.678), mag_mom!(0.002))),
            Self::Ruthenium99 => Some((mag_mom!(-0.641), mag_mom!(0.005))),
            Self::Ruthenium101 => Some((mag_mom!(-0.718), mag_mom!(0.006))),
            Self::Rhodium103 => Some((mag_mom!(-0.088_29), mag_mom!(3e-05))),
            Self::Palladium105 => Some((mag_mom!(-0.642), mag_mom!(0.003))),
            Self::Silver107 => Some((mag_mom!(-0.113_52), mag_mom!(5e-05))),
            Self::Silver109 => Some((mag_mom!(0.130_51), mag_mom!(5e-05))),
            Self::Cadmium111 => Some((mag_mom!(-0.594), mag_mom!(0.0003))),
            Self::Cadmium113 => Some((mag_mom!(-0.621_3), mag_mom!(0.0003))),
            Self::Indium113 => Some((mag_mom!(5.520_8), mag_mom!(0.0004))),
            Self::Indium115 => Some((mag_mom!(5.532_6), mag_mom!(0.0004))),
            Self::Tin115 => Some((mag_mom!(-0.917_4), mag_mom!(0.0004))),
            Self::Tin117 => Some((mag_mom!(-0.999_5), mag_mom!(0.0005))),
            Self::Tin119 => Some((mag_mom!(-1.045_9), mag_mom!(0.0005))),
            Self::Antimony121 => Some((mag_mom!(3.358), mag_mom!(0.0016))),
            Self::Antimony123 => Some((mag_mom!(2.545_7), mag_mom!(0.0012))),
            Self::Tellurium123 => Some((mag_mom!(-0.735_8), mag_mom!(0.0003))),
            Self::Tellurium125 => Some((mag_mom!(-0.887), mag_mom!(0.0005))),
            Self::Iodine127 => Some((mag_mom!(2.808_7), mag_mom!(0.0014))),
            Self::Xenon129 => Some((mag_mom!(-0.777_961), mag_mom!(1.6e-05))),
            Self::Xenon131 => Some((mag_mom!(0.691_845), mag_mom!(7e-06))),
            Self::Caesium133 => Some((mag_mom!(2.577_8), mag_mom!(0.0014))),
            Self::Barium135 => Some((mag_mom!(0.838_1), mag_mom!(0.0002))),
            Self::Barium137 => Some((mag_mom!(0.937_5), mag_mom!(0.0002))),
            Self::Lanthanum139 => Some((mag_mom!(2.779_1), mag_mom!(0.0002))),
            Self::Hafnium177 => Some((mag_mom!(0.791), mag_mom!(0.0009))),
            Self::Hafnium179 => Some((mag_mom!(-0.638_9), mag_mom!(0.0014))),
            Self::Tantalum181 => Some((mag_mom!(2.365), mag_mom!(0.004))),
            Self::Tungsten183 => Some((mag_mom!(0.117_39), mag_mom!(0.00011))),
            Self::Rhenium185 => Some((mag_mom!(3.176), mag_mom!(0.003))),
            Self::Rhenium187 => Some((mag_mom!(3.209), mag_mom!(0.003))),
            Self::Osmium187 => Some((mag_mom!(0.064_42), mag_mom!(7e-05))),
            Self::Osmium189 => Some((mag_mom!(0.657_6), mag_mom!(0.0007))),
            Self::Iridium191 => Some((mag_mom!(0.150_2), mag_mom!(0.0006))),
            Self::Iridium193 => Some((mag_mom!(0.163), mag_mom!(0.0006))),
            Self::Platinum195 => Some((mag_mom!(0.607_3), mag_mom!(0.0007))),
            Self::Gold197 => Some((mag_mom!(0.145_2), mag_mom!(0.0002))),
            Self::Mercury199 => Some((mag_mom!(0.503_9), mag_mom!(0.0006))),
            Self::Mercury201 => Some((mag_mom!(-0.558), mag_mom!(0.0007))),
            Self::Thallium203 => Some((mag_mom!(1.616), mag_mom!(0.002))),
            Self::Thallium205 => Some((mag_mom!(1.632), mag_mom!(0.002))),
            Self::Lead207 => Some((mag_mom!(0.590_6), mag_mom!(0.0004))),
            Self::Bismuth209 => Some((mag_mom!(4.092), mag_mom!(0.002))),

            // Lanthanides
            Self::Praseodymium141 => Some((mag_mom!(4.266), mag_mom!(0.003))),
            Self::Neodymium143 => Some((mag_mom!(-1.065), mag_mom!(0.005))),
            Self::Neodymium145 => Some((mag_mom!(-0.656), mag_mom!(0.004))),
            Self::Samarium147 => Some((mag_mom!(-0.809), mag_mom!(0.0014))),
            Self::Samarium149 => Some((mag_mom!(-0.667_7), mag_mom!(0.0011))),
            Self::Europium151 => Some((mag_mom!(3.463_5), mag_mom!(0.0025))),
            Self::Europium153 => Some((mag_mom!(1.529_4), mag_mom!(0.0011))),
            Self::Gadolinium155 => Some((mag_mom!(-0.259_1), mag_mom!(0.0004))),
            Self::Gadolinium157 => Some((mag_mom!(-0.339_8), mag_mom!(0.0006))),
            Self::Terbium159 => Some((mag_mom!(2.009), mag_mom!(0.004))),
            Self::Dysprosium161 => Some((mag_mom!(-0.479), mag_mom!(0.003))),
            Self::Dysprosium163 => Some((mag_mom!(0.671), mag_mom!(0.004))),
            Self::Holmium165 => Some((mag_mom!(4.16), mag_mom!(0.03))),
            Self::Erbium167 => Some((mag_mom!(-0.5623), mag_mom!(0.0004))),
            Self::Thulium169 => Some((mag_mom!(-0.231), mag_mom!(0.0015))),
            Self::Ytterbium171 => Some((mag_mom!(0.492_3), mag_mom!(0.0004))),
            Self::Ytterbium173 => Some((mag_mom!(-0.678), mag_mom!(0.0006))),
            Self::Lutetium175 => Some((mag_mom!(2.225_7), mag_mom!(0.0019))),
            Self::Lutetium176 => Some((mag_mom!(3.16), mag_mom!(0.005))),

            // Graceful fallback for unrecognized nuclei
            Self::Other(_) => None,
        }
    }

    /// Returns the gyromagnetic ratio.
    ///
    /// In the case of the fallback variant [`Nucleus::Other`], `None` is
    /// returned.
    ///
    /// The first value is the gyromagnetic ratio and the second value is the
    /// one-sigma uncertainty.
    ///
    /// Magnetic momenta taken from [IAEA] and converted to gyromagnetic ratios
    /// using [CODATA] recommended values for the planck constant with the
    /// formula:
    ///
    /// [IAEA]: https://www-nds.iaea.org/publications/indc/indc-nds-0794.pdf
    /// [CODATA]: https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.93.025010
    ///
    /// ```text
    /// γ_n = μ / (I * h)
    /// ```
    ///
    /// where μ is the nuclear magnetic moment, I is the spin quantum number
    /// and h is the planck constant.
    ///
    /// The value for [`Nucleus::Hydrogen`] is not computed and instead uses the
    /// gyromagnetic ratio of a proton provided by [CODATA].
    ///
    /// [CODATA]: https://physics.nist.gov/cgi-bin/cuu/Value?gammap
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use uom::si::gyromagnetic_ratio::{megahertz_per_tesla, radian_per_second_tesla};
    /// use zeenmr_spectrum::Nucleus;
    ///
    /// let (gamma, uncertainty) = Nucleus::Hydrogen.gyromagnetic_ratio().unwrap();
    /// assert_approx_eq!(f64, gamma.get::<megahertz_per_tesla>(), 42.577_478_461);
    /// assert_approx_eq!(f64, uncertainty.get::<megahertz_per_tesla>(), 1.8e-8);
    /// ```
    pub fn gyromagnetic_ratio(&self) -> Option<(GyromagneticRatio, GyromagneticRatio)> {
        if *self == Self::Hydrogen {
            Some((
                GyromagneticRatio::new::<megahertz_per_tesla>(42.577_478_461),
                GyromagneticRatio::new::<megahertz_per_tesla>(0.000_000_018),
            ))
        } else if let (Some(spin), Some((mu, dmu))) = (self.spin(), self.magnetic_moment()) {
            Some((
                mu / (spin * Action::new::<joule_second>(PLANCK)),
                dmu / (spin * Action::new::<joule_second>(PLANCK)),
            ))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use static_assertions::assert_impl_all;
    use strum::IntoEnumIterator;

    #[test]
    fn thread_safety() {
        assert_impl_all!(Nucleus: Send, Sync);
    }

    #[test]
    fn from_str() {
        let nuclei = Nucleus::iter()
            .chain(std::iter::once(Nucleus::Other("Leed 207".to_string())))
            .map(String::from)
            .map(Nucleus::from)
            .collect::<Vec<Nucleus>>();
        let expected = Nucleus::iter()
            .chain(std::iter::once(Nucleus::Other("Leed 207".to_string())))
            .collect::<Vec<Nucleus>>();
        assert_eq!(expected, nuclei);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let nuclei = Nucleus::iter().chain(std::iter::once(Nucleus::Other("Leed 207".to_string())));
        let deserialized = nuclei
            .clone()
            .map(|nucleus| serde_json::to_string(&nucleus).unwrap())
            .map(|serialized| serde_json::from_str::<Nucleus>(&serialized).unwrap())
            .collect::<Vec<Nucleus>>();
        assert_eq!(nuclei.collect::<Vec<Nucleus>>(), deserialized);
    }
}
