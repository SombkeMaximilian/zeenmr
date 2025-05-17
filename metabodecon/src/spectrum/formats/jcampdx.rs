use crate::Result;
use crate::spectrum::Spectrum;
use crate::spectrum::error::{Error, Kind};
use crate::spectrum::formats::{extract_capture, extract_row};
use crate::spectrum::meta::{Nucleus, ReferenceCompound};
use regex::{Captures, Regex};
use std::ffi::OsStr;
use std::fs::read_to_string;
use std::path::Path;
use std::sync::LazyLock;

/// Interface for reading 1D NMR spectra in JCAMP-DX format.
///
/// The JCAMP-DX format is a text-based format for storing 1D NMR spectra. Both
/// the metadata and the data are stored in the same file, which can be divided
/// into the header and the data section. Since the format allows for a lot of
/// optionally included information, this implementation is restricted to the
/// minimally required sets to construct a 1D NMR spectrum from.
///
/// # Supported Versions and Formats
///
/// This implementation will only ever handle version 5.xx and 6.xx of the
/// format. Due to the fact that version 4.xx was not designed for NMR spectra
/// but for other kinds of spectroscopy data, it is not possible to construct
/// meaningful NMR spectra from it consistently. As a result, version 4.xx will
/// never be supported, but it may be possible to read in such files by changing
/// the version to 5.00 or 6.00, as long as the relevant metadata is present.
///
/// NMR data can be stored in a few different ways. The `NMR Spectrum` variant
/// is currently supported for the XYDATA and NTUPLES format. `NMR FID` may be
/// supported in the future. `NMR PEAK TABLE` and `NMR PEAK ASSIGNMENT` will
/// never be supported.
///
/// # Header Metadata
///
/// The metadata is stored as key-value pairs, where the lines start with
/// `##key=`. To successfully parse a 1D NMR spectrum, a minimal set of metadata
/// is required. Some additional information will also be extracted from the
/// metadata if it is present. The following information must be present in the
/// header:
///
/// | Key                  | Description                     |
/// |----------------------|---------------------------------|
/// | `JCAMPDX`            | Version of the JCAMP-DX format. |
/// | `DATA_TYPE`          | Type of data in the file.       |
/// | `DATA_CLASS`         | Way the data is stored.         |
/// | `.OBSERVE FREQUENCY` | Frequency of the spectrometer.  |
/// | `.OBSERVE NUCLEUS`   | Nucleus being observed.         |
///
/// The following additional information will only be extracted if present:
///
/// | Key                  | Description                     |
/// |----------------------|---------------------------------|
/// | `.SOLVENT NAME`      | Solvent used in the experiment. |
/// | `.SOLVENT REFERENCE` | Reference for the solvent.      |
/// | `.SHIFT REFERENCE`   | Chemical shift reference.       |
///
/// # Format Specific Metadata
///
/// Depending on which format is used, the metadata labels to reconstruct the
/// data are different. While there is more information within a JCAMP-DX file,
/// only the minimum set of information required to reconstruct the spectrum is
/// parsed. The following sections define this set.
///
/// ## XYData
///
/// | Key       | Description                        |
/// |-----------|------------------------------------|
/// | `XUNITS`  | Units of the x values (Hz or ppm). |
/// | `YFACTOR` | Scale factor of the y values.      |
/// | `FIRSTX`  | First x value.                     |
/// | `LASTX`   | Last x value.                      |
/// | `NPOINTS` | Number of x and y values           |
///
/// ## NTuples
///
/// | Key       | Description                                                 |
/// |-----------|-------------------------------------------------------------|
/// | `SYMBOL`  | Column labels of the metadata table (X and Y/R are needed). |
/// | `VAR_DIM` | Number values (only X).                                     |
/// | `UNITS`   | Units of the values (Hz or ppm, only X).                    |
/// | `FIRST`   | First value (only X).                                       |
/// | `LAST`    | Last value (only X).                                        |
/// | `FACTOR`  | Scale factor of the values (only Y/R).                      |
///
/// # Data Integrity Checks
///
/// Officially, the JCAMP-DX format specifies some integrity checks that should
/// be performed when parsing the files. These will not be performed by this
/// implementation, as they often aren't adhered to due to how much freedom
/// JCAMP-DX allows when writing the files.
///
/// # Compression
///
/// JCAMP-DX uses encoding schemes to compress the data. For XYData and NTuples,
/// the x values are stored as indices and only the first x value is stored in
/// each new line. The y values are stored in various forms depending on the
/// compression schemes that were used. Generally, the following structure is
/// expected:
///
/// ```text
/// X1 Y1 Y2 Y3 ... Yi
/// Xi Yi+1 Yi+2 ... Yn
/// ```
///
/// ## AFFN
///
/// JCAMP-DX stores uncompressed data in ASCII Free Format Numeric (AFFN), which
/// is human-readable and easily parsable. It consists of only the usual
/// characters used for displaying numeric values (digits, '+', '-', decimal
/// point).
///
/// #### Example
///
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     21    -68
/// ```
///
/// ### PAC
///
/// One simple modification of AFFN is the Packed (PAC) form, which specifies
/// that adjacent values are separated by their sign rather than whitespace.
///
/// #### Example
///
/// ```text
/// 9 +23-73+92-12+77
/// 4 +81+19+21+21-68
/// ```
///
/// A space is usually inserted between the x value and sequence of y values by
/// convention.
///
/// ## ASDF
///
/// JCAMP-DX uses the ASCII Squeezed Difference Form (ASDF) compression scheme,
/// which is a collection of encoding strategies. This ranges from light
/// compression, which is still somewhat human-readable, to heavier compression,
/// which is more difficult to read.
///
/// ### SQZ
///
/// The first compression strategy that is included in the JCAMP-DX format is
/// the Squeezed (SQZ) form. This encoding replaces the leading digit of each y
/// value, including its sign, with an ASCII character and contains no delimiter
/// between values. This format is still fairly human-readable as it simply uses
/// the first 9 letters of the alphabet as the corresponding digits, with the
/// upper case letters representing positive leading digits and the lower case
/// letters representing negative leading digits.
///
/// | Digit    | Encoded (positive) | Encoded (negative) |
/// |----------|--------------------|--------------------|
/// | 0        | @                  | @                  |
/// | 1        | A                  | a                  |
/// | 2        | B                  | b                  |
/// | 3        | C                  | c                  |
/// | 4        | D                  | d                  |
/// | 5        | E                  | e                  |
/// | 6        | F                  | f                  |
/// | 7        | G                  | g                  |
/// | 8        | H                  | h                  |
/// | 9        | I                  | i                  |
///
/// #### Example
///
/// Uncompressed:
///
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     21    -68
/// ```
///
/// Compressed:
///
/// ```text
/// 9 B3g3R2a2G7
/// 4 H1A9B1B1f8
/// ```
///
/// ### DIF
///
/// The second compression scheme is the Difference (DIF) form, which uses SQZ
/// for the first y value in each line and encodes the subsequent y values as
/// differences to the respective previous y value. Similarly to SQZ, leading
/// digits of the differences are replaced by ASCII characters, with upper case
/// representing positive values and lower case representing negative values.
///
/// | Digit    | Encoded (positive) | Encoded (negative) |
/// |----------|--------------------|--------------------|
/// | 0        | %                  | %                  |
/// | 1        | J                  | j                  |
/// | 2        | K                  | k                  |
/// | 3        | L                  | l                  |
/// | 4        | M                  | m                  |
/// | 5        | N                  | n                  |
/// | 6        | O                  | o                  |
/// | 7        | P                  | p                  |
/// | 8        | Q                  | q                  |
/// | 9        | R                  | r                  |
///
/// This scheme also includes a data integrity check. If the last y value in a
/// line is encoded using DIF, the first y value in the next line will repeat
/// the last y value of the previous line.
///
/// #### Example
///
/// Uncompressed:
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     21    -68
/// ```
///
/// Compressed:
/// ```text
/// 9 B3r6J65j04Q9
/// 5 G7Mo2K%q9
/// ```
///
/// Note the repetition of the last value in the first line (77 encoded as G7 on
/// line 2) and the x value being incremented to account for it.
///
/// ### DUP
///
/// The final compression scheme is Duplicate Suppression (DUP). If 2 or more
/// duplicate values are identical, only the first value is kept and subsequent
/// values are encoded as the number of repetitions (including the initial
/// value). The leading digit of this new representation is then replaced by
/// the respective ASCII character from the table below.
///
/// | Repetitions | Encoded |
/// |-------------|---------|
/// | 1           | S       |
/// | 2           | T       |
/// | 3           | U       |
/// | 4           | V       |
/// | 5           | W       |
/// | 6           | X       |
/// | 7           | Y       |
/// | 8           | Z       |
/// | 9           | s       |
///
/// #### Example
///
/// Uncompressed:
///
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     21    -68
/// ```
///
/// Compressed:
///
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     T     -68
/// ```
///
/// ### DIFDUP
///
/// DIF and DUP are often combined for full compression. The DIF compression
/// is performed first, and any occurrences of % are replaced by the respective
/// DUP characters.
///
/// #### Example
///
/// Uncompressed:
///
/// ```text
/// 9     23    -73     92    -12     77
/// 4     81     19     21     21    -68
/// ```
///
/// Compressed (DIF):
///
/// ```text
/// 9 B3r6J65j04Q9
/// 5 G7Mo2K%q9
/// ```
///
/// Compressed (DUP):
///
/// ```text
/// 9 B3r6J65j04Q9
/// 5 G7Mo2KTq9
/// ```
///
/// # Example: Reading a Spectrum
///
/// ```
/// use metabodecon::spectrum::JcampDx;
///
/// # fn main() -> metabodecon::Result<()> {
/// let path = "path/to/spectrum.dx";
/// # let path = "../data/jcamp-dx/blood/blood_01.dx";
///
/// // Read a spectrum from a JCAMP-DX file.
/// let spectrum = JcampDx::read_spectrum(
///     path,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
/// # Ok(())
/// # }
/// ```
///
/// # Example: Reading Multiple Spectra
///
/// ```
/// use metabodecon::spectrum::JcampDx;
///
/// # fn main() -> metabodecon::Result<()> {
/// let path = "path/to/root";
/// # let path = "../data/jcamp-dx/blood";
///
/// // Read all spectra from a directory of JCAMP-DX files.
/// let spectra = JcampDx::read_spectra(
///     path,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub enum JcampDx {}

/// Type of data that is stored in the JCAMP-DX file.
///
/// This can be either NMR Spectrum or NMR FID. Currently only Spectrum is
/// supported. Other options will never be supported.
#[derive(Debug)]
enum DataType {
    /// An already fourier transformed FID is a Spectrum.
    Spectrum,
}

/// Which specific format of JCAMP-DX is used to store the data.
///
/// Depending on which format is used, the metadata labels to reconstruct the
/// data are different. While there is more information within a JCAMP-DX file,
/// only the minimum set of information required to reconstruct the spectrum is
/// parsed.
///
/// # XYData
///
/// | Key       | Description                        |
/// |-----------|------------------------------------|
/// | `XUNITS`  | Units of the x values (Hz or ppm). |
/// | `YFACTOR` | Scale factor of the y values.      |
/// | `FIRSTX`  | First x value.                     |
/// | `LASTX`   | Last x value.                      |
/// | `NPOINTS` | Number of x and y values           |
///
/// # NTuples
///
/// | Key       | Description                                                 |
/// |-----------|-------------------------------------------------------------|
/// | `SYMBOL`  | Column labels of the metadata table (X and Y/R are needed). |
/// | `VAR_DIM` | Number values (only X).                                     |
/// | `UNITS`   | Units of the values (Hz or ppm, only X).                    |
/// | `FIRST`   | First value (only X).                                       |
/// | `LAST`    | Last value (only X).                                        |
/// | `FACTOR`  | Scale factor of the values (only Y/R).                      |
#[derive(Debug)]
enum Format {
    /// File contains only one data block (one real part).
    XYData,
    /// File contains multiple data blocks (real and imaginary parts).
    NTuples,
}

/// Units of the x values.
///
/// Usually spectra are stored using Hz and displayed using ppm, but they can
/// also be stored using ppm.
#[derive(Debug)]
enum XUnits {
    /// Hertz.
    Hz,
    /// Parts per million.
    Ppm,
}

/// Metadata that can be parsed irrespective of the specific JCAMP-DX format.
#[derive(Debug)]
struct Header {
    /// The type of data (processed Spectrum or raw FID).
    ///
    /// Currently unused, as only Spectrum is supported.
    #[allow(dead_code)]
    data_type: DataType,
    /// The data format (NTuples or XYData).
    format: Format,
    /// The spectrometer frequency used to calculate ppm values.
    frequency: f64,
    /// The observed nucleus (e.g. 1H, 13C).
    nucleus: Nucleus,
    /// Optional reference compound used to offset the ppm values
    reference_compound: Option<ReferenceCompound>,
}

/// Regex patterns to search for the header metadata.
static HEADER_RE: LazyLock<[Regex; 11]> = LazyLock::new(|| {
    [
        Regex::new(r"(?m)^(##JCAMP(\s*|_|-)DX=\s*)(?P<version>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##DATA(\s|_)TYPE=\s*)(?P<type>\w+\s\w+)").unwrap(),
        Regex::new(r"(?m)^(##DATA(\s|_)CLASS=\s*)(?P<format>\w+(\s\w+)?)").unwrap(),
        Regex::new(r"(?m)^(##\.OBSERVE(\s|_)FREQUENCY=\s*)(?P<frequency>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##\.OBSERVE(\s|_)NUCLEUS=\s*)(?P<nucleus>\^\w+)").unwrap(),
        Regex::new(r"(?m)^(##\.SOLVENT(\s|_)NAME=\s*)(?P<name>.*)").unwrap(),
        Regex::new(r"(?m)^(##\.SOLVENT(\s|_)REFERENCE=\s*)(?P<shift>\d+(\.\d+))?").unwrap(),
        Regex::new(r"(?m)^(##\.SHIFT(\s|_)REFERENCE=\s*)(?P<method>[^,]*)").unwrap(),
        Regex::new(r"(?m)^(##\.SHIFT(\s|_)REFERENCE=[^,]*,\s*)(?P<name>[^,]*)").unwrap(),
        Regex::new(r"(?m)^(##\.SHIFT(\s|_)REFERENCE=[^,]*,[^,]*,\s*)(?P<index>\d+)").unwrap(),
        Regex::new(r"(?m)^(##\.SHIFT(\s|_)REFERENCE=[^,]*,[^,]*,[^,]*,\s*)(?P<shift>\d+(\.\d+)?)")
            .unwrap(),
    ]
});

/// Keys used in the header regex patterns, used for error messages.
static HEADER_KEYS: LazyLock<[&str; 11]> = LazyLock::new(|| {
    [
        "JCAMPDX",
        "DATA_TYPE",
        "DATA_CLASS",
        ".OBSERVE FREQUENCY",
        ".OBSERVE NUCLEUS",
        ".SOLVENT NAME",
        ".SOLVENT REFERENCE",
        ".SHIFT REFERENCE [METHOD]",
        ".SHIFT REFERENCE [COMPOUND]",
        ".SHIFT REFERENCE [INDEX]",
        ".SHIFT REFERENCE [SHIFT]",
    ]
});

/// A block of data in a JCAMP-DX file from which spectra can be reconstructed.
///
/// This defines the minimum set of metadata required to build a valid spectrum.
#[derive(Debug)]
struct DataBlock {
    /// Units of the x values (Hz or ppm).
    x_units: XUnits,
    /// Scale factor of the y values.
    factor: f64,
    /// First x value.
    first: f64,
    /// Last x value.
    last: f64,
    /// Number of x and y values.
    data_size: usize,
    /// The data, encoded as a JCAMP-DX string.
    data: String,
}

/// Regex patterns to search for the XYData format specific metadata.
static XY_DATA_RE: LazyLock<[Regex; 6]> = LazyLock::new(|| {
    [
        Regex::new(r"(?m)^(##XUNITS=\s*)(?P<xunits>\w+)").unwrap(),
        Regex::new(r"(?m)^(##YFACTOR=\s*)(?P<factor>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##FIRSTX=\s*)(?P<first>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##LASTX=\s*)(?P<last>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##NPOINTS=\s*)(?P<data_size>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##XYDATA=\s*\(X\+\+\([RY]\.\.[RY]\)\)(.*)?)(?P<data>[^#$]*)").unwrap(),
    ]
});

/// Keys used in the XYData regex patterns, used for error messages.
static XY_DATA_KEYS: LazyLock<[&str; 6]> =
    LazyLock::new(|| ["XUNITS", "YFACTOR", "FIRSTX", "LASTX", "NPOINTS", "XYDATA"]);

/// Regex patterns to search for the NTuples format specific metadata.
static N_TUPLES_RE: LazyLock<[Regex; 7]> = LazyLock::new(|| {
    [
        Regex::new(r"(?m)^(##SYMBOL=\s*)(?P<symbols>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##VAR(\s*|_)DIM=\s*)(?P<data_sizes>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##UNITS=\s*)(?P<units>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##FIRST=\s*)(?P<first>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##LAST=\s*)(?P<last>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##FACTOR=\s*)(?P<factor>.*)(\r\n|\n|\r)").unwrap(),
        Regex::new(r"(?m)^(##DATA(\s|_)TABLE=\s*\(X\+\+\(([RY])\.\.[RY]\)\)(.*)?)(?P<data>[^#$]*)")
            .unwrap(),
    ]
});

/// Keys used in the NTuples regex patterns, used for error messages.
static N_TUPLES_KEYS: LazyLock<[&str; 7]> = LazyLock::new(|| {
    [
        "SYMBOL",
        "VAR DIM",
        "UNITS",
        "FIRST",
        "LAST",
        "FACTOR",
        "DATA TABLE",
    ]
});

/// Regex patterns used to find values encoded using specific schemes.
static ENCODING: LazyLock<[Regex; 6]> = LazyLock::new(|| {
    [
        Regex::new(r"(?P<asdf>[@%A-Za-z+-])").unwrap(), // ASDF
        Regex::new(r"(?P<pac>[+-]\d)").unwrap(),        // PAC
        Regex::new(r"(?P<sqz>[@A-Ia-i])").unwrap(),     // SQZ
        Regex::new(
            r"\s+(?P<dif>[%J-Rj-r]\d*)\s*(?P<dup>([S-Zs]\d*)?)\s*((\r\n|\n|\r)\s*(?P<next>\d+))",
        )
        .unwrap(), // DIFDUP Checkpoints
        Regex::new(r"\s+(?P<val>[+-]*\d*)\s+(?P<dif>[%J-Rj-r]\d*)").unwrap(), // DIF
        Regex::new(r"\s+(?P<val>[+-]*\d+)\s+(?P<dup>[S-Zs]\d*)").unwrap(), // DUP
    ]
});

impl JcampDx {
    /// Reads the spectrum from a JCAMP-DX file.
    ///
    /// # Errors
    ///
    /// The read data is checked for validity to ensure that the `Spectrum` is
    /// well-formed and in a consistent state. The following conditions are
    /// checked:
    /// - The Intensities are not empty.
    /// - The lengths of the chemical shifts and intensities match. The data
    ///   size is read from the metadata and used to generate the chemical
    ///   shifts.
    /// - All intensity values are finite.
    /// - The signal region boundaries are within the range of the chemical
    ///   shifts.
    /// - All required key-value pairs are extracted from the metadata files.
    ///
    /// Additionally, if any [`I/O`] errors occur, an error variant containing
    /// the original error is returned.
    ///
    /// [`I/O`]: std::io
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::JcampDx;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let path = "path/to/spectrum.dx";
    /// # let path = "../data/jcamp-dx/blood/blood_01.dx";
    ///
    /// // Read a spectrum from a JCAMP-DX file.
    /// let spectrum = JcampDx::read_spectrum(
    ///     path,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_spectrum<P: AsRef<Path>>(
        path: P,
        signal_boundaries: (f64, f64),
    ) -> Result<Spectrum> {
        let path = path.as_ref();
        let dx = read_to_string(path)?;
        let header = Self::read_header(&dx, path)?;
        let block = match header.format {
            Format::XYData => Self::read_xydata(&dx, path)?,
            Format::NTuples => Self::read_ntuples(&dx, path)?,
        };
        let conversion = match block.x_units {
            XUnits::Hz => 1.0 / header.frequency,
            XUnits::Ppm => 1.0,
        };
        let step = (block.last - block.first) * conversion / (block.data_size as f64 - 1.0);
        let offset = match &header.reference_compound {
            Some(reference) => reference.chemical_shift() - reference.index() as f64 * step,
            None => block.first * conversion,
        };
        let chemical_shifts = (0..block.data_size)
            .map(|i| offset + (i as f64) * step)
            .collect();
        let intensities = match ENCODING[0].is_match(block.data.as_str()) {
            true => Self::decode_asdf(&block.data, block.factor, path)?,
            false => Self::decode_affn(&block.data, block.factor, path)?,
        };
        let mut spectrum = Spectrum::new(chemical_shifts, intensities, signal_boundaries)?;
        spectrum.set_nucleus(header.nucleus);
        spectrum.set_frequency(header.frequency);
        if let Some(reference) = header.reference_compound {
            spectrum.set_reference_compound(reference);
        }

        Ok(spectrum)
    }

    /// Reads all spectra from a directory of JCAMP-DX files.
    ///
    /// Skips any files that do not have the `.dx` extension.
    ///
    /// # Errors
    ///
    /// The read data is checked for validity to ensure that the `Spectrum` is
    /// well-formed and in a consistent state. The following conditions are
    /// checked:
    /// - The Intensities are not empty.
    /// - The lengths of the chemical shifts and intensities match. The data
    ///   size is read from the metadata and used to generate the chemical
    ///   shifts.
    /// - All intensity values are finite.
    /// - The signal region boundaries are within the range of the chemical
    ///   shifts.
    /// - All required key-value pairs are extracted from the metadata files.
    ///
    /// Additionally, if any [`I/O`] errors occur, an error variant containing
    /// the original error is returned.
    ///
    /// [`I/O`]: std::io
    ///
    /// # Example
    ///
    /// ```
    /// use metabodecon::spectrum::JcampDx;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let path = "path/to/root";
    /// # let path = "../data/jcamp-dx/blood";
    ///
    /// // Read all spectra from a directory of JCAMP-DX files.
    /// let spectra = JcampDx::read_spectra(
    ///     path,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_spectra<P: AsRef<Path>>(
        path: P,
        signal_boundaries: (f64, f64),
    ) -> Result<Vec<Spectrum>> {
        let spectra = path
            .as_ref()
            .read_dir()?
            .filter_map(|entry| {
                entry.ok().and_then(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(OsStr::to_str)
                        .and_then(|extension| match extension.to_lowercase().as_str() {
                            "dx" => Some(entry.path()),
                            _ => None,
                        })
                })
            })
            .map(|entry| Self::read_spectrum(entry, signal_boundaries))
            .collect::<Result<Vec<Spectrum>>>()?;

        Ok(spectra)
    }

    /// Internal helper function to read the metadata from the general file
    /// header.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MissingMetadata`](Kind::MissingMetadata)
    /// - [`MalformedMetadata`](Kind::MalformedMetadata)
    fn read_header<P: AsRef<Path>>(dx: &str, path: P) -> Result<Header> {
        let re = &*HEADER_RE;
        let keys = &*HEADER_KEYS;

        match extract_capture::<f64, _>(&re[0], "version", dx, &path, keys[0])?.trunc() {
            5.0 | 6.0 => (),
            _ => return Err(Error::new(Kind::UnsupportedJcampDxFile).into()),
        };
        let data_type = match extract_capture::<String, _>(&re[1], "type", dx, &path, keys[1])?
            .to_uppercase()
            .as_str()
        {
            "NMR SPECTRUM" => DataType::Spectrum,
            _ => return Err(Error::new(Kind::UnsupportedJcampDxFile).into()),
        };
        let format = match extract_capture::<String, _>(&re[2], "format", dx, &path, keys[2])?
            .to_uppercase()
            .as_str()
        {
            "XYDATA" => Format::XYData,
            "NTUPLES" => Format::NTuples,
            _ => return Err(Error::new(Kind::UnsupportedJcampDxFile).into()),
        };
        let frequency = extract_capture(&re[3], "frequency", dx, &path, keys[3])?;
        let nucleus = extract_capture(&re[4], "nucleus", dx, &path, keys[4])?;
        let reference_compound = {
            let method = extract_capture(&re[7], "method", dx, &path, keys[7]).ok();
            let name = extract_capture::<String, _>(&re[8], "name", dx, &path, keys[8]).ok();
            let index = extract_capture::<usize, _>(&re[9], "index", dx, &path, keys[9]).ok();
            let shift = extract_capture(&re[10], "shift", dx, &path, keys[10]).ok();

            if let (Some(shift), Some(index)) = (shift, index) {
                Some(ReferenceCompound::new(shift, index - 1, name, method))
            } else {
                let name = extract_capture::<String, _>(&re[5], "name", dx, &path, keys[5]).ok();
                let shift = extract_capture(&re[6], "shift", dx, &path, keys[6]).ok();

                shift.map(|shift| ReferenceCompound::new(shift, 0, name, None))
            }
        };

        Ok(Header {
            data_type,
            format,
            frequency,
            nucleus,
            reference_compound,
        })
    }

    /// Internal helper function to read the metadata from the XYData specific
    /// file header and the data table.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MissingMetadata`](Kind::MissingMetadata)
    /// - [`MalformedMetadata`](Kind::MalformedMetadata)
    /// - [`MissingData`](Kind::MissingData)
    /// - [`MalformedData`](Kind::MalformedData)
    fn read_xydata<P: AsRef<Path>>(dx: &str, path: P) -> Result<DataBlock> {
        let re = &*XY_DATA_RE;
        let keys = &*XY_DATA_KEYS;

        let x_units = match extract_capture::<String, _>(&re[0], "xunits", dx, &path, keys[0])?
            .to_uppercase()
            .as_str()
        {
            "HZ" => XUnits::Hz,
            "PPM" => XUnits::Ppm,
            _ => return Err(Error::new(Kind::UnsupportedJcampDxFile).into()),
        };
        let factor = extract_capture(&re[1], "factor", dx, &path, keys[1])?;
        let first = extract_capture(&re[2], "first", dx, &path, keys[2])?;
        let last = extract_capture(&re[3], "last", dx, &path, keys[3])?;
        let data_size = extract_capture(&re[4], "data_size", dx, &path, keys[4])?;
        let data = extract_capture::<String, _>(&re[5], "data", dx, &path, keys[5])?
            .as_str()
            .trim()
            .to_string();

        if data.is_empty() {
            return Err(Error::new(Kind::MissingData {
                path: path.as_ref().to_path_buf(),
            })
            .into());
        }

        Ok(DataBlock {
            x_units,
            factor,
            first,
            last,
            data_size,
            data,
        })
    }

    /// Internal helper function to read the metadata from the NTuples specific
    /// file header and the data table.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MissingMetadata`](Kind::MissingMetadata)
    /// - [`MalformedMetadata`](Kind::MalformedMetadata)
    /// - [`MissingData`](Kind::MissingData)
    /// - [`MalformedData`](Kind::MalformedData)
    fn read_ntuples<P: AsRef<Path>>(dx: &str, path: P) -> Result<DataBlock> {
        let re = &*N_TUPLES_RE;
        let keys = &*N_TUPLES_KEYS;

        let symbols = extract_capture::<String, _>(&re[0], "symbols", dx, &path, keys[0])?
            .split(",")
            .map(|symbol| symbol.trim().to_string())
            .collect::<Vec<_>>();
        let x_column = symbols
            .iter()
            .position(|symbol| symbol.to_uppercase() == "X")
            .ok_or_else(|| {
                Error::new(Kind::MissingMetadata {
                    key: keys[0].to_string(),
                    path: path.as_ref().to_path_buf(),
                })
            })?;
        let r_column = symbols
            .iter()
            .position(|symbol| symbol.to_uppercase() == "R" || symbol.to_uppercase() == "Y")
            .ok_or_else(|| {
                Error::new(Kind::MissingMetadata {
                    key: keys[0].to_string(),
                    path: path.as_ref().to_path_buf(),
                })
            })?;

        let data_size = extract_row::<usize, _>(&re[1], "data_sizes", dx, &path, keys[1])?
            .get(x_column)
            .copied()
            .ok_or_else(|| {
                Error::new(Kind::MalformedMetadata {
                    key: keys[1].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: "Could not find X column".to_string(),
                })
            })?;
        let x_units = match extract_row::<String, _>(&re[2], "units", dx, &path, keys[2])?
            .get(x_column)
            .ok_or_else(|| {
                Error::new(Kind::MalformedMetadata {
                    key: keys[2].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: "Could not find X column".to_string(),
                })
            })?
            .to_uppercase()
            .as_str()
        {
            "HZ" => XUnits::Hz,
            "PPM" => XUnits::Ppm,
            unit => {
                return Err(Error::new(Kind::MalformedMetadata {
                    key: keys[2].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: format!("Unsupported x unit: {}", unit),
                })
                .into());
            }
        };
        let first = extract_row::<f64, _>(&re[3], "first", dx, &path, keys[3])?
            .get(x_column)
            .copied()
            .ok_or_else(|| {
                Error::new(Kind::MalformedMetadata {
                    key: keys[3].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: "Could not find X column".to_string(),
                })
            })?;
        let last = extract_row::<f64, _>(&re[4], "last", dx, &path, keys[4])?
            .get(x_column)
            .copied()
            .ok_or_else(|| {
                Error::new(Kind::MalformedMetadata {
                    key: keys[4].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: "Could not find X column".to_string(),
                })
            })?;
        let factor = extract_row::<f64, _>(&re[5], "factor", dx, &path, keys[5])?
            .get(r_column)
            .copied()
            .ok_or_else(|| {
                Error::new(Kind::MalformedMetadata {
                    key: keys[5].to_string(),
                    path: path.as_ref().to_path_buf(),
                    details: "Could not find R column".to_string(),
                })
            })?;
        let data = extract_capture::<String, _>(&re[6], "data", dx, &path, keys[6])?
            .as_str()
            .trim()
            .to_string();

        if data.is_empty() {
            return Err(Error::new(Kind::MissingData {
                path: path.as_ref().to_path_buf(),
            })
            .into());
        }

        Ok(DataBlock {
            x_units,
            factor,
            first,
            last,
            data_size,
            data,
        })
    }

    /// Internal helper function to convert an AFFN string into a `Vec<f64>`.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MalformedData`](Kind::MalformedData)
    fn decode_affn<P: AsRef<Path>>(data: &str, factor: f64, path: P) -> Result<Vec<f64>> {
        let intensities = data
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .skip(1)
                    .map(|value| {
                        value.parse::<f64>().map_err(|error| {
                            Error::new(Kind::MalformedData {
                                path: path.as_ref().to_path_buf(),
                                details: format!("{} ({})", value, error),
                            })
                            .into()
                        })
                    })
                    .collect::<Result<Vec<f64>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .map(|intensity| intensity * factor)
            .collect();

        Ok(intensities)
    }

    /// Internal helper function to convert a string encoded using any
    /// combination of ASDF schemes into a `Vec<f64>`.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MalformedData`](Kind::MalformedData)
    fn decode_asdf<P: AsRef<Path>>(data: &str, factor: f64, path: P) -> Result<Vec<f64>> {
        let re = &*ENCODING;

        let data = re[0].replace_all(data, " $asdf");
        let data = re[1].replace_all(&data, " $pac");
        let data = re[2].replace_all(&data, |captures: &Captures| {
            Self::undo_sqz(captures.name("sqz").unwrap().as_str())
        });
        let mut data = re[3]
            .replace_all(&data, |captures: &Captures| {
                let dif = captures.name("dif").unwrap().as_str();
                let dup = captures.name("dup").unwrap().as_str();
                let next = captures.name("next").unwrap().as_str();

                match dup {
                    "" | "S" => format!(" \n{}", next),
                    _ => format!(" {} {} \n{}", dif, Self::decrement_dup(dup), next),
                }
            })
            .to_string();
        loop {
            let tmp_data_dif = re[4].replace_all(&data, |captures: &Captures| {
                let value = captures.name("val").unwrap().as_str();
                let encoded = captures.name("dif").unwrap().as_str();

                Self::undo_dif(value, encoded)
            });
            let tmp_data_dup = re[5].replace_all(&tmp_data_dif, |captures: &Captures| {
                let value = captures.name("val").unwrap().as_str();
                let encoded = captures.name("dup").unwrap().as_str();

                Self::undo_dup(value, encoded)
            });
            data = tmp_data_dup.to_string();

            if !re[4].is_match(&data) && !re[5].is_match(&data) {
                break;
            }
        }

        Self::decode_affn(&data, factor, path)
    }

    /// Maps the SQZ characters onto the respective digit and returns it as a
    /// string.
    fn undo_sqz(character: &str) -> String {
        match character {
            "@" => "0",
            "A" => "1",
            "B" => "2",
            "C" => "3",
            "D" => "4",
            "E" => "5",
            "F" => "6",
            "G" => "7",
            "H" => "8",
            "I" => "9",
            "a" => "-1",
            "b" => "-2",
            "c" => "-3",
            "d" => "-4",
            "e" => "-5",
            "f" => "-6",
            "g" => "-7",
            "h" => "-8",
            "i" => "-9",
            _ => unreachable!("Invalid SQZ character: {}", character),
        }
        .to_string()
    }

    /// Maps the DUP characters onto the respective repetitions and returns
    /// the value repeated that many times as a string.
    fn undo_dup(value: &str, encoded: &str) -> String {
        let mut decoded = match encoded.chars().next().unwrap() {
            'S' => "1",
            'T' => "2",
            'U' => "3",
            'V' => "4",
            'W' => "5",
            'X' => "6",
            'Y' => "7",
            'Z' => "8",
            's' => "9",
            _ => unreachable!("Invalid DUP character: {}", encoded),
        }
        .to_string();
        decoded.extend(encoded.chars().skip(1));
        let duplicates = decoded.parse::<usize>().unwrap();

        format!(" {}", value).repeat(duplicates)
    }

    /// Maps the DIF characters onto the respective digit and returns the
    /// previous value and the decoded value as a string.
    fn undo_dif(value: &str, encoded: &str) -> String {
        let mut decoded = match encoded.chars().next().unwrap() {
            '%' => "0",
            'J' => "1",
            'K' => "2",
            'L' => "3",
            'M' => "4",
            'N' => "5",
            'O' => "6",
            'P' => "7",
            'Q' => "8",
            'R' => "9",
            'j' => "-1",
            'k' => "-2",
            'l' => "-3",
            'm' => "-4",
            'n' => "-5",
            'o' => "-6",
            'p' => "-7",
            'q' => "-8",
            'r' => "-9",
            _ => unreachable!("Invalid DIF character: {}", encoded),
        }
        .to_string();
        decoded.extend(encoded.chars().skip(1));
        let value = value.parse::<i64>().unwrap();
        let difference = decoded.parse::<i64>().unwrap();

        format!(" {} {}", value, value + difference)
    }

    /// Internal helper function to decrement a DUP encoded value by one.
    ///
    /// This is used to remove the data integrity checkpoints in the encoded
    /// data if the last value in a line is a DUP of a previous DIF.
    fn decrement_dup(encoded: &str) -> String {
        let mut decoded = match encoded.chars().next().unwrap() {
            'S' => "1",
            'T' => "2",
            'U' => "3",
            'V' => "4",
            'W' => "5",
            'X' => "6",
            'Y' => "7",
            'Z' => "8",
            's' => "9",
            _ => unreachable!("Invalid DUP character: {}", encoded),
        }
        .to_string();
        decoded.extend(encoded.chars().skip(1));
        let decremented = (decoded.parse::<usize>().unwrap() - 1).to_string();
        let mut encoded = match decremented.chars().next().unwrap() {
            '0' => "",
            '1' => "S",
            '2' => "T",
            '3' => "U",
            '4' => "V",
            '5' => "W",
            '6' => "X",
            '7' => "Y",
            '8' => "Z",
            '9' => "s",
            _ => unreachable!(
                "Non-numeric leading character in parsed usize: {}",
                decremented
            ),
        }
        .to_string();
        encoded.extend(decremented.chars().skip(1));

        encoded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_blood_spectrum;
    use float_cmp::assert_approx_eq;

    #[test]
    fn read_affn_spectrum() {
        let affn_paths = [
            "../data/jcamp-dx/test/v5/xydata_affn.dx",
            "../data/jcamp-dx/test/v6/xydata_affn.dx",
            "../data/jcamp-dx/test/v5/ntuples_affn.dx",
            "../data/jcamp-dx/test/v6/ntuples_affn.dx",
        ];
        affn_paths
            .map(|path| JcampDx::read_spectrum(path, (1.0, 1.1)).unwrap())
            .into_iter()
            .for_each(|mut spectrum| {
                spectrum.set_reference_compound(spectrum.range().0);
                check_blood_spectrum!(spectrum);
            });
    }

    #[test]
    fn read_pac_spectrum() {
        let pac_paths = [
            "../data/jcamp-dx/test/v5/xydata_pac.dx",
            "../data/jcamp-dx/test/v6/xydata_pac.dx",
            "../data/jcamp-dx/test/v5/ntuples_pac.dx",
            "../data/jcamp-dx/test/v6/ntuples_pac.dx",
        ];
        pac_paths
            .map(|path| JcampDx::read_spectrum(path, (1.0, 1.1)).unwrap())
            .into_iter()
            .for_each(|mut spectrum| {
                spectrum.set_reference_compound(spectrum.range().0);
                check_blood_spectrum!(spectrum);
            });
    }

    #[test]
    fn read_sqz_spectrum() {
        let sqz_paths = [
            "../data/jcamp-dx/test/v5/xydata_sqz.dx",
            "../data/jcamp-dx/test/v6/xydata_sqz.dx",
            "../data/jcamp-dx/test/v5/ntuples_sqz.dx",
            "../data/jcamp-dx/test/v6/ntuples_sqz.dx",
        ];
        sqz_paths
            .map(|path| JcampDx::read_spectrum(path, (1.0, 1.1)).unwrap())
            .into_iter()
            .for_each(|mut spectrum| {
                spectrum.set_reference_compound(spectrum.range().0);
                check_blood_spectrum!(spectrum);
            });
    }

    #[test]
    fn read_dif_dup_spectrum() {
        let difdup_paths = [
            "../data/jcamp-dx/test/v5/xydata_difdup.dx",
            "../data/jcamp-dx/test/v6/xydata_difdup.dx",
            "../data/jcamp-dx/test/v5/ntuples_difdup.dx",
            "../data/jcamp-dx/test/v6/ntuples_difdup.dx",
        ];
        difdup_paths
            .map(|path| JcampDx::read_spectrum(path, (1.0, 1.1)).unwrap())
            .into_iter()
            .for_each(|mut spectrum| {
                spectrum.set_reference_compound(spectrum.range().0);
                check_blood_spectrum!(spectrum);
            });
    }

    #[test]
    fn read_header() {
        let path = "../data/jcamp-dx/test/v6/ntuples_difdup.dx";
        let dx = read_to_string(path).unwrap();
        let header = JcampDx::read_header(&dx, path).unwrap();
        match header.data_type {
            DataType::Spectrum => (),
        };
        match header.format {
            Format::XYData => panic!("Expected NTuples"),
            Format::NTuples => (),
        };
        assert_approx_eq!(f64, header.frequency, 600.252821089118);
        assert_eq!(header.nucleus, Nucleus::Hydrogen1);
        if let Some(reference) = header.reference_compound {
            assert_approx_eq!(f64, reference.chemical_shift(), 14.81146);
            assert_eq!(reference.index(), 0);
            assert_eq!(reference.name().unwrap(), "Plasma");
            assert_eq!(
                reference.method().unwrap(),
                crate::spectrum::meta::ReferencingMethod::Internal
            );
        } else {
            panic!("Expected Some");
        }
    }

    #[test]
    fn read_xydata() {
        let path = "../data/jcamp-dx/test/v6/xydata_difdup.dx";
        let dx = read_to_string(path).unwrap();
        let xy_data = JcampDx::read_xydata(&dx, path).unwrap();
        match xy_data.x_units {
            XUnits::Hz => (),
            XUnits::Ppm => panic!("Expected Hz"),
        }
        assert_approx_eq!(f64, xy_data.factor, 1.0);
        assert_approx_eq!(f64, xy_data.first, 12019.1390697773);
        assert_approx_eq!(f64, xy_data.last, 0.0);
        assert_eq!(xy_data.data_size, 2_usize.pow(17));
    }

    #[test]
    fn read_ntuples() {
        let path = "../data/jcamp-dx/test/v6/ntuples_difdup.dx";
        let dx = read_to_string(path).unwrap();
        let n_tuples = JcampDx::read_ntuples(&dx, path).unwrap();
        match n_tuples.x_units {
            XUnits::Hz => (),
            XUnits::Ppm => panic!("Expected Hz"),
        }
        assert_approx_eq!(f64, n_tuples.factor, 1.0);
        assert_approx_eq!(f64, n_tuples.first, 12019.1390697773);
        assert_approx_eq!(f64, n_tuples.last, 0.0);
        assert_eq!(n_tuples.data_size, 2_usize.pow(17));
    }

    #[test]
    fn decode_affn() {
        let data = "\
            19        482       -763        215       -632\n\
            15       -924        357       -678        841\n\
            11        512       -194        321       -467\n\
            7        -689        278        278        732\n\
            3         835       -619        247       -193";
        let expected = [
            482.0, -763.0, 215.0, -632.0, -924.0, 357.0, -678.0, 841.0, 512.0, -194.0, 321.0,
            -467.0, -689.0, 278.0, 278.0, 732.0, 835.0, -619.0, 247.0, -193.0,
        ];
        let decoded = JcampDx::decode_affn(data, 1.0, "decode_affn_test").unwrap();
        decoded
            .into_iter()
            .zip(expected)
            .for_each(|(decoded, expected)| {
                assert_approx_eq!(f64, decoded, expected);
            });
    }

    #[test]
    fn decode_pac() {
        let data = "\
            19 +482-763+215-632-924+357-678+841+512-194\n\
            9  +321-467-689+278+278+732+835-619+247-193";
        let expected = [
            482.0, -763.0, 215.0, -632.0, -924.0, 357.0, -678.0, 841.0, 512.0, -194.0, 321.0,
            -467.0, -689.0, 278.0, 278.0, 732.0, 835.0, -619.0, 247.0, -193.0,
        ];
        let decoded = JcampDx::decode_asdf(data, 1.0, "decode_pac_test").unwrap();
        decoded
            .into_iter()
            .zip(expected)
            .for_each(|(decoded, expected)| {
                assert_approx_eq!(f64, decoded, expected);
            });
    }

    #[test]
    fn decode_sqz() {
        let data = "\
            19 D82g63B15f32i24C57f78H41E12a94\n\
            9  C21d67f89B78B78G32H35f19B47a93";
        let expected = [
            482.0, -763.0, 215.0, -632.0, -924.0, 357.0, -678.0, 841.0, 512.0, -194.0, 321.0,
            -467.0, -689.0, 278.0, 278.0, 732.0, 835.0, -619.0, 247.0, -193.0,
        ];
        let decoded = JcampDx::decode_asdf(data, 1.0, "decode_sqz_test").unwrap();
        decoded
            .into_iter()
            .zip(expected)
            .for_each(|(decoded, expected)| {
                assert_approx_eq!(f64, decoded, expected);
            });
    }

    #[test]
    fn decode_dif_dup() {
        let data = "\
            19 D82j245R78q47k92J281j035J519l29p06\n\
            10 a94N15p88k22R67TM54J03j454Q66m40";
        let expected = [
            482.0, -763.0, 215.0, -632.0, -924.0, 357.0, -678.0, 841.0, 512.0, -194.0, 321.0,
            -467.0, -689.0, 278.0, 278.0, 732.0, 835.0, -619.0, 247.0, -193.0,
        ];
        let decoded = JcampDx::decode_asdf(data, 1.0, "decode_sqz_test").unwrap();
        decoded
            .into_iter()
            .zip(expected)
            .for_each(|(decoded, expected)| {
                assert_approx_eq!(f64, decoded, expected);
            });
    }
}
