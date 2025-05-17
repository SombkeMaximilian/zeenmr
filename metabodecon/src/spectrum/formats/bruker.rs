use crate::Result;
use crate::spectrum::Spectrum;
use crate::spectrum::formats::extract_capture;
use crate::spectrum::meta::Nucleus;
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use regex::Regex;
use std::fs::{File, read_to_string};
use std::io::Read;
use std::path::Path;
use std::sync::LazyLock;

/// Interface for reading 1D NMR spectra in the Bruker TopSpin format.
///
/// # Format
///
/// The Bruker TopSpin file format stores metadata and data in various files.
/// Most of the stored information is not used in this implementation, but the
/// following files are required to read a spectrum:
///
/// ```text
/// name
/// └── name_01
///     └── experiment
///         ├── pdata
///         │   └── processing
///         │       ├── 1r
///         │       └── procs
///         └── acqus
/// ```
///
/// `name` is the name of the dataset, which can be any string. `name_01` is
/// the name of the sample. `experiment` is an integer that represents the
/// type of experiment. Usually a lab will have a convention for which number
/// corresponds to which type of experiment. For example 10 being a 1D NMR
/// experiment. `pdata` is the processing data directory and `processing` is
/// the processing number, which is an arbitrary integer.
///
/// ## Metadata
///
/// The `acqus` and `procs` files contain the acquisition and processing
/// parameters, respectively. They are plain text files with key-value pairs,
/// where each line starts with `##$key=`.
///
/// From the `acqus` file, the following keys are required:
/// * `SW`: The spectral width in ppm as a floating point number.
///
/// From the `procs` file, the following keys are required:
/// * `OFFSET`: The maximum chemical shift in ppm as a floating point number.
/// * `SI`: The size of the data. 2^15 and 2^17 are the expected values.
/// * `BYTORDP`: The endianness of the data, encoded as an integer.
///
///   | Value | Endianness |
///   | ----- | ---------- |
///   | 0     | Little     |
///   | 1     | Big        |
///
/// * `DTYPP`: The data type the raw signal intensities are stored as, encoded
///   as an integer.
///
///   | Value | Type |
///   | ----- | ---- |
///   | 0     | i32  |
///   | 1     | f64  |
///
/// * `NC_proc`: The scaling exponent of the data. If the data is stored as
///   integers, it is scaled by 2 to the power of this value. If the data is
///   stored as floats, this value is unused.
///
/// ## Raw Data
///
/// The raw data is stored in the `1r` file in binary format. The metadata
/// specifies how the data has to be read.
///
/// # Example: Reading a Spectrum
///
/// ```
/// use metabodecon::spectrum::Bruker;
///
/// # fn main() -> metabodecon::Result<()> {
/// let path = "path/to/spectrum";
/// # let path = "../data/bruker/blood/blood_01";
///
/// // Read a single spectrum from a Bruker TopSpin format directory.
/// let spectrum = Bruker::read_spectrum(
///     path,
///     // Experiment number
///     10,
///     // Processing number
///     10,
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
/// use metabodecon::spectrum::Bruker;
///
/// # fn main() -> metabodecon::Result<()> {
/// let path = "path/to/root";
/// # let path = "../data/bruker/blood";
///
/// // Read all spectra from Bruker TopSpin format directories within the root.
/// let spectra = Bruker::read_spectra(
///     path,
///     // Experiment number
///     10,
///     // Processing number
///     10,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub enum Bruker {}

/// Endianness of the raw data. Extracted from the `procs` file.
///
/// | BYTORDP | Endianness |
/// | ------- | ---------- |
/// | 0       | Little     |
/// | 1       | Big        |
#[derive(Debug)]
enum Endian {
    /// Little-endian byte order.
    Little,
    /// Big-endian byte order.
    Big,
}

/// Data type of the raw data. Extracted from the `procs` file.
///
/// | DTYPP | Type |
/// | ----- | ---- |
/// | 0     | i32  |
/// | 1     | f64  |
#[derive(Debug)]
enum Type {
    /// Data stored as 32-bit integers.
    I32,
    /// Data stored as 64-bit floating point numbers.
    F64,
}

/// Acquisition parameters extracted from the `acqus` file.
#[derive(Debug)]
struct AcquisitionParameters {
    /// Spectral width in ppm.
    width: f64,
    /// Frequency of the spectrometer in MHz.
    frequency: f64,
    /// Nucleus observed in the NMR experiment.
    nucleus: Nucleus,
}

/// Regex patterns to search for the acquisition parameters.
static ACQUS_RE: LazyLock<[Regex; 3]> = LazyLock::new(|| {
    [
        Regex::new(r"(?m)^(##\$SW=\s*)(?P<width>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##\$SFO1=\s*)(?P<frequency>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##\$NUC1=\s*<)(?P<nucleus>\w+)").unwrap(),
    ]
});

/// Keys used in the acquisition parameter regex patterns, used for error
/// messages
static ACQUS_KEYS: LazyLock<[&str; 3]> = LazyLock::new(|| ["SW", "SFO1", "NUC1"]);

/// Processing parameters extracted from the `procs` file.
#[derive(Debug)]
struct ProcessingParameters {
    /// Maximum chemical shift in ppm.
    maximum: f64,
    /// Scaling exponent of the data, if it is stored as integers.
    exponent: i32,
    /// Endianness of the data.
    endian: Endian,
    /// Data type of the raw signal intensities.
    data_type: Type,
    /// Size of the data, expected to be 2^15 or 2^17.
    data_size: usize,
}

/// Regex patterns to search for the processing parameters.
static PROCS_RE: LazyLock<[Regex; 5]> = LazyLock::new(|| {
    [
        Regex::new(r"(?m)^(##\$OFFSET=\s*)(?P<maximum>\d+(\.\d+)?)").unwrap(),
        Regex::new(r"(?m)^(##\$NC_proc=\s*)(?P<exponent>-?\d+)").unwrap(),
        Regex::new(r"(?m)^(##\$BYTORDP=\s*)(?P<endian>\d)").unwrap(),
        Regex::new(r"(?m)^(##\$DTYPP=\s*)(?P<data_type>\d)").unwrap(),
        Regex::new(r"(?m)^(##\$SI=\s*)(?P<data_size>\d+)").unwrap(),
    ]
});

/// Keys used in the processing parameter regex patterns, used for error
/// messages
static PROCS_KEYS: LazyLock<[&str; 5]> =
    LazyLock::new(|| ["OFFSET", "NC_proc", "BYTORDP", "DTYPP", "SI"]);

impl Bruker {
    /// Reads the spectrum from a Bruker TopSpin format directory.
    ///
    /// ```text
    /// name
    /// └── name_01 ← the path needs to point to this directory
    ///     └── experiment
    ///         ├── pdata
    ///         │   └── processing
    ///         │       ├── 1r
    ///         │       └── procs
    ///         └── acqus
    /// ```
    ///
    /// # Errors
    ///
    /// The read data is checked for validity to ensure that the `Spectrum` is
    /// well-formed and in a consistent state. The following conditions are
    /// checked:
    /// - The Intensities are not empty.
    /// - The lengths of the chemical shifts and intensities match. The data
    ///   size is read from the metadata files and used to generate the chemical
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
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let path = "path/to/spectrum";
    /// # let path = "../data/bruker/blood/blood_01";
    ///
    /// // Read a single spectrum from a Bruker TopSpin format directory.
    /// let spectrum = Bruker::read_spectrum(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_spectrum<P: AsRef<Path>>(
        path: P,
        experiment: u32,
        processing: u32,
        signal_boundaries: (f64, f64),
    ) -> Result<Spectrum> {
        let acqus_path = path
            .as_ref()
            .join(format!("{}/acqus", experiment));
        let procs_path = path
            .as_ref()
            .join(format!("{}/pdata/{}/procs", experiment, processing));
        let one_r_path = path
            .as_ref()
            .join(format!("{}/pdata/{}/1r", experiment, processing));

        let acqus = Self::read_acquisition_parameters(acqus_path)?;
        let procs = Self::read_processing_parameters(procs_path)?;
        let chemical_shifts = (0..procs.data_size)
            .map(|i| procs.maximum - (i as f64) * acqus.width / (procs.data_size as f64 - 1.0))
            .collect();
        let intensities = Self::read_one_r(one_r_path, procs)?;
        let mut spectrum = Spectrum::new(chemical_shifts, intensities, signal_boundaries)?;
        spectrum.set_nucleus(acqus.nucleus);
        spectrum.set_frequency(acqus.frequency);

        Ok(spectrum)
    }

    /// Reads all spectra from the Bruker TopSpin format directories under the
    /// given path.
    ///
    /// ```text
    /// name ← the path needs to point to this directory
    /// ├── name_01
    /// │   └── experiment
    /// │       ├── pdata
    /// │       │   └── processing
    /// │       │       ├── 1r
    /// │       │       └── procs
    /// │       └── acqus
    /// ├── name_02
    /// │   └── experiment
    /// │       ├── pdata
    /// │       │   └── processing
    /// │       │       ├── 1r
    /// │       │       └── procs
    /// │       └── acqus
    /// ·
    /// ·
    /// ·
    /// ```
    ///
    /// # Errors
    ///
    /// The read data is checked for validity to ensure that the `Spectrum` is
    /// well-formed and in a consistent state. The following conditions are
    /// checked:
    /// - The Intensities are not empty.
    /// - The lengths of the chemical shifts and intensities match. The data
    ///   size is read from the metadata files and used to generate the chemical
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
    /// use metabodecon::spectrum::Bruker;
    ///
    /// # fn main() -> metabodecon::Result<()> {
    /// let path = "path/to/root";
    /// # let path = "../data/bruker/blood";
    ///
    /// // Read all spectra from Bruker TopSpin format directories within the root.
    /// let spectra = Bruker::read_spectra(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn read_spectra<P: AsRef<Path>>(
        path: P,
        experiment: u32,
        processing: u32,
        signal_boundaries: (f64, f64),
    ) -> Result<Vec<Spectrum>> {
        let spectra = path
            .as_ref()
            .read_dir()?
            .filter_map(|entry| {
                entry.ok().and_then(|entry| {
                    if entry.path().is_dir() {
                        Some(entry.path())
                    } else {
                        None
                    }
                })
            })
            .map(|entry| Self::read_spectrum(entry, experiment, processing, signal_boundaries))
            .collect::<Result<Vec<Spectrum>>>()?;

        Ok(spectra)
    }

    /// Internal helper function to read the acquisition parameters from the
    /// `acqus` file and return them.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MissingMetaData`](crate::spectrum::error::Kind::MissingMetadata)
    /// - [`MalformedMetaData`](crate::spectrum::error::Kind::MalformedMetadata)
    /// - [`Error::IoError`](crate::Error::IoError)
    fn read_acquisition_parameters<P: AsRef<Path>>(path: P) -> Result<AcquisitionParameters> {
        let acqus = read_to_string(path.as_ref())?;
        let re = &*ACQUS_RE;
        let keys = &*ACQUS_KEYS;

        let width = extract_capture(&re[0], "width", &acqus, &path, keys[0])?;
        let frequency = extract_capture(&re[1], "frequency", &acqus, &path, keys[1])?;
        let nucleus = extract_capture(&re[2], "nucleus", &acqus, &path, keys[2])?;

        Ok(AcquisitionParameters {
            width,
            frequency,
            nucleus,
        })
    }

    /// Internal helper function to read the processing parameters from the
    /// `procs` file and return them.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`MissingMetaData`](crate::spectrum::error::Kind::MissingMetadata)
    /// - [`MalformedMetaData`](crate::spectrum::error::Kind::MalformedMetadata)
    /// - [`Error::IoError`](crate::Error::IoError)
    fn read_processing_parameters<P: AsRef<Path>>(path: P) -> Result<ProcessingParameters> {
        let procs = read_to_string(path.as_ref())?;
        let re = &*PROCS_RE;
        let keys = &*PROCS_KEYS;

        let spectrum_maximum = extract_capture(&re[0], "maximum", &procs, &path, keys[0])?;
        let scaling_exponent = extract_capture(&re[1], "exponent", &procs, &path, keys[1])?;
        let endian = match extract_capture(&re[2], "endian", &procs, &path, keys[2])? {
            0 => Endian::Little,
            _ => Endian::Big,
        };
        let data_type = match extract_capture(&re[3], "data_type", &procs, &path, keys[3])? {
            0 => Type::I32,
            _ => Type::F64,
        };
        let data_size = extract_capture(&re[4], "data_size", &procs, &path, keys[4])?;

        Ok(ProcessingParameters {
            maximum: spectrum_maximum,
            exponent: scaling_exponent,
            endian,
            data_type,
            data_size,
        })
    }

    /// Internal helper function to read the raw data from the `1r` file and
    /// return it as a vector of floating point numbers. As working with
    /// chemical shifts in increasing order is generally simpler, the vector is
    /// reversed before being returned.
    ///
    /// # Errors
    ///
    /// The following errors are possible:
    /// - [`Error::IoError`](crate::Error::IoError)
    fn read_one_r<P: AsRef<Path>>(path: P, procs: ProcessingParameters) -> Result<Vec<f64>> {
        let mut one_r = File::open(path.as_ref())?;
        let mut buffer = vec![
            0;
            procs.data_size
                * match procs.data_type {
                    Type::I32 => 4,
                    Type::F64 => 8,
                }
        ];
        one_r.read_exact(&mut buffer)?;

        match procs.data_type {
            Type::I32 => {
                let mut temp = vec![0_i32; procs.data_size];
                match procs.endian {
                    Endian::Little => buffer
                        .as_slice()
                        .read_i32_into::<LittleEndian>(&mut temp)?,
                    Endian::Big => buffer
                        .as_slice()
                        .read_i32_into::<BigEndian>(&mut temp)?,
                }

                Ok(temp
                    .into_iter()
                    .map(|value| (value as f64) * 2_f64.powi(procs.exponent))
                    .collect::<Vec<f64>>())
            }
            Type::F64 => {
                let mut temp = vec![0_f64; procs.data_size];
                match procs.endian {
                    Endian::Little => buffer
                        .as_slice()
                        .read_f64_into::<LittleEndian>(&mut temp)?,
                    Endian::Big => buffer
                        .as_slice()
                        .read_f64_into::<BigEndian>(&mut temp)?,
                }

                Ok(temp)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{check_blood_spectrum, check_sim_spectrum};
    use float_cmp::assert_approx_eq;

    #[test]
    fn read_spectrum() {
        let sim_path = "../data/bruker/sim/sim_01";
        let blood_path = "../data/bruker/blood/blood_01";
        let sim = Bruker::read_spectrum(sim_path, 10, 10, (3.34, 3.56)).unwrap();
        let blood = Bruker::read_spectrum(blood_path, 10, 10, (-2.2, 11.8)).unwrap();
        check_sim_spectrum!(sim);
        check_blood_spectrum!(blood);
    }

    #[test]
    fn read_spectra() {
        let sim_path = "../data/bruker/sim";
        let blood_path = "../data/bruker/blood";
        let sim = Bruker::read_spectra(sim_path, 10, 10, (3.34, 3.56)).unwrap();
        let blood = Bruker::read_spectra(blood_path, 10, 10, (-2.2, 11.8)).unwrap();
        assert_eq!(sim.len(), 16);
        assert_eq!(blood.len(), 16);
        sim.iter().for_each(|spectrum| {
            check_sim_spectrum!(spectrum);
        });
        blood.iter().for_each(|spectrum| {
            check_blood_spectrum!(spectrum);
        });
    }

    #[test]
    fn read_acquisition_parameters() {
        let path = "../data/bruker/blood/blood_01/10/acqus";
        let acqus = Bruker::read_acquisition_parameters(path).unwrap();
        assert_approx_eq!(f64, acqus.width, 20.0236139622347);
        assert_approx_eq!(f64, acqus.frequency, 600.252821089118);
        assert_eq!(acqus.nucleus, Nucleus::Hydrogen1);
    }

    #[test]
    fn read_processing_parameters() {
        let path = "../data/bruker/blood/blood_01/10/pdata/10/procs";
        let procs = Bruker::read_processing_parameters(path).unwrap();
        assert_approx_eq!(f64, procs.maximum, 14.81146);
        assert_eq!(procs.exponent, 0);
        match procs.endian {
            Endian::Little => {}
            _ => panic!("Expected Little, got {:?}", procs.endian),
        };
        match procs.data_type {
            Type::I32 => {}
            Type::F64 => panic!("Expected I32, got F64"),
        }
        assert_eq!(procs.data_size, 2_usize.pow(17));
    }
}
