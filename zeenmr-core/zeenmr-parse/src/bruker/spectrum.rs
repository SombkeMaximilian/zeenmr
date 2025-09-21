use crate::bruker::{DataType, Endian, Parser, Value, read_bruker_binary};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;
use uom::si::f64::{Frequency, Ratio};
use uom::si::frequency::{hertz, megahertz};
use uom::si::ratio::part_per_million as ppm;
use zeenmr_spectrum::Spectrum;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A processed spectrum read from the Bruker TopSpin file format.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrukerSpectrum {
    /// Experiment identifier.
    id: Option<String>,
    /// Acquisition parameters from the `acqus` file.
    acqus: HashMap<String, Value>,
    /// Processing parameters from the `procs` file.
    procs: HashMap<String, Value>,
    /// Real values of the fourier-transformed FID from the `1r` file.
    real: Vec<f64>,
    /// Imaginary values of the fourier-transformed FID from the `1i` file.
    imag: Vec<f64>,
}

impl TryFrom<BrukerSpectrum> for Spectrum {
    type Error = ();

    fn try_from(value: BrukerSpectrum) -> Result<Self, Self::Error> {
        let range = match value.procs.get("SW_p") {
            Some(Value::Float(sw)) => (
                Frequency::new::<hertz>(*sw),
                Frequency::new::<hertz>(0.0),
            ),
            _ => return Err(()),
        };
        let larmor = match value.procs.get("SF") {
            Some(Value::Float(sf)) => Frequency::new::<megahertz>(*sf),
            _ => return Err(()),
        };
        let mut spectrum = Spectrum::new(
            value.real,
            larmor,
            range,
        ).unwrap();
        if let Some(id) = value.id {
            spectrum.set_id(id);
        }
        if let Some(Value::String(nucleus)) = value.acqus.get("NUC1") {
            spectrum.set_nucleus(nucleus);
        }
        if let Some(Value::Float(offset)) = value.procs.get("OFFSET") {
            spectrum.set_shift_reference_value(Ratio::new::<ppm>(*offset)).unwrap();
        }

        Ok(spectrum)
    }
}

impl BrukerSpectrum {
    /// Reads a processed Bruker spectrum from the specified directory,
    /// experiment number, and processing number.
    #[inline]
    pub fn read<P>(path: P, experiment: u32, processing: u32) -> Self
    where
        P: AsRef<Path>,
    {
        let id = path
            .as_ref()
            .file_name()
            .map(|name| name.to_string_lossy().to_string());
        let acqus_path = path
            .as_ref()
            .join(format!("{experiment}"))
            .join("acqus");
        let procs_dir = path
            .as_ref()
            .join(format!("{experiment}"))
            .join("pdata")
            .join(format!("{processing}"));
        let procs_path = procs_dir.join("procs");
        let real_path = procs_dir.join("1r");
        let imag_path = procs_dir.join("1i");
        let acqus = read_to_string(acqus_path).unwrap();
        let acqus = Parser::from(acqus.as_str()).parse_source();
        let procs = read_to_string(procs_path).unwrap();
        let procs = Parser::from(procs.as_str()).parse_source();
        let data_type = match procs.get("DTYPP") {
            Some(Value::Integer(0)) => DataType::I32,
            Some(Value::Integer(1)) => DataType::F64,
            _ => panic!("missing DTYPP value in procs file"),
        };
        let endian = match procs.get("BYTORDP") {
            Some(Value::Integer(0)) => Endian::Little,
            Some(_) => Endian::Big,
            _ => panic!("missing BYTORDA value in procs file"),
        };
        let exponent = match procs.get("NC_proc") {
            Some(Value::Integer(exp)) => (*exp) as i32,
            _ => panic!("missing NC_proc value in procs file"),
        };
        let size = match procs.get("SI") {
            Some(Value::Integer(si)) => (*si) as usize,
            _ => panic!("missing TD value in procs file"),
        };
        let real = read_bruker_binary(real_path, size, data_type, endian, exponent);
        let imag = read_bruker_binary(imag_path, size, data_type, endian, exponent);

        Self {
            id,
            acqus,
            procs,
            real,
            imag,
        }
    }

    /// Returns the experiment identifier.
    #[inline]
    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    /// Return the acquisition parameter with the specified key, if it exists.
    #[inline]
    pub fn get_acqus(&self, key: &str) -> Option<&Value> {
        self.acqus.get(key)
    }

    /// Return the processing parameter with the specified key, if it exists.
    #[inline]
    pub fn get_procs(&self, key: &str) -> Option<&Value> {
        self.procs.get(key)
    }

    /// Returns the real part of the spectrum.
    #[inline]
    pub fn real(&self) -> &[f64] {
        &self.real
    }

    /// Returns the imaginary part of the spectrum.
    #[inline]
    pub fn imag(&self) -> &[f64] {
        &self.imag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn workspace_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    #[test]
    fn read() {
        let path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01");
        let spectrum = BrukerSpectrum::read(path, 10, 10);
        assert_eq!(spectrum.acqus.len(), 240);
        assert_eq!(spectrum.procs.len(), 124);
        assert_eq!(spectrum.real().len(), 2_usize.pow(17));
        assert_eq!(spectrum.imag().len(), 2_usize.pow(17));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01");
        let spectrum = BrukerSpectrum::read(path, 10, 10);
        let serialized = serde_json::to_string(&spectrum).unwrap();
        let deserialized = serde_json::from_str::<BrukerSpectrum>(&serialized).unwrap();
        assert_eq!(spectrum, deserialized);
    }
}
