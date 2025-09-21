use crate::bruker::{DataType, Endian, Parser, Value, read_bruker_binary};
use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::Path;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// An FID dataset read from the Bruker TopSpin file format.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BrukerFid {
    /// Acquisition parameters from the `acqus` file.
    acqus: HashMap<String, Value>,
    /// Raw data from the `fid` file.
    fid: Vec<f64>,
}

impl BrukerFid {
    /// Reads a Bruker FID dataset from the specified directory and experiment
    /// number.
    #[inline]
    pub fn read<P>(path: P, experiment: u32) -> Self
    where
        P: AsRef<Path>,
    {
        let acqus_path = path
            .as_ref()
            .join(format!("{experiment}"))
            .join("acqus");
        let fid_path = path
            .as_ref()
            .join(format!("{experiment}"))
            .join("fid");
        let acqus = read_to_string(acqus_path).unwrap();
        let acqus = Parser::from(acqus.as_str()).parse_source();
        let data_type = match acqus.get("DTYPA") {
            Some(Value::Integer(0)) => DataType::I32,
            Some(Value::Integer(1)) => DataType::F64,
            _ => panic!("unsupported or missing DTYPA value in acqus file"),
        };
        let endian = match acqus.get("BYTORDA") {
            Some(Value::Integer(0)) => Endian::Little,
            Some(_) => Endian::Big,
            _ => panic!("unsupported or missing BYTORDA value in acqus file"),
        };
        let exponent = match acqus.get("NC") {
            Some(Value::Integer(exp)) => (*exp) as i32,
            _ => panic!("missing NC value in acqus file"),
        };
        let size = match acqus.get("TD") {
            Some(Value::Integer(td)) => (*td) as usize,
            _ => panic!("missing TD value in acqus file"),
        };
        let raw = read_bruker_binary(fid_path, size, data_type, endian, exponent);

        Self { acqus, fid: raw }
    }

    /// Return the acquisition parameter with the specified key, if it exists.
    #[inline]
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.acqus.get(key)
    }

    /// Returns the raw FID data.
    #[inline]
    pub fn fid(&self) -> &[f64] {
        &self.fid
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
        let fid = BrukerFid::read(path, 10);
        assert_eq!(fid.acqus.len(), 240);
        assert_eq!(fid.fid.len(), 2_usize.pow(16));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_round_trip() {
        let path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01");
        let fid = BrukerFid::read(path, 10);
        let serialized = serde_json::to_string(&fid).unwrap();
        let deserialized = serde_json::from_str::<BrukerFid>(&serialized).unwrap();
        assert_eq!(fid, deserialized);
    }
}
