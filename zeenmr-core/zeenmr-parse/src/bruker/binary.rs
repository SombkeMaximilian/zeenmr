use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Data type of the raw data. Extracted from the `acqus`/`procs` file.
///
/// | DTYPA/P | Type |
/// | ------- | ---- |
/// | 0       | i32  |
/// | 1       | f64  |
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) enum DataType {
    /// Data stored as 32-bit signed integers.
    I32,
    /// Data stored as 64-bit floating point numbers.
    F64,
}

impl DataType {
    /// Returns the size in bytes of the data type.
    fn size_of(&self) -> usize {
        match self {
            DataType::I32 => size_of::<i32>(),
            DataType::F64 => size_of::<f64>(),
        }
    }
}

/// Endianness of the raw data. Extracted from the `acqus`/`procs` file.
///
/// | BYTORDA/P | Endianness |
/// | --------- | ---------- |
/// | 0         | Little     |
/// | 1         | Big        |
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub(crate) enum Endian {
    /// Little-endian byte order.
    Little,
    /// Big-endian byte order.
    Big,
}

/// Reads a Bruker binary file (e.g., `1r`, `1i`, `fid`) and returns its
/// contents as a collection of `f64` values.
pub(crate) fn read_bruker_binary<P>(
    path: P,
    size: usize,
    data_type: DataType,
    endian: Endian,
    exponent: i32,
) -> Vec<f64>
where
    P: AsRef<Path>,
{
    let mut file = File::open(path).unwrap();
    let mut buffer = vec![0; size * data_type.size_of()];
    file.read_exact(&mut buffer).unwrap();

    match data_type {
        DataType::I32 => {
            let mut data = vec![0_i32; size];
            match endian {
                Endian::Little => buffer
                    .as_slice()
                    .read_i32_into::<LittleEndian>(&mut data)
                    .unwrap(),
                Endian::Big => buffer
                    .as_slice()
                    .read_i32_into::<BigEndian>(&mut data)
                    .unwrap(),
            };

            data.into_iter()
                .map(|x| (x as f64) * 2_f64.powi(exponent))
                .collect()
        }
        DataType::F64 => {
            let mut data = vec![0_f64; size];
            match endian {
                Endian::Little => buffer
                    .as_slice()
                    .read_f64_into::<LittleEndian>(&mut data)
                    .unwrap(),
                Endian::Big => buffer
                    .as_slice()
                    .read_f64_into::<BigEndian>(&mut data)
                    .unwrap(),
            };

            data
        }
    }
}
