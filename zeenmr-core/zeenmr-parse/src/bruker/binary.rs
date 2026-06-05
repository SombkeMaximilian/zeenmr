use crate::data::DataTable;
use byteorder::{ByteOrder, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader};
use std::marker::PhantomData;
use std::path::Path;

/// Reader adapter for Bruker binary files.
#[derive(Debug)]
pub(crate) struct BrukerBinaryReader<E> {
    /// Reader for the main file.
    reader: BufReader<File>,
    /// Endianness of the data in the file.
    endian: PhantomData<E>,
}

impl<E> BrukerBinaryReader<E> {
    /// Attempts to open a Bruker binary file in read-only mode.
    pub(crate) fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
            endian: PhantomData,
        })
    }
}

impl<E> BrukerBinaryReader<E>
where
    E: ByteOrder,
{
    /// Attempts to read complex `i32` values from the file, and returns a data
    /// table with "R" and "I" columns.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`io::Read::read_exact`].
    pub(crate) fn read_i32_complex(self, n: usize, exp: u32) -> io::Result<DataTable<'static>> {
        let mut table = DataTable::new();
        let mut complex = vec![0_i32; n];
        self.reader
            .into_inner()
            .read_i32_into::<E>(&mut complex)?;
        let (real, imag) = complex
            .chunks_exact(2)
            .map(|c| ((c[0] as i64) << exp, (c[1] as i64) << exp))
            .unzip::<i64, i64, Vec<i64>, Vec<i64>>();
        table.insert("R".into(), real.into());
        table.insert("I".into(), imag.into());

        Ok(table)
    }

    /// Attempts to read complex `f64` values from the file, and returns a data
    /// table with "R" and "I" columns.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`io::Read::read_exact`].
    pub(crate) fn read_f64_complex(self, n: usize) -> io::Result<DataTable<'static>> {
        let mut table = DataTable::new();
        let mut complex = vec![0_f64; n];
        self.reader
            .into_inner()
            .read_f64_into::<E>(&mut complex)?;
        let (real, imag) = complex
            .chunks_exact(2)
            .map(|c| (c[0], c[1]))
            .unzip::<f64, f64, Vec<f64>, Vec<f64>>();
        table.insert("R".into(), real.into());
        table.insert("I".into(), imag.into());

        Ok(table)
    }
}
