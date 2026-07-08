use crate::bruker::error::{Error, Result};
use crate::data::{Column, DataTable, Dataset, ParameterTable, Value};
use crate::jcampdx::parse_jcamp_dx;
use byteorder::{BigEndian, ByteOrder, LittleEndian, ReadBytesExt};
use std::path::Path;
use std::{fs, io};

/// Maximum number of dimensions to try.
const MAX_DIM: usize = 16;

/// Dataset that can be read from Bruker directories.
pub type BrukerDataset = Dataset<'static, Error>;

/// Reads raw and processed data from a Bruker directory.
///
/// The following directory structure is expected.
///
/// ```text
/// root
/// └── exp_id
///     ├── pdata
///     │   └── proc_id
///     │       ├── <processed binary files>
///     │       └── <procs file(s)>
///     ├── <raw binary file>
///     └── <acqus file(s)>
/// ```
///
/// # Errors
///
/// Returns an error if
/// - the `acqus` file is missing or can't be parsed, or
/// - the raw binary file is missing (`fid` or `ser`), or
/// - the `procs` file is missing or can't be parsed, or
/// - the processed binary file is missing (`1r` and/or `1i`, etc.)
pub fn read_bruker_dir<P>(root: P, exp_id: u32, proc_id: u32) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    let exp_root = root.as_ref().join(exp_id.to_string());
    let mut dataset = read_bruker_exp(&exp_root)?;
    let proc_root = exp_root.join("pdata").join(proc_id.to_string());
    dataset
        .children
        .push(read_bruker_proc(proc_root, false)?);
    if dataset.data_parameters.len() != dataset.children[0].data_parameters.len() {
        // maybe this should be a fatal error, but that makes fixing things
        // impossible for users
        dataset.errors.insert(0, Error::incoherent_dimensionality());
    }

    Ok(dataset)
}

/// Reads raw data from a Bruker experiment directory.
///
/// The number of dimensions is inferred from how many `acquNs` files are found
/// alongside `acqus` (see below), and determines which processed binary file is
/// read.
///
/// The following directory structure is expected.
///
/// ```text
/// root
/// ├── <raw binary file>
/// └── <acqus file(s)>
/// ```
///
/// # Errors
///
/// Returns an error if
/// - the `acqus` file is missing or can't be parsed, or
/// - the raw binary file is missing (`fid` or `ser`)
pub fn read_bruker_exp<P>(root: P) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    let mut dataset = BrukerDataset::default();

    let exp_root = root.as_ref();
    let acqus_path = exp_root.join("acqus");
    let acqus_src = fs::read_to_string(acqus_path)?;
    let mut parsed = parse_jcamp_dx(&acqus_src)?;
    parsed.parameters.set_id("acqus");
    dataset.parameters = parsed.parameters.into_owned();
    dataset
        .errors
        .extend(parsed.errors.into_iter().map(Error::from));
    drop(acqus_src);

    for dim in 2..=MAX_DIM {
        let next_acqus_name = format!("acqu{dim}s");
        let next_acqus_path = exp_root.join(&next_acqus_name);
        let parameters = match fs::read_to_string(next_acqus_path) {
            Ok(acqus_src) => {
                let mut parsed = parse_jcamp_dx(&acqus_src)?;
                dataset
                    .errors
                    .extend(parsed.errors.into_iter().map(Error::from));
                parsed.parameters.set_id(next_acqus_name);

                parsed.parameters.into_owned()
            }
            Err(_) => break,
        };
        dataset.data_parameters.push(parameters);
    }

    let raw_name = if dataset.data_parameters.is_empty() { "fid" } else { "ser" };
    let raw = read_raw_data(exp_root.join(&raw_name), &dataset.parameters)?;
    let mut table = DataTable::new();
    table.set_id("RAW");
    table.insert(raw_name.into(), raw);
    dataset.data_tables.push(table);

    Ok(dataset)
}

/// Reads processed data from a Bruker directory.
///
/// The number of dimensions is inferred from how many `procNs` files are found
/// alongside `procs` (see below), and determines which processed binary file(s)
/// are read. A processed binary file for an `n`-dimensional dataset is named
/// `<n><components>`. Components is a string of `n` characters, each either `r`
/// (real) or `i` (imaginary), one per dimension, ordered from the indirect
/// dimension(s) to the direct (acquisition) dimension. For example, a 2D
/// dataset's fully real binary file is named `2rr`, while its fully imaginary
/// counterpart is `2ii`.
///
/// If `real_only` is `true`, only the all-real binary file (`1r`, `2rr`,
/// `3rrr`, etc.) is read. Otherwise, every combination of `r`/`i` across all
/// `n` dimensions is read.
///
/// The following directory structure is expected.
///
/// ```text
/// root
/// ├── <processed binary files>
/// └── <procs file(s)>
/// ```
///
/// # Errors
///
/// Returns an error if
/// - the `procs` file is missing or can't be parsed, or
/// - the processed binary file is missing (`1r` and/or `1i`, etc.)
pub fn read_bruker_proc<P>(root: P, real_only: bool) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    let mut dataset = BrukerDataset::default();

    let proc_root = root.as_ref();
    let procs_path = proc_root.join("procs");
    let procs_src = fs::read_to_string(procs_path)?;
    let mut parsed = parse_jcamp_dx(&procs_src)?;
    parsed.parameters.set_id("procs");
    dataset.parameters = parsed.parameters.into_owned();
    dataset
        .errors
        .extend(parsed.errors.into_iter().map(Error::from));
    drop(procs_src);

    for dim in 2..=MAX_DIM {
        let next_procs_name = format!("proc{dim}s");
        let next_procs_path = proc_root.join(&next_procs_name);
        let parameters = match fs::read_to_string(next_procs_path) {
            Ok(procs_src) => {
                let mut parsed = parse_jcamp_dx(&procs_src)?;
                dataset
                    .errors
                    .extend(parsed.errors.into_iter().map(Error::from));
                parsed.parameters.set_id(next_procs_name);

                parsed.parameters.into_owned()
            }
            Err(_) => break,
        };
        dataset.data_parameters.push(parameters);
    }
    let dims = 1 + dataset.data_parameters.len() as u8;

    if real_only {
        let mut table = DataTable::new();
        table.set_id("RAW");
        let name = format!("{dims}{}", "r".repeat(dims as usize));
        let raw_path = proc_root.join(&name);
        let raw = read_proc_data(raw_path, &dataset.parameters)?;
        table.insert(name.into(), raw);
        dataset.data_tables.push(table);
    } else {
        let mut table = DataTable::new();
        table.set_id("RAW");
        let char_gen = |bit: u8| if bit & 1 == 1 { 'r' } else { 'i' };
        for i in 0..2_u8.pow(dims as u32) {
            let mut file_name = dims.to_string();
            file_name.extend((0..dims).map(|bit| char_gen(i >> bit)));
            let raw_path = proc_root.join(&file_name);
            let raw = read_proc_data(raw_path, &dataset.parameters)?;
            table.insert(file_name.into(), raw);
        }
        dataset.data_tables.push(table);
    }

    Ok(dataset)
}

/// Passes the correct keys to [`read_bin_to_col`] and returns its result.
///
/// # Errors
///
/// Returns the same errors as [`read_bin_to_col`].
fn read_raw_data<P>(path: P, main_acqus: &ParameterTable) -> Result<Column<'static>>
where
    P: AsRef<Path>,
{
    read_bin_to_col(path, main_acqus, "DTYPA", "BYTORDA")
}

/// Passes the correct keys to [`read_bin_to_col`], undoes the scaling, and
/// returns its result.
///
/// # Errors
///
/// Returns the same errors as [`read_bin_to_col`], and returns an error if
/// the `NC_proc` key is missing from the parameter table.
fn read_proc_data<P>(path: P, main_procs: &ParameterTable) -> Result<Column<'static>>
where
    P: AsRef<Path>,
{
    let mut raw = read_bin_to_col(path, main_procs, "DTYPP", "BYTORDP")?;
    let scale_exponent = main_procs
        .get("NC_proc")
        .and_then(Value::as_i64)
        .ok_or(Error::missing_parameter())?
        .clamp(-31, 31);
    match raw {
        Column::Integer(ref mut inner) if scale_exponent > 0 => {
            inner
                .iter_mut()
                .for_each(|x| *x <<= scale_exponent);
        }
        Column::Integer(ref mut inner) if scale_exponent < 0 => {
            inner
                .iter_mut()
                .for_each(|x| *x >>= -scale_exponent);
        }
        _ => {}
    }

    Ok(raw)
}

/// Reads the binary file and returns its contents as a `Column`.
///
/// The values in the parameter table corresponding to the keys determine the
/// type and endianness to use for interpreting the data. These values must be
/// integers. The data is interpreted as `i32` if the value corresponding to the
/// type key is `0`, and as `f64` otherwise. Similarly, it is interpreted in
/// little endian, if the value corresponding to the endian key is `0`, and in
/// big endian otherwise.
///
/// # Errors
///
/// Returns an error if either of the two keys are not in the parameter table,
/// or if they are not integers. Also returns any [`io`] errors encountered
/// while querying the file's size, opening it, or reading its contents.
fn read_bin_to_col<P>(
    path: P,
    table: &ParameterTable,
    type_key: &str,
    endian_key: &str,
) -> Result<Column<'static>>
where
    P: AsRef<Path>,
{
    let is_integer = table
        .get(type_key)
        .and_then(Value::as_i64)
        .map(|v| v == 0)
        .ok_or(Error::missing_parameter())?;
    let is_little_endian = table
        .get(endian_key)
        .and_then(Value::as_i64)
        .map(|v| v == 0)
        .ok_or(Error::missing_parameter())?;
    let file_len = fs::metadata(path.as_ref())?.len() as usize;
    let count = file_len / if is_integer { 4 } else { 8 };
    let mut file = fs::File::open(path.as_ref())?;
    let raw = match (is_integer, is_little_endian) {
        (true, true) => read_n_i32::<_, LittleEndian>(&mut file, count)?
            .into_iter()
            .map(|x| x as i64)
            .collect::<Column>(),
        (true, false) => read_n_i32::<_, BigEndian>(&mut file, count)?
            .into_iter()
            .map(|x| x as i64)
            .collect::<Column>(),
        (false, true) => Column::from(read_n_f64::<_, LittleEndian>(&mut file, count)?),
        (false, false) => Column::from(read_n_f64::<_, BigEndian>(&mut file, count)?),
    };

    Ok(raw)
}

/// Attempts to read `i32` values from the file and returns them.
///
/// # Errors
///
/// Returns the same errors as [`io::Read::read_exact`].
fn read_n_i32<R, E>(reader: &mut R, n: usize) -> io::Result<Vec<i32>>
where
    R: io::Read,
    E: ByteOrder,
{
    let mut values = vec![0_i32; n];
    reader.read_i32_into::<E>(&mut values)?;

    Ok(values)
}

/// Attempts to read `f64` values from the file and returns them.
///
/// # Errors
///
/// Returns the same errors as [`io::Read::read_exact`].
fn read_n_f64<R, E>(reader: &mut R, n: usize) -> io::Result<Vec<f64>>
where
    R: io::Read,
    E: ByteOrder,
{
    let mut values = vec![0_f64; n];
    reader.read_f64_into::<E>(&mut values)?;

    Ok(values)
}
