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
/// # Errors
///
/// Returns an error if
/// - the `acqus` file is missing, or
/// - the raw binary file is missing (`fid` or `ser`), or
/// - the `procs` file is missing, or
/// - the processed binary file is missing (`1r` and/or `1i`, etc.)
pub fn read_bruker_dir<P>(root: P, exp_id: u32, proc_id: u32) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    todo!()
}

/// Reads raw data from a Bruker experiment directory.
///
/// # Errors
///
/// Returns an error if
/// - the `acqus` file is missing, or
/// - the raw binary file is missing (`fid` or `ser`)
pub fn read_bruker_exp<P>(root: P) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    let mut dataset = BrukerDataset::default();

    let exp_root = root.as_ref();
    let acqus_path = exp_root.join("acqus");
    let acqus_src = fs::read_to_string(acqus_path)?;
    let parsed = parse_jcamp_dx(&acqus_src)?;
    dataset.parameters = parsed.parameters.into_owned();
    dataset
        .errors
        .extend(parsed.errors.into_iter().map(Error::from));
    drop(acqus_src);

    for dim in 2..=MAX_DIM {
        let next_acqus_path = exp_root.join(format!("acqu{dim}s"));
        let parameters = match fs::read_to_string(next_acqus_path) {
            Ok(acqus_src) => {
                let parsed = parse_jcamp_dx(&acqus_src)?;
                dataset
                    .errors
                    .extend(parsed.errors.into_iter().map(Error::from));

                parsed.parameters.into_owned()
            }
            Err(_) => break,
        };
        dataset.data_parameters.push(parameters);
    }

    let raw_path = if dataset.data_parameters.is_empty() {
        exp_root.join("fid")
    } else {
        exp_root.join("ser")
    };
    let raw = read_raw_data(raw_path, &dataset.parameters)?;
    let mut table = DataTable::new();
    table.insert("RAW".into(), raw);
    dataset.data_tables.push(table);

    Ok(dataset)
}

/// Reads processed data from a Bruker directory.
///
/// # Errors
///
/// Returns an error if
/// - the `procs` file is missing, or
/// - the processed binary file is missing (`1r` and/or `1i`, etc.)
pub fn read_bruker_proc<P>(root: P, dims: usize, real_only: bool) -> Result<BrukerDataset>
where
    P: AsRef<Path>,
{
    let mut dataset = BrukerDataset::default();

    let proc_root = root.as_ref();
    let procs_path = proc_root.join("procs");
    let procs_src = fs::read_to_string(procs_path)?;
    let parsed = parse_jcamp_dx(&procs_src)?;
    dataset.parameters = parsed.parameters.into_owned();
    dataset
        .errors
        .extend(parsed.errors.into_iter().map(Error::from));
    drop(procs_src);

    for dim in 2..=dims {
        let next_procs_path = proc_root.join(format!("proc{dim}s"));
        let parameters = match fs::read_to_string(next_procs_path) {
            Ok(procs_src) => {
                let parsed = parse_jcamp_dx(&procs_src)?;
                dataset.errors.extend(parsed.errors.into_iter().map(Error::from));

                parsed.parameters.into_owned()
            }
            Err(_) => break,
        };
        dataset.data_parameters.push(parameters);
    }

    if real_only {
        let name = format!("{dims}{}", "r".repeat(dims));
        let raw_path = proc_root.join(&name);
        let raw = read_proc_data(raw_path, &dataset.parameters)?;
        let mut table = DataTable::new();
        table.insert(name.into(), raw);
        dataset.data_tables.push(table);
    } else {
        todo!()
    }

    Ok(dataset)
}

fn read_raw_data<P>(path: P, main_acqus: &ParameterTable) -> Result<Column<'static>>
where
    P: AsRef<Path>,
{
    read_data_to_column(path, main_acqus, "DTYPA", "BYTORDA")
}

fn read_proc_data<P>(path: P, main_procs: &ParameterTable) -> Result<Column<'static>>
where
    P: AsRef<Path>,
{
    let mut raw = read_data_to_column(path, main_procs, "DTYPP", "BYTORDP")?;
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

fn read_data_to_column<P>(
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
    fn read_exp() {
        let path = workspace_dir()
            .join("data")
            .join("bruker")
            .join("blood")
            .join("blood_01")
            .join("10");
        let dataset = read_bruker_exp(path).unwrap();
        println!("{:#?}", dataset.parameters);
    }
}
