use byteorder::{ByteOrder, ReadBytesExt};
use std::io::{self, Read};

/// Attempts to read `i32` values from the file and returns them.
///
/// # Errors
///
/// Returns the same errors as [`Read::read_exact`].
fn read_n_i32<R, E>(reader: &mut R, n: usize) -> io::Result<Vec<i32>>
where
    R: Read,
    E: ByteOrder,
{
    let mut values = vec![0_i32; n];
    reader.read_i32_into::<E>(&mut values)?;

    Ok(values)
}

/// Attempts to read complex `i32` values from the file and returns them.
///
/// The first returned vector contains the real values, the second contains the
/// imaginary values.
///
/// # Errors
///
/// Returns the same errors as [`Read::read_exact`].
fn read_n_i32_complex<R, E>(reader: &mut R, n: usize) -> io::Result<(Vec<i32>, Vec<i32>)>
where
    R: Read,
    E: ByteOrder,
{
    let mut values = vec![0_i32; n];
    reader.read_i32_into::<E>(&mut values)?;

    Ok(values
        .chunks_exact(2)
        .map(|c| (c[0], c[1]))
        .unzip())
}

/// Attempts to read `f64` values from the file and returns them.
///
/// # Errors
///
/// Returns the same errors as [`Read::read_exact`].
fn read_n_f64<R, E>(reader: &mut R, n: usize) -> io::Result<Vec<f64>>
where
    R: Read,
    E: ByteOrder,
{
    let mut values = vec![0_f64; n];
    reader.read_f64_into::<E>(&mut values)?;

    Ok(values)
}

/// Attempts to read complex `f64` values from the file and returns them.
///
/// The first returned vector contains the real values, the second contains
/// the imaginary values.
///
/// # Errors
///
/// Returns the same errors as [`Read::read_exact`].
fn read_f64_complex<R, E>(reader: &mut R, n: usize) -> io::Result<(Vec<f64>, Vec<f64>)>
where
    R: Read,
    E: ByteOrder,
{
    let mut complex = vec![0_f64; n];
    reader.read_f64_into::<E>(&mut complex)?;

    Ok(complex
        .chunks_exact(2)
        .map(|c| (c[0], c[1]))
        .unzip())
}

/// Undoes scaling as specified by the `NC_proc` parameter.
///
/// This should only be used for processed data. Raw FIDs are not rescaled.
fn undo_scaling(data: &mut [i32], exp: i32) {
    if exp > 0 {
        data.iter_mut().for_each(|x| *x <<= exp)
    } else if exp < 0 {
        data.iter_mut().for_each(|x| *x >>= -exp)
    }
}
