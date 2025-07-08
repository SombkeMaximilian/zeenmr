use crate::Result;
use crate::spectrum::error::{Error, Kind};
use regex::Regex;
use std::path::Path;

/// Internal helper function to extract a single capture group from a regex
/// match and parse it into a desired type.
pub(crate) fn extract_capture<T, P>(
    regex: &Regex,
    name: &str,
    text: &str,
    path: P,
    key: &str,
) -> Result<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    P: AsRef<Path>,
{
    let missing_error = || {
        Error::new(Kind::MissingMetadata {
            path: path.as_ref().to_path_buf(),
            key: key.to_string(),
        })
    };
    let malformed_error = |error: <T as std::str::FromStr>::Err| {
        Error::new(Kind::MalformedMetadata {
            path: path.as_ref().to_path_buf(),
            key: key.to_string(),
            details: error.to_string(),
        })
    };

    let result = regex
        .captures(text)
        .ok_or_else(missing_error)?
        .name(name)
        .ok_or_else(missing_error)?
        .as_str()
        .parse::<T>()
        .map_err(malformed_error)?;

    Ok(result)
}

/// Internal helper function to extract a single capture group from a regex
/// match, which represents a list of values, and parse it into a Vec of the
/// desired type.
pub(crate) fn extract_row<T, P>(
    regex: &Regex,
    name: &str,
    text: &str,
    path: P,
    key: &str,
) -> Result<Vec<T>>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
    P: AsRef<Path>,
{
    let malformed_error = |error: <T as std::str::FromStr>::Err| {
        Error::new(Kind::MalformedMetadata {
            path: path.as_ref().to_path_buf(),
            key: key.to_string(),
            details: error.to_string(),
        })
    };

    let raw = extract_capture::<String, _>(regex, name, text, &path, key)?;
    let row = raw
        .split(",")
        .map(|value| {
            value
                .trim()
                .parse::<T>()
                .map_err(|error| malformed_error(error).into())
        })
        .collect::<Result<Vec<T>>>()?;

    Ok(row)
}
