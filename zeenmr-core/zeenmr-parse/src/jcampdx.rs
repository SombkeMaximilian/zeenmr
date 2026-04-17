//! JCAMP-DX file format parsing.

mod block_format;
mod decoding;
mod lexer;
mod parser;
mod tabulation;

pub mod error;

use crate::data::Dataset;
use error::{Error, Result};
use parser::Parser;

/// Dataset that can be parsed from JCAMP-DX files.
pub type JcampDxDataset<'source> = Dataset<'source, Error>;

/// Parses a JCAMP-DX source string into a dataset.
pub fn parse_jcamp_dx(source: &str) -> Result<JcampDxDataset<'_>> {
    // this allows changes to the parser without being a breaking change
    Parser::from(source).parse_source()
}
