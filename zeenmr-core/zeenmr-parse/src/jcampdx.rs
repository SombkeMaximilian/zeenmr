//! JCAMP-DX file format parsing.

pub mod error;

pub(crate) mod block_format;

pub(crate) mod decoding;

mod lexer;
pub(crate) use lexer::Token;

mod dataset;
pub use dataset::{Column, Dataset, Value};

mod parser;
pub(crate) use parser::Parser;
