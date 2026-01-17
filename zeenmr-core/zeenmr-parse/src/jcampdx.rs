//! JCAMP-DX file format parsing.

pub mod error;

pub(crate) mod block_format;

pub(crate) mod decoding;

mod lexer;
pub(crate) use lexer::Token;

mod parser;
pub(crate) use parser::Parser;
pub use parser::Value;
