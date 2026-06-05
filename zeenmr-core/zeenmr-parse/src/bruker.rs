//! Bruker NMR data parsing.

mod binary;
pub(crate) use binary::{DataType, Endian, read_bruker_binary};

mod parser;
pub(crate) use parser::Parser;
pub use parser::Value;
