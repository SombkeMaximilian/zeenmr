//! Bruker NMR data parsing.

mod binary;
pub(crate) use binary::BrukerBinaryReader;

mod parser;
pub(crate) use parser::Parser;
pub use parser::Value;
