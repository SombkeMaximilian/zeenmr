//! JCAMP-DX file format parsing.

mod parser;
pub(crate) use parser::Parser;
pub use parser::Value;
