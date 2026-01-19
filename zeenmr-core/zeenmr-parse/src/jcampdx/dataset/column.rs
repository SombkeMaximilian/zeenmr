use crate::jcampdx::Value;

/// A column in a JCAMP-DX data block.
#[derive(Clone, PartialEq, Debug)]
pub enum Column {
    /// A column of only integer values (`i64`).
    Int(Vec<i64>),
    /// A column of only float values (`f64`).
    Float(Vec<f64>),
    /// A column of mixed values (may still be integer or float only).
    Raw(Vec<Value>),
}
