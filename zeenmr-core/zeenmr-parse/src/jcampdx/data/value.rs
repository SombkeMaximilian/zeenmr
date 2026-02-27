/// Values in JCAMP-DX files.
#[derive(Clone, PartialEq, Debug, Default)]
pub enum Value {
    /// Empty values for unset parameters.
    #[default]
    Empty,
    /// Integer values.
    Integer(i64),
    /// Float values.
    Float(f64),
    /// Everything else is treated as a string.
    String(String),
    /// Arrays of values.
    Array(Vec<Self>),
}
