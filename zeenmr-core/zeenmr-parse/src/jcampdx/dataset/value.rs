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
    /// Version values with two or three parts.
    Version(u32, u32, Option<u32>),
    /// Everything else is treated as a string.
    String(String),
    /// Arrays of values.
    Array(Vec<Self>),
}
