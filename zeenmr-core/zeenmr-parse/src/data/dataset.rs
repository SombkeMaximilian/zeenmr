use crate::data::{DataTable, ParameterTable};

/// Bundled return type of parsed datasets.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Dataset<'source, E> {
    /// General parameters.
    pub parameters: ParameterTable<'source>,
    /// Raw data.
    pub data_tables: Vec<DataTable<'source>>,
    /// Nested structures.
    pub children: Vec<Self>,
    /// Non-fatal errors during parsing.
    pub errors: Vec<E>,
}
