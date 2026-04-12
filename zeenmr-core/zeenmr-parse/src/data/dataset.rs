use crate::data::{DataTable, ParameterTable};

/// Bundled return type of parsed datasets.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Dataset<'source, E> {
    /// General parameters.
    pub parameters: ParameterTable<'source>,
    /// Raw data.
    pub data_tables: Vec<DataTable<'source>>,
    /// Parameters specific to the raw data.
    pub data_parameters: Vec<ParameterTable<'source>>,
    /// Nested structures.
    pub children: Vec<Self>,
    /// Non-fatal errors during parsing.
    pub errors: Vec<E>,
}

impl<'source, E> Dataset<'source, E> {
    /// Converts this `Dataset` into an owned form with a `'static` lifetime.
    ///
    /// This is useful when you need to store a dataset beyond the lifetime of
    /// the input buffer. Borrowed string data is cloned into `Cow::Owned`.
    /// Array types and child datasets are converted recursively. Everything
    /// else is moved.
    pub fn into_owned(self) -> Dataset<'static, E> {
        Dataset::<'static> {
            parameters: self.parameters.into_owned(),
            data_tables: self
                .data_tables
                .into_iter()
                .map(DataTable::into_owned)
                .collect(),
            data_parameters: self
                .data_parameters
                .into_iter()
                .map(ParameterTable::into_owned)
                .collect(),
            children: self
                .children
                .into_iter()
                .map(Self::into_owned)
                .collect(),
            errors: self.errors,
        }
    }
}
