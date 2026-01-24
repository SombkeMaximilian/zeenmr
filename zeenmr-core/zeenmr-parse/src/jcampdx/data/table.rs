use crate::jcampdx::Column;

/// Data table in a JCAMP-DX dataset.
///
/// Not optimized for downstream computations.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Table {
    /// Identifier of the table.
    id: String,
    /// Columns in the table.
    columns: Vec<Column>,
}

impl Table {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn set_id<T: Into<String>>(&mut self, id: T) {
        self.id = id.into();
    }

    pub(crate) fn push<T: Into<Column>>(&mut self, column: T) {
        self.columns.push(column.into());
    }
}
