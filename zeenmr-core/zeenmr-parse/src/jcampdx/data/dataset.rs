use crate::jcampdx::{Table, Value};
use std::collections::HashMap;

/// Representation of a JCAMP-DX dataset.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Dataset {
    /// General parameters.
    parameters: HashMap<String, Value>,
    /// Tables in the dataset.
    tables: Vec<Table>,
    /// Nested structures.
    children: Vec<Self>,
}

impl Dataset {
    /// Builder pattern for a new `Dataset`.
    pub(crate) fn builder() -> DatasetBuilder {
        DatasetBuilder::default()
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct DatasetBuilder {
    parameters: HashMap<String, Value>,
    tables: Vec<Table>,
}

impl DatasetBuilder {
    /// Finalizes the `Dataset`.
    pub(crate) fn finalize(self) -> Dataset {
        Dataset {
            parameters: self.parameters,
            tables: self.tables,
            children: Vec::new(),
        }
    }

    /// Inserts a new parameter.
    pub(crate) fn insert(&mut self, key: &str, value: Value) {
        self.parameters.insert(key.to_owned(), value);
    }

    /// Pushes a table onto the stack.
    pub(crate) fn push(&mut self, table: Table) {
        self.tables.push(table);
    }
}
