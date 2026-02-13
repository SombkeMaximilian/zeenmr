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

/// Builder pattern for [`Dataset`].
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct DatasetBuilder {
    /// General parameters.
    parameters: HashMap<String, Value>,
    /// Tables in the dataset.
    tables: Vec<Table>,
    /// Nested structures.
    children: Vec<Dataset>,
}

impl DatasetBuilder {
    /// Finalizes the `Dataset`.
    pub(crate) fn finalize(self) -> Dataset {
        Dataset {
            parameters: self.parameters,
            tables: self.tables,
            children: self.children,
        }
    }

    /// Inserts a new parameter.
    pub(crate) fn insert_parameter(&mut self, key: &str, value: Value) {
        self.parameters.insert(key.to_string(), value);
    }

    /// Pushes a `Table` onto the stack.
    pub(crate) fn push_table(&mut self, table: Table) {
        self.tables.push(table);
    }

    /// Pushes a nested `Dataset` onto the stack.
    pub(crate) fn push_child(&mut self, child: Dataset) {
        self.children.push(child);
    }
}
