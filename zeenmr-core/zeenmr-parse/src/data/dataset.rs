use crate::{Table, Value};
use std::collections::HashMap;
use std::error::Error;

/// Representation of a JCAMP-DX dataset.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Dataset<E> {
    /// General parameters.
    parameters: HashMap<String, Value>,
    /// Tables in the dataset.
    tables: Vec<Table>,
    /// Nested structures.
    children: Vec<Self>,
    /// Non-fatal errors during parsing.
    errors: Vec<E>,
}

impl<E> Dataset<E>
where
    E: Error + Default + Send + Sync,
{
    /// Builder pattern for a new `Dataset`.
    pub(crate) fn builder() -> DatasetBuilder<E> {
        DatasetBuilder::default()
    }
}

/// Builder pattern for [`Dataset`].
#[derive(Clone, PartialEq, Debug, Default)]
pub(crate) struct DatasetBuilder<E> {
    /// General parameters.
    parameters: HashMap<String, Value>,
    /// Tables in the dataset.
    tables: Vec<Table>,
    /// Nested structures.
    children: Vec<Dataset<E>>,
    /// Non-fatal errors during parsing.
    errors: Vec<E>,
}

impl<E> DatasetBuilder<E>
where
    E: Error + Default + Send + Sync,
{
    /// Finalizes the `Dataset`.
    pub(crate) fn finalize(self) -> Dataset<E> {
        Dataset {
            parameters: self.parameters,
            tables: self.tables,
            children: self.children,
            errors: self.errors,
        }
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub(crate) fn parameters_contain_key(&self, key: &str) -> bool {
        self.parameters.contains_key(key)
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
    pub(crate) fn push_child(&mut self, child: Dataset<E>) {
        self.children.push(child);
    }

    /// Pushes an error onto the stack.
    pub(crate) fn push_error(&mut self, error: E) {
        self.errors.push(error);
    }

    /// Extends the error stack with the contents of an iterator.
    pub(crate) fn extend_errors<I: IntoIterator<Item = E>>(&mut self, iter: I) {
        self.errors.extend(iter);
    }
}
