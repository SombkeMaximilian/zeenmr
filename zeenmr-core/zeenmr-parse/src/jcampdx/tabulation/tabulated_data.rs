use crate::jcampdx::tabulation::error::Error;
use crate::jcampdx::{ChildParserExit, Column, RawColumn, Table, Value};

/// Tabulated data block.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlock {
    /// Exit status of the parser (end of input or header key).
    pub(crate) exit: ChildParserExit,
    /// Tabulated data.
    pub(crate) table: Table,
    /// Non-fatal errors during tabulation.
    pub(crate) errors: Vec<Error>,
}

/// Constructs a collection of columns by pushing to them in a cycle.
#[derive(Clone, PartialEq, Debug, Default)]
struct ColumnCycle {
    /// Columns being constructed.
    columns: Vec<Column>,
    /// Number of columns.
    capacity: usize,
    /// Current column to push to.
    current: usize,
}

impl<T> FromIterator<T> for ColumnCycle
where
    T: Into<String>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let columns = iter
            .into_iter()
            .map(Into::into)
            .enumerate()
            .map(|(column_index, identifier)| {
                Column::Integer(RawColumn::<i64> {
                    id: identifier,
                    values: Vec::new(),
                })
            })
            .collect::<Vec<Column>>();
        let capacity = columns.len();

        Self {
            columns,
            capacity,
            current: 0,
        }
    }
}

impl ColumnCycle {
    fn cycle(&mut self) {
        self.current = (self.current + 1) % self.capacity;
    }

    fn push_i64(&mut self, value: i64) {
        self.columns[self.current].push_i64(value);
        self.cycle();
    }

    fn push_f64(&mut self, value: f64) {
        self.columns[self.current].push_f64(value);
        self.cycle();
    }

    fn push_string<T: Into<String>>(&mut self, value: T) {
        self.columns[self.current].push_string(value);
        self.cycle();
    }
}

/// Builder pattern for [`TabulatedBlock`].
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct TabulatedBlockBuilder {
    exit: ChildParserExit,
    title: String,
    errors: Vec<Error>,
}
