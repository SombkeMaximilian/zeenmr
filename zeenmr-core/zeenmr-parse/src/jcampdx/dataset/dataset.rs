use crate::jcampdx::{Column, Value};
use std::collections::HashMap;

/// Representation of a JCAMP-DX dataset.
#[derive(Clone, Debug)]
pub struct Dataset {
    /// Parameters of the dataset.
    parameters: HashMap<String, Value>,
    /// Data blocks which are encoded as `Column`s with identifiers.
    data_blocks: Vec<HashMap<String, Column>>,
    /// Nested datasets.
    children: Vec<Self>,
}
