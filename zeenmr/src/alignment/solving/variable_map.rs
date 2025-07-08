use crate::alignment::solving::FeatureVariable;

#[derive(Debug)]
pub(crate) struct VariableMap {
    layer_i: usize,
    layer_j: usize,
    variables: Vec<FeatureVariable>,
}

impl VariableMap {
    pub(crate) fn new(layer_i: usize, layer_j: usize, variables: Vec<FeatureVariable>) -> Self {
        Self {
            layer_i,
            layer_j,
            variables,
        }
    }

    pub(crate) fn layer_i(&self) -> usize {
        self.layer_i
    }

    pub(crate) fn layer_j(&self) -> usize {
        self.layer_j
    }

    pub(crate) fn variables(&self) -> &[FeatureVariable] {
        &self.variables
    }
}
