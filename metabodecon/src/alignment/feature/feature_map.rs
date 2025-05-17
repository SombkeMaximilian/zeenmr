use crate::alignment::assignment::Assignment;

#[derive(Debug)]
pub(crate) struct FeatureMap {
    layer_i: usize,
    layer_j: usize,
    assignments: Vec<Assignment>,
}

impl FeatureMap {
    pub(crate) fn new(layer_a: usize, layer_b: usize, assignments: Vec<Assignment>) -> Self {
        Self {
            layer_i: layer_a,
            layer_j: layer_b,
            assignments,
        }
    }

    pub(crate) fn layer_i(&self) -> usize {
        self.layer_i
    }

    pub(crate) fn layer_j(&self) -> usize {
        self.layer_j
    }

    pub(crate) fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }
}
