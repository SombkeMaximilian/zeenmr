#[derive(Debug)]
pub(crate) struct Assignment {
    feature_a: usize,
    feature_b: usize,
    similarity: f64,
}

impl Assignment {
    pub(crate) fn new(feature_a: usize, feature_b: usize, similarity: f64) -> Self {
        Self {
            feature_a,
            feature_b,
            similarity,
        }
    }

    pub(crate) fn feature_a(&self) -> usize {
        self.feature_a
    }

    pub(crate) fn feature_b(&self) -> usize {
        self.feature_b
    }

    pub(crate) fn similarity(&self) -> f64 {
        self.similarity
    }
}
