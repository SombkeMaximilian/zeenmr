use good_lp::{Expression, Variable};

#[derive(Debug)]
pub(crate) struct FeatureVariable {
    feature_a: usize,
    feature_b: usize,
    similarity: f64,
    variable: Variable,
}

impl FeatureVariable {
    pub(crate) fn new(
        feature_a: usize,
        feature_b: usize,
        similarity: f64,
        variable: Variable,
    ) -> Self {
        Self {
            feature_a,
            feature_b,
            similarity,
            variable,
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

    pub(crate) fn variable(&self) -> Variable {
        self.variable
    }

    pub(crate) fn expression(&self) -> Expression {
        self.variable * self.similarity
    }
}
