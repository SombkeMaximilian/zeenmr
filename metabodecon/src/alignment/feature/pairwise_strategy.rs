use crate::alignment::assignment::Filter;
use crate::alignment::feature::{AlignmentStrategy, FeatureLayer, FeatureMap, Strategy};

#[derive(Debug)]
pub(crate) struct PairwiseStrategy;

impl Strategy for PairwiseStrategy {
    fn generate_maps(&self, layers: &[FeatureLayer], filter: &dyn Filter) -> Vec<FeatureMap> {
        layers
            .iter()
            .enumerate()
            .zip(
                layers
                    .iter()
                    .enumerate()
                    .skip(1)
                    .chain(layers.iter().enumerate().take(1)),
            )
            .map(
                |((first_index, first_layer), (second_index, second_layer))| {
                    FeatureMap::new(
                        first_index,
                        second_index,
                        filter.filter_assignments(first_layer, second_layer),
                    )
                },
            )
            .collect()
    }

    fn settings(&self) -> AlignmentStrategy {
        AlignmentStrategy::Pairwise
    }
}

impl PairwiseStrategy {
    /// Creates a new [`PairwiseStrategy`].
    pub(crate) fn new() -> Self {
        Self
    }
}
