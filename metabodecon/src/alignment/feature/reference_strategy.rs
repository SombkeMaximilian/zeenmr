use crate::alignment::assignment::Filter;
use crate::alignment::feature::{AlignmentStrategy, FeatureLayer, FeatureMap, Strategy};

#[derive(Debug)]
pub(crate) struct ReferenceStrategy {
    reference_index: usize,
}

impl Strategy for ReferenceStrategy {
    fn generate_maps(&self, layers: &[FeatureLayer], filter: &dyn Filter) -> Vec<FeatureMap> {
        layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                if index != self.reference_index {
                    Some(FeatureMap::new(
                        self.reference_index,
                        index,
                        filter.filter_assignments(&layers[self.reference_index], layer),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    fn settings(&self) -> AlignmentStrategy {
        AlignmentStrategy::Reference(self.reference_index)
    }
}

impl ReferenceStrategy {
    /// Creates a new [`ReferenceStrategy`] with the given reference index.
    pub(crate) fn new(reference_index: usize) -> Self {
        Self { reference_index }
    }
}
