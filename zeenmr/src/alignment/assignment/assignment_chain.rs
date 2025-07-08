use std::collections::HashSet;

#[derive(Debug)]
pub(crate) struct AssignmentChain {
    layers: Vec<usize>,
    features: Vec<usize>,
}

impl AssignmentChain {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            layers: Vec::with_capacity(capacity),
            features: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, layer: usize, feature: usize) {
        self.layers.push(layer);
        self.features.push(feature);
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (&usize, &usize)> {
        self.layers.iter().zip(self.features.iter())
    }

    pub(crate) fn drop_duplicates(&mut self) {
        let mut pairs = self
            .iter()
            .map(|(layer, feature)| (*layer, *feature))
            .collect::<HashSet<(usize, usize)>>()
            .into_iter()
            .collect::<Vec<_>>();
        pairs.sort_by(|(layer1, _), (layer2, _)| layer1.cmp(layer2));
        self.layers = pairs.iter().map(|(layer, _)| *layer).collect();
        self.features = pairs
            .iter()
            .map(|(_, feature)| *feature)
            .collect();
    }
}
