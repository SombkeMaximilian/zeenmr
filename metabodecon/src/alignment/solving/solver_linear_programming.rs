use crate::alignment::SolvingSettings;
use crate::alignment::feature::FeatureMap;
use crate::alignment::solving::{AlignmentProblem, Solver};
use rayon::prelude::*;

#[derive(Debug)]
pub(crate) struct LinearProgramming;

impl Solver for LinearProgramming {
    fn solve(&self, candidate_maps: Vec<FeatureMap>) -> Vec<FeatureMap> {
        candidate_maps
            .into_par_iter()
            .flat_map(|feature_map| {
                let mut problem = AlignmentProblem::default();
                problem.add_assignments(feature_map);

                problem.best_assignment()
            })
            .collect()
    }

    fn settings(&self) -> SolvingSettings {
        SolvingSettings::LinearProgramming
    }
}
