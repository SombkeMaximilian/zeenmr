use crate::alignment::Alignment;
use crate::alignment::assignment::{
    AssignmentChain, DistanceSimilarityFilter, Filter, FilteringSettings,
};
use crate::alignment::feature::{
    AlignmentStrategy, FeatureLayer, FeatureMap, PairwiseStrategy, ReferenceStrategy, Strategy,
};
use crate::alignment::solving::{LinearProgramming, Solver, SolvingSettings};
use crate::deconvolution::Deconvolution;
use crate::{Result, Settings};
use std::sync::Arc;

/// Alignment pipeline for [`Deconvolution`]s.
///
/// The output of the pipeline is an [`Alignment`] struct containing the aligned
/// [`Deconvolution`]s.
///
/// # Example: Aligning Deconvolutions
///
/// ```
/// use zeenmr::alignment::Aligner;
/// use zeenmr::deconvolution::Deconvoluter;
/// use zeenmr::spectrum::Bruker;
///
/// # fn main() -> zeenmr::Result<()> {
/// // Read all spectra from Bruker TopSpin format directories within the root.
/// let path = "path/to/root";
/// # let path = "../data/bruker/blood";
/// let spectra = Bruker::read_spectra(
///     path,
///     // Experiment number
///     10,
///     // Processing number
///     10,
///     // Signal boundaries
///     (-2.2, 11.8),
/// )?;
///
/// // Deconvolute the spectra in parallel.
/// let deconvoluter = Deconvoluter::default();
/// let deconvolutions = deconvoluter.par_deconvolute_spectra(&spectra)?;
///
/// // Align the deconvolutions using the default settings.
/// let aligner = Aligner::default();
/// let alignment = aligner.align_deconvolutions(&deconvolutions);
/// # Ok(())
/// # }
/// ```
///
/// # Example: Configuring the Aligner
///
/// `Aligner` can be configured with different strategies, filtering settings,
/// and solving methods independently, though currently only a few options are
/// available for the filtering settings, while alignment strategy and solving
/// method have one option each.
///
/// ```
/// use zeenmr::alignment::{
///     Aligner, AlignmentStrategy, FilteringSettings, SimilarityMetric, SolvingSettings,
/// };
///
/// # fn main() -> zeenmr::Result<()> {
/// let aligner = Aligner::new(
///     AlignmentStrategy::Pairwise,
///     FilteringSettings::DistanceSimilarity {
///         similarity_metric: SimilarityMetric::ShapeDistance,
///         max_distance: 0.035,
///         min_similarity: 0.6,
///     },
///     SolvingSettings::LinearProgramming,
/// )?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct Aligner {
    /// Alignment strategy used to generate feature maps.
    strategy: Arc<dyn Strategy>,
    /// Filter used to remove unviable assignments.
    filter: Arc<dyn Filter>,
    /// Solver used to find the optimal assignments.
    solver: Arc<dyn Solver>,
}

impl Default for Aligner {
    fn default() -> Self {
        Self::new(
            AlignmentStrategy::default(),
            FilteringSettings::default(),
            SolvingSettings::default(),
        )
        .unwrap()
    }
}

impl Aligner {
    /// Constructs a new `Aligner` with the specified settings.
    ///
    /// # Errors
    ///
    /// An error is returned if any of the provided settings are invalid, for
    /// example, a `max_distance` in a distance-similarity filter that is
    /// negative.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{
    ///     Aligner, AlignmentStrategy, FilteringSettings, SimilarityMetric, SolvingSettings,
    /// };
    ///
    /// # fn main() -> zeenmr::Result<()> {
    /// let aligner = Aligner::new(
    ///     AlignmentStrategy::Pairwise,
    ///     FilteringSettings::DistanceSimilarity {
    ///         similarity_metric: SimilarityMetric::ShapeDistance,
    ///         max_distance: 0.035,
    ///         min_similarity: 0.6,
    ///     },
    ///     SolvingSettings::LinearProgramming,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(
        alignment_strategy: AlignmentStrategy,
        filtering_settings: FilteringSettings,
        solving_settings: SolvingSettings,
    ) -> Result<Self> {
        alignment_strategy.validate()?;
        filtering_settings.validate()?;
        solving_settings.validate()?;

        let strategy: Arc<dyn Strategy> = match alignment_strategy {
            AlignmentStrategy::Pairwise => Arc::new(PairwiseStrategy::new()),
            AlignmentStrategy::Reference(index) => Arc::new(ReferenceStrategy::new(index)),
        };
        let filter: Arc<dyn Filter> = match filtering_settings {
            FilteringSettings::DistanceSimilarity {
                similarity_metric,
                max_distance,
                min_similarity,
            } => Arc::new(DistanceSimilarityFilter::new(
                similarity_metric,
                max_distance,
                min_similarity,
            )),
        };
        let solver: Arc<dyn Solver> = match solving_settings {
            SolvingSettings::LinearProgramming => Arc::new(LinearProgramming),
        };

        Ok(Self {
            strategy,
            filter,
            solver,
        })
    }

    /// Returns the alignment strategy.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{Aligner, AlignmentStrategy};
    ///
    /// let aligner = Aligner::default();
    ///
    /// match aligner.alignment_strategy() {
    ///     AlignmentStrategy::Reference(ref_spectrum) => {
    ///         assert_eq!(ref_spectrum, 0);
    ///     }
    ///     _ => panic!("Unexpected alignment strategy"),
    /// }
    /// ```
    pub fn alignment_strategy(&self) -> AlignmentStrategy {
        self.strategy.settings()
    }

    /// Returns the filtering settings.
    ///
    /// # Example
    ///
    /// ```
    /// use float_cmp::assert_approx_eq;
    /// use zeenmr::alignment::{Aligner, FilteringSettings, SimilarityMetric};
    ///
    /// let aligner = Aligner::default();
    ///
    /// match aligner.filtering_settings() {
    ///     FilteringSettings::DistanceSimilarity {
    ///         similarity_metric,
    ///         max_distance,
    ///         min_similarity,
    ///     } => {
    ///         match similarity_metric {
    ///             SimilarityMetric::Shape => {}
    ///             _ => panic!("Unexpected similarity metric"),
    ///         }
    ///         assert_approx_eq!(f64, max_distance, 0.025);
    ///         assert_approx_eq!(f64, min_similarity, 0.5);
    ///     }
    ///     _ => panic!("Unexpected filtering settings"),
    /// }
    /// ```
    pub fn filtering_settings(&self) -> FilteringSettings {
        self.filter.settings()
    }

    /// Returns the solving settings.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{Aligner, SolvingSettings};
    ///
    /// let aligner = Aligner::default();
    ///
    /// match aligner.solving_settings() {
    ///     SolvingSettings::LinearProgramming => {}
    ///     _ => panic!("Unexpected solving settings"),
    /// }
    /// ```
    pub fn solving_settings(&self) -> SolvingSettings {
        self.solver.settings()
    }

    /// Sets the alignment strategy.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided strategy is invalid, though
    /// currently, none of the strategies can fail validation.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{Aligner, AlignmentStrategy};
    ///
    /// # fn main() -> zeenmr::Result<()> {
    /// let mut aligner = Aligner::default();
    ///
    /// aligner.set_alignment_strategy(AlignmentStrategy::Pairwise)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_alignment_strategy(&mut self, strategy: AlignmentStrategy) -> Result<()> {
        strategy.validate()?;
        self.strategy = match strategy {
            AlignmentStrategy::Pairwise => Arc::new(PairwiseStrategy::new()),
            AlignmentStrategy::Reference(index) => Arc::new(ReferenceStrategy::new(index)),
        };

        Ok(())
    }

    /// Sets the filtering settings.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided settings are invalid, such as a
    /// negative `max_distance`.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{Aligner, FilteringSettings, SimilarityMetric};
    ///
    /// # fn main() -> zeenmr::Result<()> {
    /// let mut aligner = Aligner::default();
    ///
    /// aligner.set_filtering_settings(FilteringSettings::DistanceSimilarity {
    ///     similarity_metric: SimilarityMetric::ShapeDistance,
    ///     max_distance: 0.035,
    ///     min_similarity: 0.6,
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_filtering_settings(&mut self, settings: FilteringSettings) -> Result<()> {
        settings.validate()?;
        self.filter = match settings {
            FilteringSettings::DistanceSimilarity {
                similarity_metric,
                max_distance,
                min_similarity,
            } => Arc::new(DistanceSimilarityFilter::new(
                similarity_metric,
                max_distance,
                min_similarity,
            )),
        };

        Ok(())
    }

    /// Sets the solving settings.
    ///
    /// # Errors
    ///
    /// An error is returned if the provided settings are invalid, though
    /// currently, none of the solving methods can fail validation.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::{Aligner, SolvingSettings};
    ///
    /// # fn main() -> zeenmr::Result<()> {
    /// let mut aligner = Aligner::default();
    ///
    /// aligner.set_solving_settings(SolvingSettings::LinearProgramming)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_solving_settings(&mut self, settings: SolvingSettings) -> Result<()> {
        settings.validate()?;
        self.solver = match settings {
            SolvingSettings::LinearProgramming => Arc::new(LinearProgramming),
        };

        Ok(())
    }

    /// Aligns the provided [`Deconvolution`]s.
    ///
    /// # Example
    ///
    /// ```
    /// use zeenmr::alignment::Aligner;
    /// use zeenmr::deconvolution::Deconvoluter;
    /// use zeenmr::spectrum::Bruker;
    ///
    /// # fn main() -> zeenmr::Result<()> {
    /// // Read all spectra from Bruker TopSpin format directories within the root.
    /// let path = "path/to/root";
    /// # let path = "../data/bruker/blood";
    /// let spectra = Bruker::read_spectra(
    ///     path,
    ///     // Experiment number
    ///     10,
    ///     // Processing number
    ///     10,
    ///     // Signal boundaries
    ///     (-2.2, 11.8),
    /// )?;
    ///
    /// // Deconvolute the spectra in parallel.
    /// let deconvoluter = Deconvoluter::default();
    /// let deconvolutions = deconvoluter.par_deconvolute_spectra(&spectra)?;
    ///
    /// // Align the deconvolutions using the default settings.
    /// let aligner = Aligner::default();
    /// let alignment = aligner.align_deconvolutions(&deconvolutions);
    /// # Ok(())
    /// # }
    /// ```
    pub fn align_deconvolutions<D: AsRef<Deconvolution>>(&self, deconvolutions: &[D]) -> Alignment {
        let layer_count = deconvolutions.len();
        let mut feature_layers = deconvolutions
            .iter()
            .map(FeatureLayer::from)
            .collect::<Vec<_>>();
        let feature_maps = self
            .strategy
            .generate_maps(&feature_layers, self.filter.as_ref());
        let solution_maps = self.solver.solve(feature_maps);
        let chains = Self::make_chains(solution_maps, layer_count);
        chains.iter().for_each(|chain| {
            let first = chain.iter().next().unwrap();
            let maxp = feature_layers[*first.0].as_ref()[*first.1].maxp();
            chain.iter().skip(1).for_each(|(layer, feature)| {
                feature_layers[*layer].as_mut()[*feature].set_maxp(maxp);
            })
        });
        let deconvolutions = feature_layers
            .into_iter()
            .zip(deconvolutions.iter())
            .map(|(feature_layer, deconvolution)| {
                let deconvolution = deconvolution.as_ref();

                Deconvolution::new(
                    feature_layer
                        .into_iter()
                        .map(Into::into)
                        .collect(),
                    deconvolution.smoothing_settings(),
                    deconvolution.selection_settings(),
                    deconvolution.fitting_settings(),
                    f64::NAN,
                )
            });

        Alignment::new(
            deconvolutions,
            self.strategy.settings(),
            self.filter.settings(),
            self.solver.settings(),
        )
    }

    /// Internal helper function to create assignment chains from solution maps.
    fn make_chains(solution_maps: Vec<FeatureMap>, layer_count: usize) -> Vec<AssignmentChain> {
        let mut chains = Vec::<AssignmentChain>::new();
        solution_maps
            .into_iter()
            .for_each(|solution_map| {
                let i = solution_map.layer_i();
                let j = solution_map.layer_j();
                solution_map
                    .assignments()
                    .iter()
                    .for_each(|assignment| {
                        let a = assignment.feature_a();
                        let b = assignment.feature_b();
                        let position = chains.iter().position(|chain| {
                            chain.iter().any(|(layer, feature)| {
                                *layer == i && *feature == a || *layer == j && *feature == b
                            })
                        });
                        if let Some(position) = position {
                            chains[position].push(i, a);
                            chains[position].push(j, b);
                        } else {
                            let mut chain = AssignmentChain::new(layer_count);
                            chain.push(i, a);
                            chain.push(j, b);
                            chains.push(chain);
                        }
                    });
            });
        chains
            .iter_mut()
            .for_each(|chain| chain.drop_duplicates());

        chains
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::SimilarityMetric;
    use crate::alignment::error::Kind;
    use crate::{Error, assert_send, assert_sync};

    #[test]
    fn thread_safety() {
        assert_send!(Aligner);
        assert_sync!(Aligner);
    }

    #[test]
    fn invalid_filtering_settings() {
        let mut aligner = Aligner::default();
        let negative_distance = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: -0.025,
            min_similarity: 0.5,
        };
        let negative_similarity = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: -0.5,
        };
        let large_similarity = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: 1.5,
        };
        let nan_distance = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: f64::NAN,
            min_similarity: 0.5,
        };
        let inf_distance = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::ShapeDistance,
            max_distance: f64::INFINITY,
            min_similarity: 0.5,
        };
        let neg_inf_distance = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: f64::NEG_INFINITY,
            min_similarity: 0.5,
        };
        let nan_similarity = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: f64::NAN,
        };
        let inf_similarity = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: f64::INFINITY,
        };
        let neg_inf_similarity = FilteringSettings::DistanceSimilarity {
            similarity_metric: SimilarityMetric::Shape,
            max_distance: 0.025,
            min_similarity: f64::NEG_INFINITY,
        };
        let errors = [
            aligner
                .set_filtering_settings(negative_distance)
                .unwrap_err(),
            aligner
                .set_filtering_settings(negative_similarity)
                .unwrap_err(),
            aligner
                .set_filtering_settings(large_similarity)
                .unwrap_err(),
            aligner
                .set_filtering_settings(nan_distance)
                .unwrap_err(),
            aligner
                .set_filtering_settings(inf_distance)
                .unwrap_err(),
            aligner
                .set_filtering_settings(neg_inf_distance)
                .unwrap_err(),
            aligner
                .set_filtering_settings(nan_similarity)
                .unwrap_err(),
            aligner
                .set_filtering_settings(inf_similarity)
                .unwrap_err(),
            aligner
                .set_filtering_settings(neg_inf_similarity)
                .unwrap_err(),
        ];
        let expected_context = [
            negative_distance,
            negative_similarity,
            large_similarity,
            nan_distance,
            inf_distance,
            neg_inf_distance,
            nan_similarity,
            inf_similarity,
            neg_inf_similarity,
        ];
        errors
            .into_iter()
            .zip(expected_context)
            .for_each(|(error, context)| match error {
                Error::Alignment(inner) => match inner.kind() {
                    Kind::InvalidFilteringSettings { settings } => {
                        assert!(FilteringSettings::compare(settings, &context))
                    }
                    _ => panic!("Unexpected kind: {:?}", inner),
                },
                _ => panic!("Unexpected error: {:?}", error),
            });
    }
}
