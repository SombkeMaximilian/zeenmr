use crate::bindings::{Alignment, Deconvolution};
use crate::error::{ThreadPoolError, ZeeNmrError};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use zeenmr::alignment;

#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct Aligner {
    inner: alignment::Aligner,
    threads: Option<Arc<ThreadPool>>,
}

#[pymethods]
impl Aligner {
    #[new]
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn set_reference_alignment(&mut self, reference: usize) -> PyResult<()> {
        match self
            .inner
            .set_alignment_strategy(alignment::AlignmentStrategy::Reference(reference))
        {
            Ok(_) => Ok(()),
            Err(error) => Err(ZeeNmrError::from(error).into()),
        }
    }

    pub(crate) fn set_pairwise_alignment(&mut self) {
        self.inner
            .set_alignment_strategy(alignment::AlignmentStrategy::Pairwise)
            .unwrap();
    }

    pub(crate) fn set_distance_similarity_filter(
        &mut self,
        max_distance: f64,
        min_similarity: f64,
    ) -> PyResult<()> {
        match self
            .inner
            .set_filtering_settings(alignment::FilteringSettings::DistanceSimilarity {
                similarity_metric: alignment::SimilarityMetric::ShapeDistance,
                max_distance,
                min_similarity,
            }) {
            Ok(_) => Ok(()),
            Err(error) => Err(ZeeNmrError::from(error).into()),
        }
    }

    pub(crate) fn set_linear_programming_solver(&mut self) {
        self.inner
            .set_solving_settings(alignment::SolvingSettings::LinearProgramming)
            .unwrap();
    }

    pub(crate) fn set_threads(&mut self, threads: usize) -> PyResult<()> {
        if threads <= 1 {
            return Err(PyValueError::new_err(
                "number of threads must be greater than 1",
            ));
        } else {
            let thread_pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| ThreadPoolError::new_err(error.to_string()))?;
            self.threads = Some(Arc::new(thread_pool));
        }

        Ok(())
    }

    pub(crate) fn clear_threads(&mut self) {
        self.threads = None;
    }

    pub(crate) fn align_deconvolutions(&self, deconvolutions: Vec<Deconvolution>) -> Alignment {
        match &self.threads {
            Some(threads) => threads.install(|| {
                self.inner
                    .align_deconvolutions(&deconvolutions)
                    .into()
            }),
            None => self
                .inner
                .align_deconvolutions(&deconvolutions)
                .into(),
        }
    }
}
