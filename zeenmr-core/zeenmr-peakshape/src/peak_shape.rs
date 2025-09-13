use crate::Evaluate;

/// Marker trait for types that can represent peak shapes in spectral data.
pub trait PeakShape: Evaluate {}

impl<E> PeakShape for E where E: Evaluate {}
