use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyIOError};
use pyo3::prelude::*;

create_exception!(metabodecon, Error, PyException);
create_exception!(metabodecon, UnexpectedError, Error);
create_exception!(metabodecon, ThreadPoolError, Error);
create_exception!(metabodecon, SerializationError, Error);

create_exception!(metabodecon, SpectrumError, Error);
create_exception!(metabodecon, EmptyData, SpectrumError);
create_exception!(metabodecon, DataLengthMismatch, SpectrumError);
create_exception!(metabodecon, NonUniformSpacing, SpectrumError);
create_exception!(metabodecon, InvalidIntensities, SpectrumError);
create_exception!(metabodecon, InvalidSignalBoundaries, SpectrumError);
create_exception!(metabodecon, MissingMetadata, SpectrumError);
create_exception!(metabodecon, MalformedMetadata, SpectrumError);
create_exception!(metabodecon, MissingData, SpectrumError);
create_exception!(metabodecon, MalformedData, SpectrumError);

create_exception!(metabodecon, DeconvolutionError, Error);
create_exception!(metabodecon, InvalidSmoothingSettings, DeconvolutionError);
create_exception!(metabodecon, InvalidSelectionSettings, DeconvolutionError);
create_exception!(metabodecon, InvalidFittingSettings, DeconvolutionError);
create_exception!(metabodecon, InvalidIgnoreRegion, DeconvolutionError);
create_exception!(metabodecon, NoPeaksDetected, DeconvolutionError);
create_exception!(metabodecon, EmptySignalRegion, DeconvolutionError);
create_exception!(metabodecon, EmptySignalFreeRegion, DeconvolutionError);

pub(crate) struct MetabodeconError(metabodecon::Error);

impl std::fmt::Display for MetabodeconError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<metabodecon::Error> for MetabodeconError {
    fn from(value: metabodecon::Error) -> Self {
        Self(value)
    }
}

impl From<MetabodeconError> for PyErr {
    fn from(value: MetabodeconError) -> PyErr {
        use metabodecon::deconvolution::error::Kind as DecErrKind;
        use metabodecon::spectrum::error::Kind as SpecErrKind;

        match value.0 {
            metabodecon::Error::Spectrum(ref inner) => match inner.kind() {
                SpecErrKind::EmptyData { .. } => EmptyData::new_err(inner.to_string()),
                SpecErrKind::DataLengthMismatch { .. } => {
                    DataLengthMismatch::new_err(inner.to_string())
                }
                SpecErrKind::NonUniformSpacing { .. } => {
                    NonUniformSpacing::new_err(inner.to_string())
                }
                SpecErrKind::InvalidIntensities { .. } => {
                    InvalidIntensities::new_err(inner.to_string())
                }
                SpecErrKind::InvalidSignalBoundaries { .. } => {
                    InvalidSignalBoundaries::new_err(inner.to_string())
                }
                SpecErrKind::MissingMetadata { .. } => MissingMetadata::new_err(inner.to_string()),
                SpecErrKind::MalformedMetadata { .. } => {
                    MalformedMetadata::new_err(inner.to_string())
                }
                SpecErrKind::MissingData { .. } => MissingData::new_err(inner.to_string()),
                SpecErrKind::MalformedData { .. } => MalformedData::new_err(inner.to_string()),
                _ => UnexpectedError::new_err(format!("unexpected error: {}", value)),
            },
            metabodecon::Error::Deconvolution(ref inner) => match inner.kind() {
                DecErrKind::InvalidSmoothingSettings { .. } => {
                    InvalidSmoothingSettings::new_err(inner.to_string())
                }
                DecErrKind::InvalidSelectionSettings { .. } => {
                    InvalidSelectionSettings::new_err(inner.to_string())
                }
                DecErrKind::InvalidFittingSettings { .. } => {
                    InvalidFittingSettings::new_err(inner.to_string())
                }
                DecErrKind::InvalidIgnoreRegion { .. } => {
                    InvalidIgnoreRegion::new_err(inner.to_string())
                }
                DecErrKind::NoPeaksDetected => NoPeaksDetected::new_err(inner.to_string()),
                DecErrKind::EmptySignalRegion => EmptySignalRegion::new_err(inner.to_string()),
                DecErrKind::EmptySignalFreeRegion => {
                    EmptySignalFreeRegion::new_err(inner.to_string())
                }
                _ => UnexpectedError::new_err(format!("unexpected error: {}", value)),
            },
            metabodecon::Error::IoError(inner) => PyIOError::new_err(inner.to_string()),
            _ => UnexpectedError::new_err(format!("unexpected error: {}", value)),
        }
    }
}

pub(crate) fn error_module(py: Python) -> PyResult<Bound<PyModule>> {
    let exceptions = PyModule::new(py, "exceptions")?;
    exceptions.add("Error", py.get_type::<Error>())?;
    exceptions.add("UnexpectedError", py.get_type::<UnexpectedError>())?;
    exceptions.add("ThreadPoolError", py.get_type::<ThreadPoolError>())?;
    exceptions.add("SerializationError", py.get_type::<SerializationError>())?;
    exceptions.add("SpectrumError", py.get_type::<SpectrumError>())?;
    exceptions.add("EmptyData", py.get_type::<EmptyData>())?;
    exceptions.add("DataLengthMismatch", py.get_type::<DataLengthMismatch>())?;
    exceptions.add("NonUniformSpacing", py.get_type::<NonUniformSpacing>())?;
    exceptions.add("InvalidIntensities", py.get_type::<InvalidIntensities>())?;
    exceptions.add(
        "InvalidSignalBoundaries",
        py.get_type::<InvalidSignalBoundaries>(),
    )?;
    exceptions.add("MissingMetadata", py.get_type::<MissingMetadata>())?;
    exceptions.add("MalformedMetadata", py.get_type::<MalformedMetadata>())?;
    exceptions.add("MissingData", py.get_type::<MissingData>())?;
    exceptions.add("MalformedData", py.get_type::<MalformedData>())?;
    exceptions.add("DeconvolutionError", py.get_type::<DeconvolutionError>())?;
    exceptions.add(
        "InvalidSmoothingSettings",
        py.get_type::<InvalidSmoothingSettings>(),
    )?;
    exceptions.add(
        "InvalidSelectionSettings",
        py.get_type::<InvalidSelectionSettings>(),
    )?;
    exceptions.add(
        "InvalidFittingSettings",
        py.get_type::<InvalidFittingSettings>(),
    )?;
    exceptions.add("InvalidIgnoreRegion", py.get_type::<InvalidIgnoreRegion>())?;
    exceptions.add("NoPeaksDetected", py.get_type::<NoPeaksDetected>())?;
    exceptions.add("EmptySignalRegion", py.get_type::<EmptySignalRegion>())?;
    exceptions.add(
        "EmptySignalFreeRegion",
        py.get_type::<EmptySignalFreeRegion>(),
    )?;

    Ok(exceptions)
}
