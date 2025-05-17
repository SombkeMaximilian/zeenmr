class Error(Exception):
    """
    Base class for all Metabodecon errors.
    """

    ...


class UnexpectedError(Error):
    """
    Exception raised when an unexpected error occurs.
    """

    ...


class SerializationError(Error):
    """
    Exception raised when serialization or deserialization fails.
    """

    ...


class SpectrumError(Error):
    """
    An exception raised for errors in the Spectrum class.
    """

    ...


class EmptyData(SpectrumError):
    """
    Input data is empty.
    """

    ...


class DataLengthMismatch(SpectrumError):
    """
    Input data lengths do not match.
    """

    ...


class NonUniformSpacing(SpectrumError):
    """
    Chemical shifts are not uniformly spaced.
    """

    ...


class InvalidIntensities(SpectrumError):
    """Intensities contain invalid values."""

    ...


class InvalidSignalBoundaries(SpectrumError):
    """
    Signal boundaries are invalid.
    """

    ...


class MissingMetadata(SpectrumError):
    """
    Metadata is missing from NMR format-related file.
    """

    ...


class MalformedMetadata(SpectrumError):
    """
    Metadata in some NMR format-related file is malformed.
    """

    ...


class MissingData(SpectrumError):
    """
    Some NMR format-related file contains no data.
    """

    ...


class MalformedData(SpectrumError):
    """
    Data in some NMR format-related file is malformed.
    """

    ...


class DeconvolutionError(Error):
    """
    An exception raised for errors during the deconvolution process.
    """

    ...


class InvalidSmoothingSettings(DeconvolutionError):
    """
    Smoothing settings are invalid.
    """

    ...


class InvalidSelectionSettings(DeconvolutionError):
    """
    Peak selection settings are invalid.
    """

    ...


class InvalidFittingSettings(DeconvolutionError):
    """
    Peak fitting settings are invalid.
    """

    ...


class InvalidIgnoreRegion(DeconvolutionError):
    """
    Ignore region boundaries are invalid.
    """

    ...


class NoPeaksDetected(DeconvolutionError):
    """
    No peaks were detected in the spectrum.
    """

    ...


class EmptySignalRegion(DeconvolutionError):
    """
    Signal region contains no peaks.
    """

    ...


class EmptySignalFreeRegion(DeconvolutionError):
    """
    Signal-free region contains no peaks.
    """

    ...


class AlignmentError(Error):
    """
    An exception raised for errors during the alignment process.
    """

    ...


class InvalidAlignmentStrategy(AlignmentError):
    """
    An exception raised when an invalid alignment strategy is provided.
    """

    ...


class InvalidFilteringSettings(AlignmentError):
    """
    An exception raised when invalid filtering settings are provided.
    """

    ...


class InvalidSolvingSettings(AlignmentError):
    """
    An exception raised when invalid solving settings are provided.
    """

    ...
