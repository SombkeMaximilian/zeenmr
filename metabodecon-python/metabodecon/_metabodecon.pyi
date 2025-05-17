import numpy as np

__version__: str


class Deconvoluter:
    def __init__(self) -> None:
        ...

    def set_identity_smoother(self) -> None:
        ...

    def set_moving_average_smoother(self, iterations: int, window_size: int) -> None:
        ...

    def set_detector_only(self) -> None:
        ...

    def set_noise_score_selector(self, threshold: float) -> None:
        ...

    def set_analytical_fitter(self, iterations: int) -> None:
        ...

    def add_ignore_region(self, boundaries: tuple[float, float]) -> None:
        ...

    def clear_ignore_regions(self) -> None:
        ...

    def set_threads(self, threads: int) -> None:
        ...

    def clear_threads(self) -> None:
        ...

    def deconvolute_spectrum(self, spectrum: "Spectrum") -> "Deconvolution":
        ...

    def par_deconvolute_spectrum(self, spectrum: "Spectrum") -> "Deconvolution":
        ...

    def deconvolute_spectra(self, spectra: list["Spectrum"]) -> list["Deconvolution"]:
        ...

    def par_deconvolute_spectra(self, spectra: list["Spectrum"]) -> list["Deconvolution"]:
        ...

    def optimize_settings(self, reference: "Spectrum") -> float:
        ...


class Deconvolution:
    lorentzians: list["Lorentzian"]
    mse: float

    def superposition(self, x: float) -> float:
        ...

    def superposition_vec(self, x: np.ndarray) -> np.ndarray:
        ...

    def par_superposition_vec(self, x: np.ndarray) -> np.ndarray:
        ...

    def write_json(self, path: str) -> None:
        ...

    @staticmethod
    def read_json(path: str) -> "Deconvolution":
        ...

    def write_bin(self, path: str) -> None:
        ...

    @staticmethod
    def read_bin(path: str) -> "Deconvolution":
        ...


class Lorentzian:
    sf: float
    hw: float
    maxp: float

    def __init__(self, sf: float, hw: float, maxp: float) -> None:
        ...

    @staticmethod
    def from_transformed(sfhw: float, hw2: float, maxp: float) -> "Lorentzian":
        ...

    def evaluate(self, x: float) -> float:
        ...

    def evaluate_vec(self, x: np.ndarray) -> np.ndarray:
        ...

    def integral(self) -> float:
        ...

    @staticmethod
    def superposition(x: float, lorentzians: list["Lorentzian"]) -> float:
        ...

    @staticmethod
    def superposition_vec(x: np.ndarray, lorentzians: list["Lorentzian"]) -> np.ndarray:
        ...

    @staticmethod
    def par_superposition_vec(x: np.ndarray, lorentzians: list["Lorentzian"]) -> np.ndarray:
        ...


class Spectrum:
    chemical_shifts: np.ndarray
    intensities: np.ndarray
    signal_boundaries: tuple[float, float]
    nucleus: str
    frequency: float
    reference_compound: dict

    def __init__(self, chemical_shifts: np.ndarray, intensities: np.ndarray,
                 signal_boundaries: tuple[float, float]) -> None:
        ...

    @staticmethod
    def read_bruker(path: str, experiment: int, processing: int,
                    signal_boundaries: tuple[float, float]) -> "Spectrum":
        ...

    @staticmethod
    def read_bruker_set(path: str, experiment: int, processing: int,
                        signal_boundaries: tuple[float, float]) -> list[
        "Spectrum"]:
        ...

    @staticmethod
    def read_jcampdx(path: str, signal_boundaries: tuple[float, float]) -> "Spectrum":
        ...

    @staticmethod
    def read_jcampdx_set(path: str, signal_boundaries: tuple[float, float]) -> list["Spectrum"]:
        ...

    def write_json(self, path: str) -> None:
        ...

    @staticmethod
    def read_json(path: str) -> "Spectrum":
        ...

    def write_bin(self, path: str) -> None:
        ...

    @staticmethod
    def read_bin(path: str) -> "Spectrum":
        ...
