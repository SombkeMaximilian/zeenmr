import numpy as np

from typing import Any, ClassVar, Iterator, final

__version__: str


@final
class DataType:
    U8: ClassVar[DataType]
    U16: ClassVar[DataType]
    U32: ClassVar[DataType]
    U64: ClassVar[DataType]
    I8: ClassVar[DataType]
    I16: ClassVar[DataType]
    I32: ClassVar[DataType]
    I64: ClassVar[DataType]
    F32: ClassVar[DataType]
    F64: ClassVar[DataType]
    C32: ClassVar[DataType]
    C64: ClassVar[DataType]
    DEFAULT: ClassVar[DataType]

    @property
    def type_str(self) -> str:
        ...

    @property
    def numpy_str(self) -> str:
        ...


class ShiftReference:
    def __init__(self, shift: float | int, freq: float | int) -> None:
        ...

    @staticmethod
    def from_shift(shift: float | int) -> ShiftReference:
        ...

    @staticmethod
    def from_freq(freq: float | int) -> ShiftReference:
        ...

    def to_f32(self) -> ShiftReference:
        ...

    def to_f64(self) -> ShiftReference:
        ...

    @property
    def shift(self) -> float:
        ...

    @property
    def freq(self) -> float:
        ...

    @property
    def data_type(self) -> str:
        ...

    def __repr__(self):
        ...

    def __reduce__(self):
        ...

    def __eq__(self, other):
        ...


class FrequencyAxis:
    def __init__(self, range: tuple[float | int, float | int], larmor: float | int, reference: ShiftReference) -> None:
        ...

    def to_f32(self) -> ShiftReference:
        ...

    def to_f64(self) -> ShiftReference:
        ...

    @property
    def range(self) -> tuple[float, float]:
        ...

    @property
    def larmor(self) -> float:
        ...

    @property
    def reference(self) -> ShiftReference:
        ...

    def __repr__(self):
        ...

    def __reduce__(self):
        ...

    def __eq__(self, other):
        ...


class Array:
    def __init__(self, array: np.ndarray) -> None:
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def rank(self) -> int:
        ...

    @property
    def data_type(self) -> str:
        ...

    @property
    def data(self) -> np.ndarray:
        ...

    def layout(self) -> dict:
        ...

    @property
    def __array_interface__(self):
        ...

    def __array__(self):
        ...

    def __len__(self):
        ...

    def __getitem__(self, item) -> Any:
        ...

    def __iter__(self) -> Iterator[Any]:
        ...

    def __repr__(self):
        ...

    def __reduce__(self):
        ...

    def __eq__(self, other):
        ...
