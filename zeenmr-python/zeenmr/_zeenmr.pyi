import numpy as np

from typing import Any, Iterator

__version__: str


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
