import numpy as np
from abc import ABC, abstractmethod


class TrackerModel(ABC):
    @abstractmethod
    def forward(self) -> np.ndarray:
        pass
