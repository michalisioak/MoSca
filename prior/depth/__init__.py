import numpy as np
from abc import ABC, abstractmethod


class DepthModel(ABC):
    @abstractmethod
    def forward(
        self, img_list: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        "returns depth_list, extrinsics, intrinsics"
        pass
