import numpy as np
import torch

from abc import ABC, abstractmethod


@torch.no_grad()
class OpticalFlowModel(ABC):
    @abstractmethod
    def forward(self, img_list: np.ndarray) -> np.ndarray:
        pass
