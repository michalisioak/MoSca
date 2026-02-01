from dataclasses import dataclass, field
from typing import Union
from typing_extensions import Literal
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from torch.optim import SparseAdam, Adam
from gsplat.optimizers import SelectiveAdam


@dataclass
class GaussianSplattingConfig:
    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    optimizer: Literal["SparseAdam", "Adam", "SelectiveAdam"] = "SelectiveAdam"

    def get_optimizer(self):
        optim = self.optimizer
        if optim == "SparseAdam":
            return SparseAdam
        elif optim == "SelectiveAdam":
            return SelectiveAdam
        else:
            return Adam
