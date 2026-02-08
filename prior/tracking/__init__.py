from dataclasses import dataclass, field

from prior.tracking.identify import TrackIdentificationConfig


@dataclass
class TrackingConfig:
    total_points: int = 20_000
    chunk_size: int = 2_000
    max_viz_cnt: int = 512
    identify: TrackIdentificationConfig = field(
        default_factory=TrackIdentificationConfig
    )
    vizualize: bool = True
