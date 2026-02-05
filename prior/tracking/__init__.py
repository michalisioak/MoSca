from dataclasses import dataclass, field

from prior.tracking.identify import TrackIdentificationConfig


@dataclass
class TrackingConfig:
    total_points: int = 2000
    chunk_size: int = 2000
    max_viz_cnt: int = 512
    identify: TrackIdentificationConfig = field(
        default_factory=TrackIdentificationConfig
    )
    vizualize: bool = True
