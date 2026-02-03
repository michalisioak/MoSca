from dataclasses import dataclass


@dataclass
class TrackingConfig:
    total_points: int = 2000
    chunk_size: int = 2000
    max_viz_cnt: int = 512
