from prior.depth.depth_anything_v3 import depth_anything_v3
from prior.tracking.bootstap import bootstap
from prior.tracking.cotracker3 import cotracker3


if __name__ == "__main__":
    import tyro

    tyro.extras.subcommand_cli_from_dict(
        {
            "depth_anything_v3": depth_anything_v3,
            "bootstap": bootstap,
            "cotracker3": cotracker3,
        }
    )
