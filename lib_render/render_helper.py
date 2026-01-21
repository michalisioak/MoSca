import os, sys, os.path as osp
import torch

sys.path.append(osp.dirname(osp.abspath(__file__)))
try:
    GS_BACKEND = os.environ["GS_BACKEND"]
except:
    # GS_BACKEND = "native"
    print(f"No GS_BACKEND env var specified, for now use native_add3 backend")
    GS_BACKEND = "native_add3"
GS_BACKEND = GS_BACKEND.lower()
print(f"GS_BACKEND: {GS_BACKEND.lower()}")

if GS_BACKEND == "native":
    from gauspl_renderer_native import render_cam_pcl
elif GS_BACKEND == "gof":
    from gauspl_renderer_gof import render_cam_pcl
elif GS_BACKEND == "native_add3":
    from gauspl_renderer_native_add3 import render_cam_pcl
else:
    raise ValueError(f"Unknown GS_BACKEND: {GS_BACKEND.lower()}")
from sh_utils import RGB2SH, SH2RGB


def render(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
):
    # * Core render interface
    # prepare gs5 param in world system
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

    render_dict = render_cam_pcl(
        mu_cam,
        fr_cam,
        s,
        o,
        color_feat=sph,
        H=H,
        W= W,
        CAM_K=K,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        add_buffer=add_buffer,
    )
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict


def fast_bg_compose_render(bg_cache_dict, render_dict, bg_color=[1.0, 1.0, 1.0]):
    assert GS_BACKEND == "native", "GOF does not support this now"

    # manually compose the fg
    # ! warning, be careful when use the visibility masks .etc, watch the len
    fg_rgb, bg_rgb = render_dict["rgb"], bg_cache_dict["rgb"]
    fg_alpha, bg_alpha = render_dict["alpha"], bg_cache_dict["alpha"]
    fg_dep, bg_dep = render_dict["dep"], bg_cache_dict["dep"]
    _fg_alp = torch.clamp(fg_alpha, 1e-8, 1.0)
    _bg_alp = torch.clamp(bg_alpha, 1e-8, 1.0)
    fg_dep_corr = fg_dep / _fg_alp
    bg_dep_corr = bg_dep / _bg_alp
    fg_in_front = (fg_dep_corr < bg_dep_corr).float()
    # compose alpha
    alpha_fg_front_compose = fg_alpha + (1.0 - fg_alpha) * bg_alpha
    alpha_fg_behind_compose = bg_alpha + (1.0 - bg_alpha) * fg_alpha
    alpha_composed = alpha_fg_front_compose * fg_in_front + alpha_fg_behind_compose * (
        1.0 - fg_in_front
)
    alpha_composed = torch.clamp(alpha_composed, 0.0, 1.0)
    # compose rgb
    bg_color = torch.as_tensor(bg_color, device=fg_rgb.device, dtype=fg_rgb.dtype)
    rgb_fg_front_compose = (
        fg_rgb * fg_alpha
        + bg_rgb * (1.0 - fg_alpha) * bg_alpha
        + (1.0 - fg_alpha) * (1.0 - bg_alpha) * bg_color[:, None, None]
    )
    rgb_fg_behind_compose = (
        bg_rgb * bg_alpha
        + fg_rgb * (1.0 - bg_alpha) * fg_alpha
        + (1.0 - bg_alpha) * (1.0 - fg_alpha) * bg_color[:, None, None]
    )
    rgb_composed = rgb_fg_front_compose * fg_in_front + rgb_fg_behind_compose * (
        1.0 - fg_in_front
    )
    # compose dep
    dep_fg_front_compose = (
        fg_dep_corr * fg_alpha + bg_dep_corr * (1.0 - fg_alpha) * bg_alpha
    )
    dep_fg_behind_compose = (
        bg_dep_corr * bg_alpha + fg_dep_corr * (1.0 - bg_alpha) * fg_alpha
    )
    dep_composed = dep_fg_front_compose * fg_in_front + dep_fg_behind_compose * (
        1.0 - fg_in_front
    )
    return {
        "rgb": rgb_composed,
        "dep": dep_composed,
        "alpha": alpha_composed,
        "visibility_filter": render_dict["visibility_filter"],
        "viewspace_points": render_dict["viewspace_points"],
        "radii": render_dict["radii"],
        "dyn_rgb": fg_rgb,
        "dyn_dep": fg_dep,
        "dyn_alpha": fg_alpha,
    }

def render_new(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
):
    if torch.is_tensor(gs_param[0]):
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    from pytorch3d.transforms import matrix_to_quaternion
    quats = matrix_to_quaternion(fr)  # shape [N, 4]

    viewmats = T_cw.unsqueeze(0)  # add batch dimension [1, 4, 4]

    Ks = K.unsqueeze(0)  # shape [1, 3, 3]
    
    from gsplat import rasterization
    o = o[...,0]
    N = len(sph)
    shs_view = sph.reshape(N, -1, 3)  # N, Deg, Channels
    _deg = shs_view.shape[1]
    if _deg == 1:
        active_sph_order = 0
    elif _deg == 4:
        active_sph_order = 1
    elif _deg == 9:
        active_sph_order = 2
    elif _deg == 16:
        active_sph_order = 3
    else:
        raise ValueError(f"Unexpected SH degree: {_deg}")
    shs_view=shs_view.permute(0, 1, 2)
    rendered_image, rendered_alpha, aux_dict = rasterization(
        means=mu,  # World coordinates
        quats=quats,  # Quaternions in world frame
        scales=s,
        opacities=o,
        colors=shs_view,
        viewmats=viewmats,  # World-to-camera
        Ks=Ks,
        width=W,
        height=H,
        sh_degree=active_sph_order,
        render_mode='RGB+D',  # or 'RGB+D' if you need depth
        # Use defaults for other parameters or add them as needed
        near_plane=0.01,
        far_plane=1000.0,
        tile_size=16,
        rasterize_mode='classic',
        packed=False,
        
    )

    screenspace_points = (
        torch.zeros_like(mu, dtype=mu.dtype, requires_grad=True, device=mu.device)
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    assert screenspace_points is not None
    
    render_dict = {
        'alpha': rendered_alpha.squeeze(0),
        "rgb": rendered_image[0][...,0:3].permute(2,0,1),
        "dep": rendered_image[0][..., 3:4].permute(2,0,1),
        "alpha": rendered_alpha[0].permute(2,0,1),
        "visibility_filter": (aux_dict["radii"] > 0).all(-1).any(0),
        "radii": torch.sqrt(aux_dict["radii"][0][:, 0] * aux_dict["radii"][0][:, 1]),
        "viewspace_points": screenspace_points,
    }

    print(aux_dict.keys())

    assert render_dict["viewspace_points"] is not None

    for key in render_dict:
        value = render_dict[key]
        print(f"{key}: {value.shape}")

    return render_dict

# if GS_BACKEND == 'gsplat':
#     render = render_new
# else:
#     render = render_old

# render = render_new