import torch

from lib_mosca.mosca import MoSca, __identify_spatial_unit_from_curves__, _compute_curve_topo_dist_


def auto_detect_levels(curve_xyz, curve_mask, max_levels=5):
    # Laplacian
    D = _compute_curve_topo_dist_(curve_xyz, curve_mask)
    L = torch.diag(D.sum(dim=1)) - D  
    
    # eigenvalues
    eigenvalues = torch.linalg.eigvalsh(L)
    spectral_gaps = eigenvalues[1:] - eigenvalues[:-1]  
    
    significant_gaps = spectral_gaps > spectral_gaps.mean() * 2.0
    level_indices = torch.where(significant_gaps)[0]

    spatial_unit = __identify_spatial_unit_from_curves__(curve_xyz, curve_mask)
    levels = [1] + [int(idx.item() + 2) for idx in level_indices[:max_levels-1]]
    
    return levels, spatial_unit

def adaptive_k_selection(levels, total_nodes):
    k_list = []
    for level in levels:
        base_k = min(32, max(4, int(total_nodes / (10 * level))))
        k_list.append(base_k)
    return k_list

def adaptive_weight_selection(levels):
    weights = [1.0 / (i+1) for i in range(len(levels))]
    weights = [w / sum(weights) for w in weights]
    return weights


class AutoMoSca(MoSca):
    def __init__(self, node_xyz, node_certain, **kwargs):
        levels, spatial_unit = auto_detect_levels(node_xyz, node_certain)
        
        k_list = adaptive_k_selection(levels, node_xyz.shape[1])
        w_list = adaptive_weight_selection(levels)
        
        super().__init__(
            node_xyz=node_xyz,
            node_certain=node_certain,
            mlevel_list=levels,
            mlevel_k_list=k_list,
            mlevel_w_list=w_list,
            **kwargs
        )