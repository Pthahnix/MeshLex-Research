"""Loss functions for MeshLex."""
import torch


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute masked Chamfer Distance between predicted and GT vertices.

    Args:
        pred: (B, max_V, 3) predicted vertex coordinates
        gt: (B, max_V, 3) ground truth vertex coordinates
        mask: (B, max_V) boolean mask — True for valid vertices

    Returns:
        Scalar mean Chamfer Distance across batch.
    """
    B = pred.shape[0]
    total_cd = 0.0

    for b in range(B):
        m = mask[b]  # (max_V,)
        p = pred[b][m]  # (Np, 3)
        g = gt[b][m]    # (Ng, 3)

        if p.shape[0] == 0 or g.shape[0] == 0:
            continue

        # pred → gt: for each predicted point, find nearest GT
        dist_p2g = torch.cdist(p, g)  # (Np, Ng)
        min_p2g = dist_p2g.min(dim=1).values  # (Np,)

        # gt → pred: for each GT point, find nearest predicted
        min_g2p = dist_p2g.min(dim=0).values  # (Ng,)

        cd = min_p2g.mean() + min_g2p.mean()
        total_cd += cd

    return total_cd / B
