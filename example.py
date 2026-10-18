# -*- coding: utf-8 -*-
"""Minimal self-contained example: instantiate SEMTRA-MML, count parameters,
and run a dummy forward pass (no dataset required)."""
import torch

from semtra_mml.models.semtra import SEMTRA_MML


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Base configuration (hidden dim 128); replace the metadata below with the
    # values of your city (num_roads/num_grids come from the preprocessing step).
    model = SEMTRA_MML(
        num_roads=7992, num_grids=13884,
        grid_map_shape=(89, 156), grid_channels=3,
        num_classes=0, pre_flag=False, device=device,
        motion_version="v5", motion_feature_dim=64, num_motion_anchors=12,
        lambda_motion=0.1, motion_supervision="full",
        motion_injection_mode="anchor", anchor_structure_weight=0.1,
        injection_mode="all", injection_scale_init=1.0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SEMTRA-MML (base) parameters: {n_params:,}")

    B, L = 4, 32
    road = torch.randint(0, 7992, (B, L), device=device)
    grid = torch.randint(0, 13884, (B, L), device=device)
    gps = torch.rand(B, L, 4, device=device)
    out = model(
        task="regression",
        road_indices=road, road_mask=torch.zeros_like(road, dtype=torch.bool),
        grid_indices=grid, grid_mask=torch.zeros_like(grid, dtype=torch.bool),
        gps_points=gps, gps_mask=torch.zeros(B, L, dtype=torch.bool, device=device),
        motion_features=torch.zeros(B, L, 64, device=device),
        motion_labels=torch.zeros(B, L, 12, device=device),
    )
    print("forward output:", out.shape if torch.is_tensor(out) else type(out))


if __name__ == "__main__":
    main()
