#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.sh_utils import eval_sh


def project_gaussian_means_to_2d(model: GaussianModel, gsplat_cam):
    gaussians_pos = model.get_xyz
    # N: Number of gaussian splats
    N = gaussians_pos.shape[0]
    # Shape (N, 1)
    ones_padding = gaussians_pos.new_ones(N, 1)
    # Shape (N, 4)
    xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
    # Shape (N, 4, 1)
    xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
    # Shape (N, 4, 4)
    # The depth channel is obtained in view space
    cam_view_projection_matrix = gsplat_cam.world_view_transform.T[None].expand(N, 4, 4)
    # Shape (N, 4, 1)
    transformed_xyz_view = cam_view_projection_matrix @ xyz_homogeneous

    # Perform perspective division to obtain (N, 4) of [x_ndc, y_ndc, depth, 1.0]
    cam_view_projection_matrix = gsplat_cam.full_proj_transform.T[None].expand(N, 4, 4)
    transformed_xyz = cam_view_projection_matrix @ xyz_homogeneous
    transformed_xyz /= transformed_xyz[:, -1:]
    transformed_xyz[:, 2] = transformed_xyz_view[:, 2]

    # Shape (N, 4)
    transformed_xyz = transformed_xyz.squeeze(-1)
    return transformed_xyz

# def project_gaussian_means_to_2d(model: GaussianModel, gsplat_cam):
#     gaussians_pos = model.get_xyz
#     # N: Number of gaussian splats
#     N = gaussians_pos.shape[0]
#     # Shape (N, 1)
#     ones_padding = gaussians_pos.new_ones(N, 1)
#     # Shape (N, 4)
#     xyz_homogeneous = torch.cat([gaussians_pos, ones_padding], dim=1)
#     # Shape (N, 4, 1)
#     xyz_homogeneous = xyz_homogeneous.unsqueeze(-1)
#     # Shape (N, 4, 4)
#     # The depth channel is obtained in view space
#     cam_view_projection_matrix = gsplat_cam.world_view_transform.T[None].expand(N, 4, 4)
#     # Shape (N, 4, 1)
#     transformed_xyz_view = cam_view_projection_matrix @ xyz_homogeneous
#     # Shape (N, 4)
#     transformed_xyz_view = transformed_xyz_view.squeeze(-1)
#     return transformed_xyz_view


def render(viewpoint_camera, pc: GaussianModel,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           override_color=None,
           directional_light=None,
           directional_light_intensity=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Shadow mapping
    gs_shadow_level = None
    if directional_light is not None and directional_light_intensity is not None:
        if isinstance(directional_light, Camera):
            directional_light = [directional_light]

        gs_shadow_level = means3D.new_full([means3D.shape[0], 1], fill_value=len(directional_light))

        for dl in directional_light:
            depth_tanfovx = math.tan(dl.FoVx * 0.5)
            depth_tanfovy = math.tan(dl.FoVy * 0.5)
            depth_raster_settings = GaussianRasterizationSettings(
                image_height=int(dl.image_height),
                image_width=int(dl.image_width),
                tanfovx=depth_tanfovx,
                tanfovy=depth_tanfovy,
                bg=bg_color,
                scale_modifier=scaling_modifier,
                viewmatrix=dl.world_view_transform,
                projmatrix=dl.full_proj_transform,
                sh_degree=pc.active_sh_degree,
                campos=dl.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )
            shadow_rasterizer = GaussianRasterizer(raster_settings=depth_raster_settings)
            _, _, shadowmap = shadow_rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp)

            # (1, H, W) -> (H, W)
            shadowmap = shadowmap[0]

            # light_coords_2d: (N, 4) of [x_ndc, y_ndc, depth, 1.0]
            gs_light_coords_2d = project_gaussian_means_to_2d(pc, dl)

            # NDC to (H, W)
            # Whatever falls within frustrum
            masked_in = (gs_light_coords_2d[:, 0] <= 1.0) & \
                        (gs_light_coords_2d[:, 0] >= -1.0) & \
                        (gs_light_coords_2d[:, 1] <= 1.0) & \
                        (gs_light_coords_2d[:, 1] >= -1.0)
            gs_light_coords_2d = gs_light_coords_2d[masked_in].contiguous()
            gs_light_coords_2d[:, 0] = torch.round(((gs_light_coords_2d[:, 0] + 1.0) * 0.5) * dl.image_height)
            gs_light_coords_2d[:, 1] = torch.round(((gs_light_coords_2d[:, 1] + 1.0) * 0.5) * dl.image_width)
            gs_light_coords_2d[:, 0] = torch.clamp(gs_light_coords_2d[:, 0], 0, dl.image_height - 1)
            gs_light_coords_2d[:, 1] = torch.clamp(gs_light_coords_2d[:, 1], 0, dl.image_width - 1)
            gs_light_coords_2d = gs_light_coords_2d.long()
            gs_shadow_level -= 1.0
            gs_shadow_level[masked_in, 0] += (gs_light_coords_2d[:, 2] <= shadowmap[gs_light_coords_2d[:, 0], gs_light_coords_2d[:, 1]]).float()

    gs_shadow_level /= len(directional_light)
    gs_shadow_level = torch.clamp(gs_shadow_level * 3.0, 0.0, 1.0)
    # gs_shadow_level[gs_shadow_level > (2/len(directional_light))] = 1.0

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        gs_shadow_level=gs_shadow_level,
        directional_light_intensity=directional_light_intensity)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "depth": depth}

    # # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)
    #
    # # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii}
