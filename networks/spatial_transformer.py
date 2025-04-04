import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpatialTransform(nn.Module):
    """
        This implementation was taken from:
        https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/layers.py
    """

    def __init__(self, size):
        super(SpatialTransform, self).__init__()

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)


    def forward(self, src, flow, mode="bilinear"):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode, align_corners=True)  # nearest is slower

    def warp_point(self, points, flow):
        locs = self.grid + flow
        points = points.long()
        x, y, z = points[0, :, 0], points[0, :, 1], points[0, :, 2]
        warped_point = locs[:, :, x, y, z].permute(0, 2, 1)
        return warped_point.float()


class warp3D(nn.Module):
    def __init__(self, padding=False):
        super(warp3D, self).__init__()
        self.padding = padding

    def __call__(self, I, flow):
        return self._transform(I, flow[:, 0, :, :, :], flow[:, 1, :, :, :], flow[:, 2, :, :, :])

    def _meshgrid(self, depth, height, width):
        ones_height = torch.ones(height, 1).cuda()
        array_width = torch.linspace(0.0, float(width) - 1.0, width).cuda()
        x_t = torch.matmul(ones_height, (array_width.unsqueeze(1).permute(1, 0).contiguous()))
        x_t = x_t.unsqueeze(0).repeat(depth, 1, 1)

        ones_width = torch.ones(1, width).cuda()
        array_height = torch.linspace(0.0, float(height) - 1.0, height).cuda()
        y_t = torch.matmul(array_height.unsqueeze(1), ones_width)
        y_t = y_t.unsqueeze(0).repeat(depth, 1, 1)

        array_depth = torch.linspace(0.0, float(depth) - 1.0, depth).cuda()
        z_t = array_depth.unsqueeze(1).unsqueeze(1).repeat(1, height, width)

        return x_t, y_t, z_t

    def _transform(self, I, dx, dy, dz):
        batch_size = dx.shape[0]
        depth = dx.shape[1]
        height = dx.shape[2]
        width = dx.shape[3]

        # Convert dx and dy to absolute locations
        d_array = torch.arange(0, depth, dtype=torch.float32).cuda()
        h_array = torch.arange(0, height, dtype=torch.float32).cuda()
        w_array = torch.arange(0, width, dtype=torch.float32).cuda()
        z_mesh, y_mesh, x_mesh = torch.meshgrid(d_array, h_array, w_array, indexing='ij')

        # x_mesh, y_mesh, z_mesh = self._meshgrid(depth, height, width)
        x_mesh = x_mesh[np.newaxis]
        y_mesh = y_mesh[np.newaxis]
        z_mesh = z_mesh[np.newaxis]

        x_mesh = x_mesh.repeat(batch_size, 1, 1, 1)
        y_mesh = y_mesh.repeat(batch_size, 1, 1, 1)
        z_mesh = z_mesh.repeat(batch_size, 1, 1, 1)
        x_new = dx + x_mesh
        y_new = dy + y_mesh
        z_new = dz + z_mesh

        return self._interpolate(I, x_new, y_new, z_new)

    def _repeat(self, x, n_repeats):
        ones_repeat = torch.ones(size=[n_repeats, ])
        rep = ones_repeat.unsqueeze(1).permute(1, 0).contiguous().int()
        x = torch.matmul(x.view([-1, 1]).int(), rep)
        return x.view([-1])

    def _interpolate(self, im, x, y, z):
        if self.padding:
            im = F.pad(im, (1, 1, 1, 1, 1, 1))

        num_batch = im.shape[0]
        channels = im.shape[1]
        depth = im.shape[2]
        height = im.shape[3]
        width = im.shape[4]

        out_depth = x.shape[1]
        out_height = x.shape[2]
        out_width = x.shape[3]
        x = x.reshape([-1])
        y = y.reshape([-1])
        z = z.reshape([-1])

        padding_constant = 1 if self.padding else 0
        x = x.float() + padding_constant
        y = y.float() + padding_constant
        z = z.float() + padding_constant

        max_x = int(width - 1)
        max_y = int(height - 1)
        max_z = int(depth - 1)

        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1
        z0 = torch.floor(z).int()
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)
        z0 = torch.clamp(z0, 0, max_z)
        z1 = torch.clamp(z1, 0, max_z)

        dim1 = width
        dim2 = width * height
        dim3 = width * height * depth

        arange = torch.arange(num_batch)
        base = self._repeat(arange * dim3, out_depth * out_height * out_width).cuda()

        idx_a = (base + x0 + y0 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_b = (base + x0 + y1 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_c = (base + x1 + y0 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_d = (base + x1 + y1 * dim1 + z0 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_e = (base + x0 + y0 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_f = (base + x0 + y1 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_g = (base + x1 + y0 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)
        idx_h = (base + x1 + y1 * dim1 + z1 * dim2)[:, np.newaxis].repeat(1, channels)

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.permute(0, 2, 3, 4, 1).contiguous().view([-1, channels]).float()

        Ia = torch.gather(im_flat, 0, idx_a.long())
        Ib = torch.gather(im_flat, 0, idx_b.long())
        Ic = torch.gather(im_flat, 0, idx_c.long())
        Id = torch.gather(im_flat, 0, idx_d.long())
        Ie = torch.gather(im_flat, 0, idx_e.long())
        If = torch.gather(im_flat, 0, idx_f.long())
        Ig = torch.gather(im_flat, 0, idx_g.long())
        Ih = torch.gather(im_flat, 0, idx_h.long())

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()
        z1_f = z1.float()

        dx = x1_f - x
        dy = y1_f - y
        dz = z1_f - z

        wa = (dz * dx * dy)[:, np.newaxis]
        wb = (dz * dx * (1 - dy))[:, np.newaxis]
        wc = (dz * (1 - dx) * dy)[:, np.newaxis]
        wd = (dz * (1 - dx) * (1 - dy))[:, np.newaxis]
        we = ((1 - dz) * dx * dy)[:, np.newaxis]
        wf = ((1 - dz) * dx * (1 - dy))[:, np.newaxis]
        wg = ((1 - dz) * (1 - dx) * dy)[:, np.newaxis]
        wh = ((1 - dz) * (1 - dx) * (1 - dy))[:, np.newaxis]

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id + we * Ie + wf * If + wg * Ig + wh * Ih
        output = output.view([-1, out_depth, out_height, out_width, channels])

        return output.permute(0, 4, 1, 2, 3)


def __warp_point1(point, flow):
    # point: input fixed points [B, N, 3]
    B, N, D = point.shape

    # Create a batch index
    batch_index = torch.arange(B).view(-1, 1).to(flow.device)

    # Expand batch index to match the number of points
    batch_index = batch_index.expand(B, N)

    # Use advanced indexing to extract the corresponding flow vectors
    point = point.long()
    flow_vectors = flow[batch_index, :, point[..., 0], point[..., 1], point[..., 2]]
    flow_vectors = flow_vectors[:, :, [0, 2, 1]]

    # Calculate the predicted moving points
    wrp = point.float() + flow_vectors
    return wrp


def __warp_point2(fixed_kpts, flow):
    """
    计算关键点的目标注册误差 (TRE)

    参数:
    - fixed_kpts (torch.Tensor): 固定图像的关键点坐标，形状为 (B, N, 3)
    - flow (torch.Tensor): 流场，形状为 (B, 3, D, H, W)

    返回:
    - tre (torch.Tensor): 关键点的 TRE，形状为 (B, N)
    """
    B, N, _ = fixed_kpts.shape

    # 获取流场的形状
    D, H, W = flow.shape[2:]

    # 假设 fixed_kpts 的顺序是 (z, y, x)，需要调整为 (x, y, z)
    fixed_kpts_reordered = fixed_kpts[:, :, [2, 1, 0]]  # 重新排序坐标

    # 将固定图像的关键点坐标归一化到流场空间范围 [-1, 1]
    fixed_kpts_normalized = fixed_kpts_reordered.clone()
    fixed_kpts_normalized[:, :, 0] = 2 * (fixed_kpts_reordered[:, :, 0] / (W - 1)) - 1
    fixed_kpts_normalized[:, :, 1] = 2 * (fixed_kpts_reordered[:, :, 1] / (H - 1)) - 1
    fixed_kpts_normalized[:, :, 2] = 2 * (fixed_kpts_reordered[:, :, 2] / (D - 1)) - 1

    fixed_kpts_normalized = fixed_kpts_normalized.view(B, 1, N, 1, 3)

    # 获取流场中的位移向量
    flow_at_kpts = F.grid_sample(flow, fixed_kpts_normalized, align_corners=True)
    flow_at_kpts = flow_at_kpts.view(B, 3, N).permute(0, 2, 1)  # 形状为 (B, N, 3)

    # 计算变形后的关键点坐标
    warped_kpts = fixed_kpts + flow_at_kpts

    return warped_kpts


