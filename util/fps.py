import torch


# def index_points(points, idx):
#     """

#     Input:
#         points: input points data, [B, N, C]
#         idx: sample index data, [B, S]
#     Return:
#         new_points:, indexed points data, [B, S, C]
#     """
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, C, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device

    B, C, N = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    centroids_vals = torch.zeros(B, C, npoint).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest  # save current chosen point index
        centroid = xyz[batch_indices, :, farthest].view(B, C, 1)  # get the current chosen point value
        centroids_vals[:, :, i] = centroid[:, :, 0].clone()
        dist = torch.sum((xyz - centroid) ** 2, 1)  # euclidean distance of points from the current centroid
        mask = dist < distance  # save index of all point that are closer than the current max distance
        distance[mask] = dist[mask]  # save the minimal distance of each point from all points that were chosen until now
        farthest = torch.max(distance, -1)[1]  # get the index of the point farthest away
    return centroids, centroids_vals

