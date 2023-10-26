import torch
import numpy as np
import os
import sys
sys.path.append(os.getcwd())



def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()




def PointcloudRotate(bs_points):
    if bs_points.shape[-2] == 3:
        bs_points = bs_points.transpose(-2, -1)
    rotated_pcs = []
    rotation_angles = [np.pi * i/2 for i in range(4)]
    for bs in range(bs_points.shape[0]):
        points = bs_points[bs]
        for angle in rotation_angles:
            rotation_matrix = angle_axis(angle, np.array([0.0, 1.0, 0.0])).cuda()
            
            rotated_pc = torch.matmul(points, rotation_matrix.t())
            rotated_pcs.append(rotated_pc)
    rotated_pcs = torch.stack(rotated_pcs).transpose(-1, -2)
    return rotated_pcs, 4


if __name__ == '__main__':
    points = np.random.random((1024, 3))
    points = torch.tensor(points).float()
    PointcloudRotate(points)