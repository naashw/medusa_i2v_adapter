"""Camera path interpolation — quaternion math + Catmull-Rom for 6-DOF motion.

Conventions:
  - Axes: X=right, Y=up, Z=forward (right-handed)
  - Quaternion: [w, x, y, z] (Hamilton)
  - Euler: YXZ intrinsic order (yaw, pitch, roll)
  - Time t: normalized [0.0, 1.0] across entire path
  - Catmull-Rom: "clamped" mode (edge keyframes duplicated)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


EULER_ORDER = "YXZ"


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix.

    Args:
        q: torch.Tensor of shape [..., 4] with components [w, x, y, z]

    Returns:
        torch.Tensor of shape [..., 3, 3] rotation matrix
    """
    # Ensure float
    q = q.float()

    # Normalize (safety)
    q = F.normalize(q, p=2, dim=-1)

    # Extract components
    w, x, y, z = torch.split(q, 1, dim=-1)
    w = w.squeeze(-1)
    x = x.squeeze(-1)
    y = y.squeeze(-1)
    z = z.squeeze(-1)

    # Pre-compute squares
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yw = y * w
    zw = z * w
    yz = y * z
    xw = x * w

    # Build rotation matrix
    mat = torch.stack([
        torch.stack([1 - 2*(yy + zz), 2*(xy - zw), 2*(xz + yw)], dim=-1),
        torch.stack([2*(xy + zw), 1 - 2*(xx + zz), 2*(yz - xw)], dim=-1),
        torch.stack([2*(xz - yw), 2*(yz + xw), 1 - 2*(xx + yy)], dim=-1),
    ], dim=-2)

    return mat


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: torch.Tensor shape [..., 4] with [w, x, y, z]
        q1: torch.Tensor shape [..., 4] with [w, x, y, z]
        t: float in [0, 1]

    Returns:
        torch.Tensor shape [..., 4] interpolated quaternion
    """
    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)

    # Dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # If dot < 0, negate one quat to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.where(dot < 0, -dot, dot)

    # Clamp dot to avoid acos numerical issues
    dot = torch.clamp(dot, -1.0, 1.0)

    # Compute angle
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Avoid division by zero
    w0 = torch.where(
        sin_theta > 1e-6,
        torch.sin((1 - t) * theta) / sin_theta,
        torch.tensor(1 - t, dtype=q0.dtype, device=q0.device)
    )
    w1 = torch.where(
        sin_theta > 1e-6,
        torch.sin(t * theta) / sin_theta,
        torch.tensor(t, dtype=q0.dtype, device=q0.device)
    )

    result = w0 * q0 + w1 * q1
    return F.normalize(result, p=2, dim=-1)


def catmull_rom_vec3(
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    p3: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Catmull-Rom cubic interpolation for 3D points.

    Args:
        p0, p1, p2, p3: torch.Tensor shape [..., 3]
        t: float in [0, 1] (local time on segment [p1, p2])

    Returns:
        torch.Tensor shape [..., 3] interpolated point
    """
    t2 = t * t
    t3 = t2 * t

    # Catmull-Rom basis
    c0 = -0.5 * t3 + t2 - 0.5 * t
    c1 = 1.5 * t3 - 2.5 * t2 + 1.0
    c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    c3 = 0.5 * t3 - 0.5 * t2

    # Expand c to shape [..., 1] for broadcasting
    c0 = c0 if isinstance(c0, torch.Tensor) else torch.tensor(c0, dtype=p1.dtype, device=p1.device)
    c1 = c1 if isinstance(c1, torch.Tensor) else torch.tensor(c1, dtype=p1.dtype, device=p1.device)
    c2 = c2 if isinstance(c2, torch.Tensor) else torch.tensor(c2, dtype=p1.dtype, device=p1.device)
    c3 = c3 if isinstance(c3, torch.Tensor) else torch.tensor(c3, dtype=p1.dtype, device=p1.device)

    # Ensure scalars
    if c0.dim() == 0:
        c0 = c0.unsqueeze(-1) if p1.dim() > 1 else c0
        c1 = c1.unsqueeze(-1) if p1.dim() > 1 else c1
        c2 = c2.unsqueeze(-1) if p1.dim() > 1 else c2
        c3 = c3.unsqueeze(-1) if p1.dim() > 1 else c3

    return c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3


def catmull_rom_quat(
    q0: torch.Tensor,
    q1: torch.Tensor,
    q2: torch.Tensor,
    q3: torch.Tensor,
    t: float,
) -> torch.Tensor:
    """Catmull-Rom cubic interpolation on quaternion manifold R⁴ + renormalization.

    Args:
        q0, q1, q2, q3: torch.Tensor shape [..., 4] with [w, x, y, z]
        t: float in [0, 1]

    Returns:
        torch.Tensor shape [..., 4] normalized interpolated quaternion
    """
    t2 = t * t
    t3 = t2 * t

    c0 = -0.5 * t3 + t2 - 0.5 * t
    c1 = 1.5 * t3 - 2.5 * t2 + 1.0
    c2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    c3 = 0.5 * t3 - 0.5 * t2

    # Convert to tensors if needed
    c0 = c0 if isinstance(c0, torch.Tensor) else torch.tensor(c0, dtype=q1.dtype, device=q1.device)
    c1 = c1 if isinstance(c1, torch.Tensor) else torch.tensor(c1, dtype=q1.dtype, device=q1.device)
    c2 = c2 if isinstance(c2, torch.Tensor) else torch.tensor(c2, dtype=q1.dtype, device=q1.device)
    c3 = c3 if isinstance(c3, torch.Tensor) else torch.tensor(c3, dtype=q1.dtype, device=q1.device)

    # Expand for broadcasting
    if c0.dim() == 0:
        c0 = c0.unsqueeze(-1) if q1.dim() > 1 else c0
        c1 = c1.unsqueeze(-1) if q1.dim() > 1 else c1
        c2 = c2.unsqueeze(-1) if q1.dim() > 1 else c2
        c3 = c3.unsqueeze(-1) if q1.dim() > 1 else c3

    result = c0 * q0 + c1 * q1 + c2 * q2 + c3 * q3
    return F.normalize(result, p=2, dim=-1)


def interpolate_camera_path(
    camera_path: list[dict],
    t: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate camera pose along a keyframed path.

    Args:
        camera_path: list of dicts with keys "t", "translation" [3], "rotation_quat" [4]
        t: float in [0.0, 1.0] normalized time
        mode: "linear" or "cubic" (Catmull-Rom)

    Returns:
        (translation [3], rotation_quat [4]) as torch tensors (device=cpu by default)
    """
    # Find segment [i, i+1] that contains t
    keyframes = sorted(camera_path, key=lambda kf: kf["t"])

    if len(keyframes) == 1:
        kf = keyframes[0]
        trans = torch.tensor(kf["translation"], dtype=torch.float32)
        quat = torch.tensor(kf["rotation_quat"], dtype=torch.float32)
        return trans, quat

    # Clamp t to valid range
    t = max(0.0, min(1.0, t))

    # Find segment
    segment_i = 0
    for i in range(len(keyframes) - 1):
        if keyframes[i]["t"] <= t <= keyframes[i + 1]["t"]:
            segment_i = i
            break
    else:
        segment_i = len(keyframes) - 2

    kf_i = keyframes[segment_i]
    kf_i1 = keyframes[segment_i + 1]

    # Local time on segment
    t_range = kf_i1["t"] - kf_i["t"]
    if t_range < 1e-6:
        local_t = 0.5
    else:
        local_t = (t - kf_i["t"]) / t_range

    # Linear or bord case
    if mode == "linear" or len(keyframes) < 4 or segment_i == 0 or segment_i >= len(keyframes) - 2:
        # Lerp translation
        trans_i = torch.tensor(kf_i["translation"], dtype=torch.float32)
        trans_i1 = torch.tensor(kf_i1["translation"], dtype=torch.float32)
        trans = (1 - local_t) * trans_i + local_t * trans_i1

        # Slerp rotation
        quat_i = torch.tensor(kf_i["rotation_quat"], dtype=torch.float32)
        quat_i1 = torch.tensor(kf_i1["rotation_quat"], dtype=torch.float32)
        quat = slerp(quat_i, quat_i1, local_t)

    else:
        # Cubic Catmull-Rom with clamped edges
        kf_i_minus_1 = keyframes[segment_i - 1]
        kf_i_plus_2 = keyframes[segment_i + 2] if segment_i + 2 < len(keyframes) else keyframes[-1]

        # Translation
        p0 = torch.tensor(kf_i_minus_1["translation"], dtype=torch.float32)
        p1 = torch.tensor(kf_i["translation"], dtype=torch.float32)
        p2 = torch.tensor(kf_i1["translation"], dtype=torch.float32)
        p3 = torch.tensor(kf_i_plus_2["translation"], dtype=torch.float32)
        trans = catmull_rom_vec3(p0, p1, p2, p3, local_t)

        # Rotation
        q0 = torch.tensor(kf_i_minus_1["rotation_quat"], dtype=torch.float32)
        q1 = torch.tensor(kf_i["rotation_quat"], dtype=torch.float32)
        q2 = torch.tensor(kf_i1["rotation_quat"], dtype=torch.float32)
        q3 = torch.tensor(kf_i_plus_2["rotation_quat"], dtype=torch.float32)
        quat = catmull_rom_quat(q0, q1, q2, q3, local_t)

    return trans, quat
