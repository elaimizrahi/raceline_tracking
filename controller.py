import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# params

K_P_V_ACCEL = 2.0
K_P_V_DECEL = 2.5 
K_P_DELTA = 6

LOOKAHEAD_BASE = 2
LOOKAHEAD_GAIN = 0.3

SPEED_LD_BASE = 2
SPEED_LD_GAIN = 1.3

MAX_FRAC_VMAX = 0.8
MIN_SPEED = 5.0
LAT_ACC_LIMIT = 6.0

CURV_WIN = 3

# cache to reduce some calculations

_PATH_CACHE: dict[int, dict[str, np.ndarray]] = {}


def _compute_path(rt: RaceTrack):
    path = getattr(rt, "raceline", rt.centerline)
    N = len(path)

    # curvature estimate
    curv = np.zeros(N)
    for i in range(N):
        p0 = path[(i - 1) % N]
        p1 = path[i]
        p2 = path[(i + 1) % N]

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)

        if a * b * c > 1e-6:
            area2 = abs(np.cross(p1 - p0, p2 - p0))
            curv[i] = area2 / (a * b * c)
        else:
            curv[i] = 0.0

    return {"path": path, "curv": curv, "N": N}


def _get_path(rt: RaceTrack):
    key = id(rt)
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = _compute_path(rt)
    return _PATH_CACHE[key]


def _closest(path, pos):
    diff = path - pos
    return int(np.argmin((diff * diff).sum(axis=1)))


def _ahead_idx(path, start, dist):
    N = len(path)
    acc = 0.0
    i = start
    steps = 0
    while acc < dist and steps < N:
        j = (i + 1) % N
        acc += np.linalg.norm(path[j] - path[i])
        i = j
        steps += 1
    return i

# low-level controller

def lower_controller(state: ArrayLike, desired: ArrayLike, params: ArrayLike):
    delta = float(state[2])
    v = float(state[3])
    delta_r, v_r = desired

    u_delta = K_P_DELTA * (delta_r - delta)
    if v_r > v:
        u_v = K_P_V_ACCEL * (v_r - v)
    else:
        u_v = K_P_V_DECEL * (v_r - v)

    return np.array([u_delta, u_v], float)


# high-level controller

def controller(state: ArrayLike, params: ArrayLike, track: RaceTrack):
    sx, sy, delta, v, phi = map(float, state)
    wheelbase = float(params[0])
    v_max = float(params[5])

    info = _get_path(track)
    path = info["path"]
    curv = info["curv"]
    N = info["N"]

    pos = np.array([sx, sy])
    idx = _closest(path, pos)

    # speed lookahead 
    ld_speed = SPEED_LD_BASE + SPEED_LD_GAIN * max(v, 0)
    idx_s = _ahead_idx(path, idx, ld_speed)

    w = CURV_WIN // 2
    k_idxs = [(idx_s + k) % N for k in range(-w, w + 1)]
    k = max(float(curv[k_idxs].mean()), 1e-4)

    v_corner = np.sqrt(LAT_ACC_LIMIT / k)
    v_r = np.clip(v_corner, MIN_SPEED, MAX_FRAC_VMAX * v_max)

    # steering lookahead 
    k_idxs2 = [(idx + k2) % N for k2 in range(-w, w + 1)]
    k2 = max(float(curv[k_idxs2].mean()), 1e-4)

    base_ld = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * v_r
    ld = base_ld / (1.0 + 8.0 * k2)

    idx_t = _ahead_idx(path, idx, ld)
    tgt = path[idx_t]

    vec = tgt - pos
    d = float(np.linalg.norm(vec))

    if d < 1e-3:
        delta_r = 0.0
    else:
        tgt_heading = np.arctan2(vec[1], vec[0])
        angle = tgt_heading - phi
        angle = np.arctan2(np.sin(angle), np.cos(angle))

        delta_r = np.arctan2(2 * wheelbase * np.sin(angle), d)

    return np.array([delta_r, v_r], float)
