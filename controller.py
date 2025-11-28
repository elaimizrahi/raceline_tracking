import numpy as np
from numpy.typing import ArrayLike
from simulator import RaceTrack

# params

K_P_V_ACCEL = 15
K_P_V_DECEL = 10
K_P_DELTA = 5.0
K_I_DELTA = 0.0
K_D_DELTA = 4.0

LOOKAHEAD_BASE = 5.0
LOOKAHEAD_GAIN = 0.4

SPEED_LD_BASE = 2
SPEED_LD_GAIN = 1.3

MIN_SPEED = 5.0
LAT_ACC_LIMIT = 19.0
LONG_ACCEL_LIMIT = 11.0

CURV_WIN = 5

_STEER_INT = 0.0
_STEER_PREV = 0.0
_FIRST_RUN = True
_STEP_COUNT = 0

# cache to reduce some calculations

_PATH_CACHE: dict[int, dict[str, np.ndarray]] = {}


def calculatePath(rt: RaceTrack):
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

    # smooth curvature and computing speed
    v_limit = np.sqrt(LAT_ACC_LIMIT / (np.abs(curv) + 1e-6))
    for _ in range(2):
        for i in range(N - 1, -1, -1):
            p_curr = path[i]
            p_next = path[(i + 1) % N]
            dist = np.linalg.norm(p_next - p_curr)
            
            v_next = v_limit[(i + 1) % N]
            max_entry_v = np.sqrt(v_next**2 + 2 * LONG_ACCEL_LIMIT * dist)
            v_limit[i] = min(v_limit[i], max_entry_v)

    return {"path": path, "curv": curv, "N": N, "v_profile": v_limit}


def getPath(rt: RaceTrack):
    key = id(rt)
    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = calculatePath(rt)
    return _PATH_CACHE[key]


def getClosest(path, pos):
    diff = path - pos
    return int(np.argmin((diff * diff).sum(axis=1)))


def getAheadIdx(path, start, dist):
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

def getTargetPoint(path, start_idx, dist):
    N = len(path)
    acc = 0.0
    i = start_idx
    steps = 0
    
    while steps < N:
        j = (i + 1) % N
        seg_vec = path[j] - path[i]
        seg_len = np.linalg.norm(seg_vec)
        
        if acc + seg_len >= dist:
            remaining = dist - acc
            ratio = remaining / (seg_len + 1e-6)
            return path[i] + ratio * seg_vec
            
        acc += seg_len
        i = j
        steps += 1
    
    return path[i]

# low-level controller

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike):
    global _STEER_INT, _STEER_PREV, _FIRST_RUN
    delta = float(state[2])
    v = float(state[3])
    delta_r, v_r = desired

    error = delta_r - delta
    
    if _FIRST_RUN:
        _STEER_PREV = error
        _FIRST_RUN = False

    _STEER_INT += error
    _STEER_INT = np.clip(_STEER_INT, -1.0, 1.0)
    
    deriv = error - _STEER_PREV
    _STEER_PREV = error
    
    u_delta = K_P_DELTA * error + K_I_DELTA * _STEER_INT + K_D_DELTA * deriv

    if v_r > v:
        u_v = K_P_V_ACCEL * (v_r - v)
    else:
        u_v = K_P_V_DECEL * (v_r - v)

    return np.array([u_delta, u_v], float)


# high-level controller

def controller(state: ArrayLike, params: ArrayLike, raceTrack: RaceTrack):
    global _STEP_COUNT
    sx, sy, delta, v, phi = map(float, state)
    wheelbase = float(params[0])
    v_max = float(params[5])

    info = getPath(raceTrack)
    path = info["path"]
    v_profile = info["v_profile"]
    
    pos = np.array([sx, sy])
    idx = getClosest(path, pos)

    idx_s = getAheadIdx(path, idx, max(v * 0.1, 0.5))
    target_v = v_profile[idx_s]
    v_r = np.clip(target_v, MIN_SPEED, v_max)

    ld = LOOKAHEAD_BASE + LOOKAHEAD_GAIN * v

    # controlled merge to raceline
    if _STEP_COUNT < 25:
        ld = max(ld, 40.0)
    
    _STEP_COUNT += 1

    tgt = getTargetPoint(path, idx, ld)

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
