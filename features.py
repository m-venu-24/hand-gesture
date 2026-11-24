"""Shared landmark -> feature utilities used by collection, training and the live app.

Features are 21 landmarks x (x, y, z) = 63 values, made invariant to:
  * hand position in the frame   (wrist moved to origin)
  * hand size / distance         (scaled by the largest wrist-to-landmark distance)
  * left vs right hand           (left hand is mirrored to look like a right hand)
"""
import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def csv_header():
    return ["label"] + [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]


def normalize(pts: np.ndarray) -> np.ndarray:
    """pts: (21, 3) raw MediaPipe coordinates -> (63,) normalised features."""
    pts = pts.astype(np.float32).copy()
    pts -= pts[0]  # wrist at origin
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()


def extract(result):
    """MediaPipe Hands result -> (features, hand_landmarks) or None if no hand."""
    if not result.multi_hand_landmarks:
        return None
    hand = result.multi_hand_landmarks[0]
    side = "Right"
    if result.multi_handedness:
        side = result.multi_handedness[0].classification[0].label
    pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    if side == "Left":
        pts[:, 0] = 1.0 - pts[:, 0]  # mirror so both hands share one feature space
    return normalize(pts), hand


def augment(X: np.ndarray, y: np.ndarray, copies: int = 4, seed: int = 42):
    """Create extra training samples: small in-plane rotation, scale jitter, noise."""
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(copies):
        P = X.reshape(-1, NUM_LANDMARKS, 3).copy()
        theta = rng.uniform(-np.radians(15), np.radians(15), size=(len(P), 1))
        c, s = np.cos(theta), np.sin(theta)
        x, yy = P[:, :, 0].copy(), P[:, :, 1].copy()
        P[:, :, 0] = x * c - yy * s
        P[:, :, 1] = x * s + yy * c
        P *= rng.uniform(0.92, 1.08, size=(len(P), 1, 1))
        P += rng.normal(0.0, 0.015, P.shape)
        Xs.append(P.reshape(len(P), -1).astype(np.float32))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)
