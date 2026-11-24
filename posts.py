"""Anchor posts: an image pinned to a spot in the real world.

A post is a folder that can be shared as-is:
  anchor.png    the patch of the scene the image is pinned to, as seen when it was placed
                (a stretch of textured wall around the image); this is what gets tracked
  content.png   the image that is shown
  post.json     where the content sits relative to the anchor: its 4 corners in anchor
                pixel coordinates (they may lie outside the anchor), plus metadata
  map.npz       optional scene calibration recorded at that spot (see calibration.py)

Anyone who loads the post and points a camera at the same spot sees the content there.
"""
import json
import os
import time

import cv2
import numpy as np

POSTS_DIR = "posts"

# The anchor is the content rectangle grown by this fraction of its size on every side:
# more surrounding wall means more features to recognise the spot by.
ANCHOR_PAD = 0.6

# Plain painted walls have almost no features and cannot be recognised again; refuse
# them up front rather than produce a post nobody can find.
MIN_ANCHOR_FEATURES = 150


def anchor_rect(frame_shape, rect, pad=ANCHOR_PAD):
    """Padded rectangle (x, y, w, h) around the content rectangle, clipped to the frame."""
    x, y, w, h = rect
    H, W = frame_shape[:2]
    x1, y1 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
    x2, y2 = min(W, int(x + w + pad * w)), min(H, int(y + h + pad * h))
    return x1, y1, x2 - x1, y2 - y1


def fit_content(rect, content_shape):
    """Largest rectangle with the content's aspect ratio centred inside rect, as 4 corners."""
    x, y, w, h = rect
    ch, cw = content_shape[:2]
    s = min(w / cw, h / ch)
    fw, fh = cw * s, ch * s
    x0, y0 = x + (w - fw) / 2, y + (h - fh) / 2
    return np.float32([[x0, y0], [x0 + fw, y0], [x0 + fw, y0 + fh], [x0, y0 + fh]])


def count_features(gray):
    return len(cv2.SIFT_create().detect(gray, None))


def check_anchor(frame, rect):
    """(ok, n_features, anchor_rect) for placing content at rect in this frame."""
    a = anchor_rect(frame.shape, rect)
    x, y, w, h = a
    if w < 40 or h < 40:
        return False, 0, a
    gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    n = count_features(gray)
    return n >= MIN_ANCHOR_FEATURES, n, a


def create(frame, rect, content, name=None, posts_dir=POSTS_DIR):
    """Save a new post pinning `content` at `rect` (x, y, w, h) of `frame`. Returns its folder."""
    ok, n, (ax, ay, aw, ah) = check_anchor(frame, rect)
    if not ok:
        raise ValueError(f"only {n} features around that spot (need {MIN_ANCHOR_FEATURES}); "
                         "pick a more detailed spot")
    name = name or time.strftime("post-%Y%m%d-%H%M%S")
    folder = os.path.join(posts_dir, name)
    os.makedirs(folder, exist_ok=True)

    quad = fit_content(rect, content.shape) - np.float32([ax, ay])

    cv2.imwrite(os.path.join(folder, "anchor.png"), frame[ay:ay + ah, ax:ax + aw])
    cv2.imwrite(os.path.join(folder, "content.png"), content)
    with open(os.path.join(folder, "post.json"), "w") as f:
        json.dump({
            "name": name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "content_quad": quad.tolist(),
            "anchor_rect_in_frame": [ax, ay, aw, ah],
            "frame_size": [frame.shape[1], frame.shape[0]],
            "anchor_features": n,
            "version": 1,
        }, f, indent=2)
    return folder


def load(folder):
    """dict with name, folder, anchor (grayscale), content (BGR), quad (4x2 float32)."""
    with open(os.path.join(folder, "post.json")) as f:
        meta = json.load(f)
    anchor = cv2.imread(os.path.join(folder, "anchor.png"), cv2.IMREAD_GRAYSCALE)
    content = cv2.imread(os.path.join(folder, "content.png"))
    if anchor is None or content is None:
        raise FileNotFoundError(f"{folder} is missing anchor.png or content.png")
    return {
        "name": meta["name"],
        "folder": folder,
        "anchor": anchor,
        "content": content,
        "quad": np.float32(meta["content_quad"]),
        "meta": meta,
    }
