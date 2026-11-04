"""Scene calibration for tep3.py.

The user stands still and sweeps the phone around (up to ~180 degrees) while the target
is tracked. Every ~15 degrees a keyframe is stored: its features and the camera rotation.
Because the camera only rotates, every view relates to the target through a 3D rotation,
so later the system can tell where the target is from ANY part of the scene, even when
the target is out of view or behind the camera.

Conventions: the world frame is the camera at the start of calibration. R maps world
directions to camera directions, so a world ray d appears at pixel x ~ K R d. Between two
views a and b of a rotating camera, x_b ~ K R_b R_a^T K^-1 x_a.
"""
import os
import time

import cv2
import numpy as np

# horizontal field of view of the phone camera (16:9 phone cameras are ~60-70 degrees)
CAMERA_HFOV_DEG = 66.0

KEYFRAME_ANGLE_DEG = 15.0     # add a keyframe after the camera turned this far from all others
KEYFRAME_SCALE = 0.5          # keyframe features are extracted on a downscaled frame
KEYFRAME_FEATURES = 800
SWEEP_TARGET_DEG = 170.0      # calibration finishes once the keyframes span this much yaw
MATCH_MIN_INLIERS = 25
RELOC_CANDIDATES = 3          # keyframes tried per relocalization when the rotation is roughly known
# Relocalizing against similar-looking parts of a room produces confident-looking wrong
# matches. On a synthetic room those topped out at 55 inliers while correct ones had a
# median of 188, and a correct match's homography is (nearly) a pure rotation.
RELOC_MIN_INLIERS = 60
MAX_ROTATION_RESIDUAL = 0.05


def intrinsics(w, h, hfov_deg=CAMERA_HFOV_DEG):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def nearest_rotation(A):
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_from_homography(H, K, max_residual=None):
    """Relative rotation of a purely rotating camera from the image homography between views.

    With max_residual, returns None when the homography is not close to a pure rotation
    (a sign of a wrong match, or of the camera having moved rather than turned)."""
    A = np.linalg.inv(K) @ H @ K
    d = np.linalg.det(A)
    if not np.isfinite(d) or abs(d) < 1e-9:
        return None
    A = A / np.cbrt(d)
    R = nearest_rotation(A)
    if max_residual is not None and np.linalg.norm(A - R) > max_residual:
        return None
    return R


def rotation_from_motion(M, K):
    """Rotation of the camera between two frames from the 2D image motion M (3x3, affine).

    M has no perspective terms, so treating K^-1 M K as a rotation roughly halves the
    angle. Instead: the ray through the image centre must move to where M moves the
    centre (yaw/pitch), and M's in-plane angle is the roll."""
    c = np.array([K[0, 2], K[1, 2], 1.0])
    r0 = np.linalg.inv(K) @ c
    r1 = np.linalg.inv(K) @ (M @ c)
    r0 /= np.linalg.norm(r0)
    r1 /= np.linalg.norm(r1)
    axis = np.cross(r0, r1)
    s, co = np.linalg.norm(axis), float(np.dot(r0, r1))
    if s < 1e-12:
        R_align = np.eye(3)
    else:
        R_align, _ = cv2.Rodrigues(axis / s * np.arctan2(s, co))
    roll = np.arctan2(M[1, 0], M[0, 0])
    R_roll, _ = cv2.Rodrigues(r1 * roll)
    return R_roll @ R_align


def angle_between(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def yaw_of(R):
    # direction of the optical axis in world coordinates, as a left/right angle
    z = R.T @ np.array([0, 0, 1.0])
    return float(np.degrees(np.arctan2(z[0], z[2])))


class SceneMap:

    def __init__(self, K, frame_size, H0, mask_top=0.0):
        self.K = K
        self.frame_size = tuple(frame_size)   # (w, h)
        self.H0 = H0                          # reference image -> calibration start frame
        self.mask_top = mask_top
        self.rotations = []
        self.points = []
        self.descriptors = []
        self.thumbs = []
        self._init_runtime()

    def _init_runtime(self):
        self.sift = cv2.SIFT_create(KEYFRAME_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.Kinv = np.linalg.inv(self.K)
        self._mask = None

    # ---------- features and matching ----------

    def features(self, gray):
        small = cv2.resize(gray, None, fx=KEYFRAME_SCALE, fy=KEYFRAME_SCALE)
        if self._mask is None or self._mask.shape != small.shape:
            self._mask = np.full(small.shape, 255, np.uint8)
            self._mask[:int(self.mask_top * small.shape[0])] = 0
        kps, des = self.sift.detectAndCompute(small, self._mask)
        if des is None or len(kps) < 8:
            return None
        pts = np.float32([k.pt for k in kps]) / KEYFRAME_SCALE
        return pts, des

    def match(self, feats_a, feats_b):
        """Homography from view a to view b, with its inlier count (or None, 0)."""
        if feats_a is None or feats_b is None:
            return None, 0
        pa, da = feats_a
        pb, db = feats_b
        good = [p[0] for p in self.matcher.knnMatch(da, db, k=2)
                if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < MATCH_MIN_INLIERS:
            return None, 0
        src = pa[[m.queryIdx for m in good]].reshape(-1, 1, 2)
        dst = pb[[m.trainIdx for m in good]].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or mask is None or mask.sum() < MATCH_MIN_INLIERS:
            return None, 0
        return H, int(mask.sum())

    # ---------- building the map ----------

    def add_keyframe(self, R, feats, gray):
        self.rotations.append(R)
        self.points.append(feats[0])
        self.descriptors.append(feats[1])
        self.thumbs.append(cv2.resize(gray, (160, 90)))

    def nearest_keyframe(self, R):
        angles = [angle_between(R, Rk) for Rk in self.rotations]
        k = int(np.argmin(angles))
        return k, angles[k]

    def maybe_add_keyframe(self, gray, R):
        """During calibration: store a keyframe once the camera turned far enough.

        Returns the rotation refined by matching against the nearest keyframe (the
        frame-to-frame estimate drifts; keyframe matching keeps the chain accurate),
        or the input rotation if no keyframe was added."""
        if not self.rotations:
            feats = self.features(gray)
            if feats is not None:
                self.add_keyframe(R, feats, gray)
            return R
        k, angle = self.nearest_keyframe(R)
        if angle < KEYFRAME_ANGLE_DEG:
            return R
        feats = self.features(gray)
        R_matched = self._match_rotation(k, feats)
        if R_matched is None:
            return R
        # the frame-to-frame estimate had drifted and we are actually still close to a
        # keyframe: take the correction, but do not store a near-duplicate
        if self.nearest_keyframe(R_matched)[1] < KEYFRAME_ANGLE_DEG / 2:
            return R_matched
        self.add_keyframe(R_matched, feats, gray)
        return R_matched

    def refine(self, gray, R):
        """Correct a drifting rotation estimate by matching against the nearest keyframe."""
        if not self.rotations:
            return None
        k, _ = self.nearest_keyframe(R)
        return self._match_rotation(k, self.features(gray))

    def _match_rotation(self, k, feats):
        H, _ = self.match(self.keyframe_features(k), feats)
        if H is None:
            return None
        R_rel = rotation_from_homography(H, self.K, MAX_ROTATION_RESIDUAL)
        return None if R_rel is None else R_rel @ self.rotations[k]

    def keyframe_features(self, k):
        return self.points[k], self.descriptors[k]

    def yaw_span(self):
        if not self.rotations:
            return 0.0, 0.0
        yaws = [yaw_of(R) for R in self.rotations]
        return min(yaws), max(yaws)

    def sweep_axis(self):
        """0 if the sweep went mostly left/right in the image, 1 if up/down.

        A phone streaming in portrait shows a left/right sweep as up/down motion."""
        if len(self.rotations) < 2:
            return 0
        a = np.array([R.T @ np.array([0, 0, 1.0]) for R in self.rotations])
        spread = [np.ptp(np.arctan2(a[:, i], a[:, 2])) for i in (0, 1)]
        return int(np.argmax(spread))

    def sweep_angle(self, R, axis=None):
        """Signed angle of a view along the sweep direction (degrees)."""
        axis = self.sweep_axis() if axis is None else axis
        a = R.T @ np.array([0, 0, 1.0])
        return float(np.degrees(np.arctan2(a[axis], a[2])))

    def sweep_span(self):
        if not self.rotations:
            return 0.0, 0.0
        axis = self.sweep_axis()
        angles = [self.sweep_angle(R, axis) for R in self.rotations]
        return min(angles), max(angles)

    # ---------- using the map ----------

    def relocalize(self, gray, R_guess=None):
        """Camera rotation of this frame from the keyframes, or None.

        With a rough guess only the nearest few keyframes are tried; without one, all."""
        if not self.rotations:
            return None
        feats = self.features(gray)
        if feats is None:
            return None
        if R_guess is None:
            order = range(len(self.rotations))
        else:
            order = np.argsort([angle_between(R_guess, Rk) for Rk in self.rotations])[:RELOC_CANDIDATES]
        best, best_inliers = None, RELOC_MIN_INLIERS - 1
        for k in order:
            H, inliers = self.match(self.keyframe_features(k), feats)
            if H is None or inliers <= best_inliers:
                continue
            R_rel = rotation_from_homography(H, self.K, MAX_ROTATION_RESIDUAL)
            if R_rel is not None:
                best, best_inliers = R_rel @ self.rotations[k], inliers
        return best

    def rotation_from_target(self, H_target):
        """Camera rotation implied by the tracked target pose (reference -> current frame)."""
        return rotation_from_homography(H_target @ np.linalg.inv(self.H0), self.K)

    def _target_rays(self, w_ref, h_ref):
        corners = np.float64([[0, 0, 1], [w_ref, 0, 1], [w_ref, h_ref, 1], [0, h_ref, 1]]).T
        rays = self.Kinv @ (self.H0 @ corners)
        rays /= np.linalg.norm(rays, axis=0)
        return rays

    def project_target(self, R, w_ref, h_ref):
        """Target corners (4,1,2) in the current frame, or None if any corner is behind the camera."""
        cam = R @ self._target_rays(w_ref, h_ref)
        if (cam[2] < 0.05).any():
            return None
        px = self.K @ cam
        return (px[:2] / px[2]).T.reshape(4, 1, 2).astype(np.float32)

    def target_direction(self, R, w_ref, h_ref):
        """Unit 2D screen direction towards the target centre (works when it is behind)."""
        c = R @ self._target_rays(w_ref, h_ref).mean(axis=1)
        d = c[:2]
        n = np.linalg.norm(d)
        return None if n < 1e-6 else d / n

    # ---------- persistence ----------

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            K=self.K, H0=self.H0, frame_size=np.int32(self.frame_size), mask_top=self.mask_top,
            rotations=np.float64(self.rotations),
            counts=np.int32([len(p) for p in self.points]),
            points=np.vstack(self.points), descriptors=np.vstack(self.descriptors),
            thumbs=np.stack(self.thumbs), created=time.time(),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path)
        m = cls(z["K"], tuple(z["frame_size"]), z["H0"], float(z["mask_top"]))
        offsets = np.concatenate([[0], np.cumsum(z["counts"])])
        for i, R in enumerate(z["rotations"]):
            a, b = offsets[i], offsets[i + 1]
            m.rotations.append(R)
            m.points.append(z["points"][a:b])
            m.descriptors.append(z["descriptors"][a:b])
            m.thumbs.append(z["thumbs"][i])
        return m
