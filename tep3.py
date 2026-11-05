import atexit
import cv2
import csv
import numpy as np
import os
import sys
import time

import calibration
import pipeline

# =====================================
# CONFIGURATION
# =====================================



# CHANGE THE URL FIRST  
# Use the Camera stream URL you see on your phone in DroidCam for Android Install It First In Your Device
# after you install DroidCam you will get Wifi Ip if your pc and mobile is both connected in same wifi you can copy the URL and keep it here this way you can get the live feed but below you also have to change the refence image and overlay image

# ================================
# ================================
URL = "http://192.168.1.7:4747/video"

# You can also pass the stream URL or a video file on the command line:
#   python tep3.py http://192.168.1.7:4747/video
#   python tep3.py video.mp4
# Options:
#   --calibrate   start a calibration sweep as soon as the target is found
#                 (or press 'c' in the video window at any time)
#   --no-map      ignore a saved calibration map
#   --async       run re-anchoring SIFT in the background for video files too (on for live
#                 streams; off for files by default so replays give repeatable results)
#   --sync        run SIFT on the main loop even for live streams
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = {a for a in sys.argv[1:] if a.startswith("--")}

if ARGS:
    URL = ARGS[0]

IS_FILE = os.path.isfile(URL)

# ---- run log ----
# Every run writes runs/<timestamp>/: log.csv (one row per frame), input.mp4 (the raw
# camera frames, so a live session can be replayed offline) and snapshots/ (a frame
# each time tracking is lost or regained). Prints a status line about once a second.
LOG_RUNS = True
RECORD_INPUT = True
MAX_SNAPSHOTS = 300



# So the Problem right now is the images which has low keypoints and descriptors have slow processing power and recognition power

# Change the reference.jpg image to the object on which you want to overlay
REFERENCE_IMAGE = "reference.jpg"
# Change the Overlay.png image to the ScreenShot you want to show over the reference image 
OVERLAY_IMAGE = "overlay.png"
FRAME_RESIZE = (960,540)

# a far-away target only yields a handful of points, so keep these low;
# the geometric sanity check is what rejects false detections
MIN_TRACKED_POINTS = 12
FB_CHECK_INTERVAL = 5
SMOOTH_ALPHA = 0.85

# SIFT detection must agree with at least this many distinct RANSAC inliers
MIN_DETECT_INLIERS = 12

# re-run SIFT every N frames while tracking to stop optical flow drift
REDETECT_INTERVAL = 15

# Frame scales tried for SIFT detection, cheapest first (e.g., 0.5 means half resolution).
# Half resolution is fast but misses a far-away target, so fall back to full resolution.
SIFT_SCALES = (0.5, 1.0)

# Reference features are extracted at these scales too, so a far-away target
# (which only shows coarse detail) still has reference features to match
REF_PYRAMID = (1.0, 0.5, 0.25)

# smallest on-screen target area (px) that still gets an overlay
MIN_OVERLAY_AREA = 400

# ---- camera motion ("visual gyro") ----
# Background pixels all over the frame are tracked to measure how the camera itself moved.
# That predicts where the target is while it is blurred, lost or even out of view, so
# tracking survives shakes and the search after a loss looks in the right place.
MOTION_SCALE = 0.5          # estimated on a downscaled frame: cheap, and big jumps become small
MOTION_MAX_POINTS = 300
MOTION_MIN_POINTS = 80      # re-seed background points when fewer survive
PREDICT_MAX_FRAMES = 90     # trust the prediction ~3 s after losing the target, then search everywhere
COAST_FRAMES = 10           # keep drawing at the predicted spot for this many frames without a fix

# DroidCam stamps "using droidcam.app" across the top. It does not move with the camera,
# so background points on it would drag the motion estimate towards zero.
MOTION_MASK_TOP = 0.12      # fraction of frame height ignored for camera motion

# ---- tracking density ----
# SIFT inliers alone are often only ~20 points, which keeps forcing expensive
# re-anchors. After every fix, extra corners inside the target are added to the tracker.
TARGET_TRACK_POINTS = 120

# ---- search while lost ----
LOST_FAST_FRAMES = 5        # search every frame right after a loss, for a quick recovery
LOST_DETECT_EVERY = 3       # then only every Nth frame, so a long loss does not tank the FPS
BLUR_RATIO = 0.2            # frames this much blurrier than a typical tracked frame skip SIFT
SAFETY_DETECT_EVERY = 15    # full-frame check every N frames while predicted out of view

# ---- background detection ----
# SIFT takes 50-130 ms; in the background, tracking keeps running meanwhile and the result
# is carried forward to the current frame with optical flow when it arrives.
ASYNC_DETECT = ("--async" in FLAGS or not IS_FILE) and "--sync" not in FLAGS
# Results from more than this many frames ago are discarded. A far-away target needs the
# full-resolution pass and took 13-18 frames in tests, so keep this generous.
MAX_DETECT_AGE = 30
SYNC_SEARCH_FRAMES = 15     # full-frame searches run immediately for this many frames after a loss
FULL_RES_EVERY = 3           # during a loss, only every Nth full-frame search also runs at full resolution

# ---- stream ----
STREAM_TIMEOUT = 3.0        # seconds without frames before reconnecting to the stream
STREAM_RECONNECTS = 3       # give up after this many failed reconnects

# ---- scene calibration (see calibration.py) ----
# A sweep around the target is saved per reference image and loaded on the next run.
CALIBRATION_DIR = os.environ.get("STITCH_CALIB_DIR","calib")
RELOC_INTERVAL = 10         # while lost, re-check the camera rotation against the map every N frames
MAP_CHECK_EVERY = 30        # while tracking, check every N frames that the map agrees with the target
MAP_AGREE_DEG = 8.0

# detection cooldown
DETECT_BASE_INTERVAL = 3
DETECT_MAX_INTERVAL = 30

detect_interval = DETECT_BASE_INTERVAL
frames_since_detect = 0

# =====================================
# LOAD IMAGES
# =====================================

reference = cv2.imread(REFERENCE_IMAGE,0)
overlay = cv2.imread(OVERLAY_IMAGE)

if reference is None or overlay is None:
    print("Image load error")
    exit()

h_ref,w_ref = reference.shape
overlay = cv2.resize(overlay,(w_ref,h_ref))

# warped separately so black pixels inside the overlay are not treated as transparent
overlay_mask = np.full((h_ref,w_ref),255,np.uint8)

ref_corners=np.float32([
    [0,0],
    [w_ref,0],
    [w_ref,h_ref],
    [0,h_ref]
]).reshape(-1,1,2)

# =====================================
# SIFT INITIALIZATION
# =====================================

sift = cv2.SIFT_create(1200)

# keypoint positions are stored in full-resolution reference coordinates
kp_ref_pts=[]
des_ref_levels=[]

for s in REF_PYRAMID:
    level = reference if s == 1.0 else cv2.resize(reference,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    kps,des = sift.detectAndCompute(level,None)
    if des is None:
        continue
    kp_ref_pts += [(kp.pt[0]/s,kp.pt[1]/s) for kp in kps]
    des_ref_levels.append(des)

kp_ref_pts=np.float32(kp_ref_pts)
des_ref=np.vstack(des_ref_levels)

index_params=dict(algorithm=1,trees=5)
search_params=dict(checks=50)

flann=cv2.FlannBasedMatcher(index_params,search_params)

# =====================================
# OPTICAL FLOW SETTINGS
# =====================================

lk_params=dict(
    winSize=(21,21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,30,0.01)
)

# large windows and pyramid: background motion between frames can be big
motion_lk_params=dict(
    winSize=(21,21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,20,0.03)
)

prev_pts=None
ref_pts=None
prev_gray=None

# best estimate of the target corners in the current frame (may be off-screen)
corners_est=None
lost_frames=0

# corners_est could not be moved with the camera at some point since the last fix
est_stale=False
# consecutive failed searches around the predicted position
search_misses=0
# frames since SIFT last confirmed the pose while tracking
frames_since_anchor=0

# ---- scene map state ----
map_path=os.path.join(CALIBRATION_DIR,os.path.splitext(os.path.basename(REFERENCE_IMAGE))[0]+".npz")
scene_map=None
calibrating=False
calibrate_requested="--calibrate" in FLAGS
# camera rotation relative to the map (calibration start), and whether it is trustworthy
R_cur=None
R_stale=True
K_cam=calibration.intrinsics(*FRAME_RESIZE)
# A loaded map is only used after this session's camera view was matched against it:
# the same reference image somewhere else, or walking around instead of turning in
# place, would make its predictions wrong.
map_ok=False
reloc_fails=0

if "--no-map" not in FLAGS and os.path.isfile(map_path):
    scene_map=calibration.SceneMap.load(map_path)
    K_cam=scene_map.K
    lo,hi=scene_map.sweep_span()
    print(f"Loaded scene map {map_path}: {len(scene_map.rotations)} keyframes, sweep {lo:.0f}..{hi:.0f} deg")


def finish_calibration():
    global calibrating
    calibrating=False
    if len(scene_map.rotations)<2:
        print("Calibration cancelled: not enough keyframes")
        return
    scene_map.save(map_path)
    lo,hi=scene_map.sweep_span()
    print(f"Calibration saved to {map_path}: {len(scene_map.rotations)} keyframes, sweep {lo:.0f}..{hi:.0f} deg")

prev_small=None
motion_pts=None
motion_mask=None

# typical sharpness of frames where the target was tracked (for blur detection)
sharp_ref=None

frame_id=0

# =====================================
# HOMOGRAPHY VALIDATION
# =====================================

def valid_homography(H):

    if H is None:
        return False

    if np.any(np.isnan(H)) or np.any(np.isinf(H)):
        return False

    det=np.linalg.det(H)

    if abs(det)<1e-6 or abs(det)>1e6:
        return False

    return True


# RANSAC happily fits a degenerate homography to unrelated matches (e.g. the whole
# reference squashed onto a few pixels). A real view of a flat target projects to a
# convex quad of plausible size and proportions.

def sane_quad(H,frame_shape):

    quad=cv2.perspectiveTransform(ref_corners,H).reshape(4,2)

    if not cv2.isContourConvex(quad.reshape(-1,1,2)):
        return False

    area=cv2.contourArea(quad)

    if area<MIN_OVERLAY_AREA or area>4*frame_shape[0]*frame_shape[1]:
        return False

    sides=np.linalg.norm(quad-np.roll(quad,1,axis=0),axis=1)

    return sides.min()>8 and sides.max()/sides.min()<6


# =====================================
# GEOMETRIC CONSISTENCY FILTER
# =====================================

# The target is planar, so frame-to-frame motion of its points is a homography.
# Fitting one with RANSAC keeps points that move consistently under rotation,
# zoom and tilt too (a median-translation check rejects those).

def geometric_filter(prev_pts,new_pts,ref_pts):

    if len(new_pts) < 4:
        return new_pts[:0], ref_pts[:0], prev_pts[:0]

    _,mask = cv2.findHomography(prev_pts,new_pts,cv2.RANSAC,3.0)

    if mask is None:
        return new_pts[:0], ref_pts[:0], prev_pts[:0]

    mask = mask.ravel().astype(bool)

    return new_pts[mask], ref_pts[mask], prev_pts[mask]


# =====================================
# DETECTION FUNCTION
# =====================================

# Returns (ref_pts, frame_pts) of RANSAC inliers, or None if the target was not found.
# roi=(x,y,w,h) restricts the search to that part of the frame.

# per-frame detection cost, for the run log
detect_calls=0
detect_ms=0.0

def detect(gray,roi=None,sift=sift,flann=flann,scales=SIFT_SCALES):

    global detect_calls,detect_ms

    t0=time.perf_counter()
    detect_calls+=1

    x0,y0 = 0,0

    if roi is not None:
        x0,y0,w,h = roi
        gray = gray[y0:y0+h,x0:x0+w]

    result=None

    for scale in scales:

        found = detect_at_scale(gray,scale,sift,flann)

        if found is not None:
            ref_pts,frame_pts = found
            frame_pts += np.float32([x0,y0])
            result = ref_pts,frame_pts
            break

    detect_ms+=1000*(time.perf_counter()-t0)

    return result


def detect_at_scale(gray,scale,sift,flann):

    # 1. Downscale the frame to drastically reduce SIFT processing time
    small_gray = gray if scale == 1.0 else cv2.resize(gray, (0, 0), fx=scale, fy=scale)

    # 2. Run SIFT on the smaller image
    kp_frame,des_frame=sift.detectAndCompute(small_gray,None)

    if des_frame is None or len(des_frame)<2:
        return None

    # Match reference -> frame. The other direction would be faster (one prebuilt index),
    # but a reference feature appears at several pyramid levels, so a frame descriptor's
    # two nearest neighbours are often the same feature and the ratio test rejects it.
    matches=flann.knnMatch(des_ref,des_frame,k=2)

    good=[]

    for pair in matches:
        if len(pair)==2 and pair[0].distance<0.7*pair[1].distance:
            good.append(pair[0])

    if len(good)<MIN_DETECT_INLIERS:
        return None

    ref_pts=kp_ref_pts[[m.queryIdx for m in good]].reshape(-1,1,2)

    # 3. Rescale the points back to original resolution so optical flow and homography work properly
    frame_pts=np.float32(
        [kp_frame[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    frame_pts /= scale

    # 4. Ratio test alone lets false positives through; require a consistent homography
    H,mask=cv2.findHomography(ref_pts,frame_pts,cv2.RANSAC,5.0)

    if not valid_homography(H) or mask is None or not sane_quad(H,gray.shape):
        return None

    mask=mask.ravel().astype(bool)

    # pyramid levels can match the same frame keypoint more than once; count it once
    _,unique_idx=np.unique(
        np.round(frame_pts[mask].reshape(-1,2)),
        axis=0,
        return_index=True
    )

    if len(unique_idx)<MIN_DETECT_INLIERS:
        return None

    return ref_pts[mask][unique_idx],frame_pts[mask][unique_idx]


# =====================================
# BACKGROUND DETECTION
# =====================================

class DetectRequest:
    # what is needed to move a result from the frame it was computed on to a later one
    def __init__(self,gray,frame_id,scales):
        self.gray=gray
        self.frame_id=frame_id
        self.scales=scales
        self.motion=np.eye(3)        # camera motion accumulated since that frame
        self.motion_known=True

# The worker gets its own SIFT and matcher: OpenCV feature objects are not safe to use
# from two threads at once, and the main loop still runs its own searches.
def make_background_detect():
    bg_sift=cv2.SIFT_create(1200)
    bg_flann=cv2.FlannBasedMatcher(index_params,search_params)
    return lambda gray,roi,req: detect(gray,roi,bg_sift,bg_flann,req.scales)

detector = pipeline.DetectWorker(make_background_detect()) if ASYNC_DETECT else None

# set by request_detect: whether a detection actually finished this frame
# (in the background a None result can also mean "not done yet")
detect_completed=False

# background detection outcomes, printed at exit
detect_stats={"completed":0,"found":0,"forwarded":0,"age":[]}


def advance_detect_request(M):
    req = detector.in_flight() if detector is not None else None
    if req is not None:
        if M is None:
            req.motion_known=False
        else:
            req.motion=M @ req.motion


# Move points found on an earlier frame to the current one with optical flow, starting
# from where the accumulated camera motion says they went.

def forward_detection(req,found,gray):

    ref_pts,pts=found
    pts=pts.reshape(-1,1,2).astype(np.float32)

    if frame_id-req.frame_id > MAX_DETECT_AGE or not req.motion_known:
        return None

    guess=cv2.perspectiveTransform(pts,req.motion).astype(np.float32)

    nxt,status,_=cv2.calcOpticalFlowPyrLK(
        req.gray,gray,pts,guess,flags=cv2.OPTFLOW_USE_INITIAL_FLOW,**lk_params
    )

    ok=status.ravel()==1

    if ok.sum()<MIN_DETECT_INLIERS:
        return None

    return ref_pts.reshape(-1,2)[ok],nxt.reshape(-1,2)[ok]


# Called once per frame: a background detection that has finished, moved to this frame
# (or None). Sets detect_completed.

def poll_detection(gray):

    global detect_completed

    detect_completed=False

    if detector is None:
        return None

    done=detector.take()

    if done is None:
        return None

    detect_completed=True
    req,found=done
    detect_stats["completed"]+=1

    if found is None:
        return None

    detect_stats["found"]+=1
    detect_stats["age"].append(frame_id-req.frame_id)
    moved=forward_detection(req,found,gray)
    detect_stats["forwarded"]+=moved is not None

    # found, but too old or lost on the way to this frame: that is not a miss (it must not
    # count towards backing off the search), just a detection to run again
    if moved is None:
        detect_completed=False

    return moved


# Detection entry point for the main loop. With background=True and the worker enabled,
# it starts a detection on this frame if the worker is free and returns None;
# poll_detection() delivers the result on a later frame. Otherwise it runs now.
#
# What goes to the background: re-anchors while tracking (the overlay keeps moving
# smoothly; that is where the 50-130 ms SIFT stalls used to show) and full-frame searches
# (in a live run these froze the display for 0.1-1 s, mostly while the target was not even
# in view). Searches in a small window around a confident prediction run now: they are
# cheap and a late result there only slows recovery.
#
# Full-frame searches are the expensive ones (two passes, half and full resolution).
# During a long loss most of them use half resolution only; every FULL_RES_EVERY-th one
# also tries full resolution, which a far-away target needs.

full_searches=0

def request_detect(gray,roi=None,background=False):

    global detect_completed,full_searches

    scales=SIFT_SCALES

    if roi is None:
        if full_searches % FULL_RES_EVERY:
            scales=SIFT_SCALES[:1]
        full_searches+=1

    if detector is None or not background:
        detect_completed=True
        return detect(gray,roi,scales=scales)

    if not detector.busy():
        g=gray.copy()
        detector.submit(DetectRequest(g,frame_id,scales),g,roi)

    return None


# =====================================
# CAMERA MOTION (VISUAL GYRO)
# =====================================

# Returns (M, pts): M is the 3x3 frame-to-frame camera motion in full-resolution
# coordinates (or None if it could not be measured), pts are the background points
# to keep tracking in the next frame.

def to_full_res(A):
    # rescale a 2x3 motion measured on the small frame to full resolution
    S=np.diag([MOTION_SCALE,MOTION_SCALE,1.0])
    return np.linalg.inv(S) @ np.vstack([A,[0,0,1]]) @ S


def estimate_camera_motion(prev_small,small,pts):

    global motion_mask

    top=int(MOTION_MASK_TOP*small.shape[0])

    if motion_mask is None:
        motion_mask=np.full(small.shape,255,np.uint8)
        motion_mask[:top]=0

    # Phase correlation finds the dominant whole-image shift even across big jumps and
    # motion blur, where LK on its own loses the points. It seeds LK, and is the
    # fallback when LK cannot produce a consistent motion.
    # (Both frames are cropped the same way, so the shift is unaffected.)
    a,b=np.float32(prev_small[top:]),np.float32(small[top:])
    window=cv2.createHanningWindow(a.shape[::-1],cv2.CV_32F)
    (dx,dy),response=cv2.phaseCorrelate(a,b,window)
    shift=None if response<0.05 else np.float32([[1,0,dx],[0,1,dy]])

    if pts is None or len(pts)<MOTION_MIN_POINTS:
        pts=cv2.goodFeaturesToTrack(prev_small,MOTION_MAX_POINTS,0.01,8,mask=motion_mask)
        if pts is None:
            return (None if shift is None else to_full_res(shift)),None

    guess=pts+np.float32([dx,dy]) if shift is not None else pts.copy()

    nxt,status,_=cv2.calcOpticalFlowPyrLK(
        prev_small,
        small,
        pts,
        guess,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
        **motion_lk_params
    )

    ok=status.ravel()==1

    if ok.sum()<10:
        return (None if shift is None else to_full_res(shift)),None

    # rotation + scale + translation: enough for hand shake, and stable to estimate
    A,inliers=cv2.estimateAffinePartial2D(
        pts[ok],
        nxt[ok],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0
    )

    if A is None or inliers.sum()<max(10,0.3*ok.sum()):
        return (None if shift is None else to_full_res(shift)),None

    keep=nxt[ok][inliers.ravel()==1].reshape(-1,1,2)

    return to_full_res(A),keep


# =====================================
# TARGET POINT SEEDING
# =====================================

# Add strong corners found inside the target (shrunk a little so the background at the
# edges is not picked up), with their reference coordinates from the current pose.
# More points means steadier tracking and far fewer SIFT re-anchors.

def seed_target_points(gray,H,pts,refs):

    missing=TARGET_TRACK_POINTS-len(pts)

    if missing<10:
        return pts,refs

    quad=cv2.perspectiveTransform(ref_corners,H).reshape(4,2)
    centre=quad.mean(axis=0)
    quad=centre+0.85*(quad-centre)

    mask=np.zeros(gray.shape,np.uint8)
    cv2.fillConvexPoly(mask,quad.astype(np.int32),255)

    # keep new corners away from the points already tracked
    for p in pts.reshape(-1,2):
        cv2.circle(mask,(int(p[0]),int(p[1])),6,0,-1)

    new=cv2.goodFeaturesToTrack(gray,missing,0.01,7,mask=mask)

    if new is None:
        return pts,refs

    new=new.reshape(-1,2).astype(np.float32)
    new_refs=cv2.perspectiveTransform(new.reshape(-1,1,2),np.linalg.inv(H)).reshape(-1,2)

    return np.vstack([pts.reshape(-1,2),new]),np.vstack([refs.reshape(-1,2),new_refs])


def visible_fraction(quad,frame_shape):

    q=quad.reshape(-1,2).astype(np.float32)
    area=cv2.contourArea(q)

    if area<=0:
        return 0.0

    h,w=frame_shape[:2]
    rect=np.float32([[0,0],[w,0],[w,h],[0,h]])

    try:
        inter,_=cv2.intersectConvexConvex(q,rect)
    except cv2.error:
        return 0.0

    return inter/area


# search window around a quad: padded bounding box clipped to the frame

def roi_around(quad,frame_shape):

    x,y,w,h = cv2.boundingRect(quad.reshape(-1,1,2).astype(np.float32))
    pad_x,pad_y = int(0.3*w)+16,int(0.3*h)+16
    x1,y1 = max(x-pad_x,0),max(y-pad_y,0)
    x2,y2 = min(x+w+pad_x,frame_shape[1]),min(y+h+pad_y,frame_shape[0])

    if x2-x1<=32 or y2-y1<=32:
        return None

    return (x1,y1,x2-x1,y2-y1)


# arrow at the frame edge pointing to where the off-screen target is; give either the
# predicted quad or (from the scene map, which also works when it is behind) a direction.
# Returns the unit direction drawn, or None.

def draw_offscreen_arrow(frame,quad=None,direction=None):

    h,w=frame.shape[:2]
    centre=np.float32([w/2,h/2])

    d = np.float32(direction) if direction is not None else quad.reshape(-1,2).mean(axis=0)-centre
    n=np.linalg.norm(d)

    if n<1e-6 or (direction is None and n<1):
        return None

    d=d/n
    tip=centre+d*min(w,h)*0.45
    tail=centre+d*min(w,h)*0.30

    cv2.arrowedLine(frame,tuple(int(v) for v in tail),tuple(int(v) for v in tip),
                    (0,200,255),4,tipLength=0.4)

    return d


# progress bar of the calibration sweep: covered yaw range out of +-90 degrees

def draw_calibration_hud(frame,scene_map,waiting):

    h,w=frame.shape[:2]
    x0,x1,y=60,w-60,h-40

    if waiting:
        text="CALIBRATION: point the camera at the target to start"
    else:
        lo,hi=scene_map.sweep_span()
        text=(f"CALIBRATING: turn slowly left and right, stay in place  "
              f"{hi-lo:.0f}/{calibration.SWEEP_TARGET_DEG:.0f} deg, {len(scene_map.rotations)} keyframes  (c = finish)")
        cv2.rectangle(frame,(x0,y-8),(x1,y+8),(80,80,80),-1)
        to_x=lambda yaw: int(x0+(x1-x0)*(np.clip(yaw,-90,90)+90)/180)
        cv2.rectangle(frame,(to_x(lo),y-8),(to_x(hi),y+8),(0,200,0),-1)
        for R in scene_map.rotations:
            cv2.line(frame,(to_x(scene_map.sweep_angle(R)),y-12),(to_x(scene_map.sweep_angle(R)),y+12),(255,255,255),2)
        if R_cur is not None:
            cv2.circle(frame,(to_x(scene_map.sweep_angle(R_cur)),y),9,(0,200,255),-1)

    cv2.putText(frame,text,(20,h-70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)


# =====================================
# STREAM
# =====================================

def stop_requested():
    return cv2.waitKey(1)&0xFF==ord('q')


def open_source():

    if IS_FILE:
        return cv2.VideoCapture(URL)

    # without timeouts a stalled stream blocks read() forever
    c=cv2.VideoCapture(URL,cv2.CAP_FFMPEG,[
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,int(1000*STREAM_TIMEOUT)
    ])
    c.set(cv2.CAP_PROP_BUFFERSIZE,1)
    return c


# A video file is read frame by frame on the main loop (replays must not skip frames);
# a live stream is read on a background thread that always offers the newest frame.
grabber=None

if IS_FILE:
    cap=open_source()
else:
    grabber=pipeline.FrameGrabber(open_source,FRAME_RESIZE)
    cap=grabber.cap

if not cap.isOpened():
    print(f"Could not open {URL}. Check the DroidCam URL or the video path.")
    exit()

# frames are resized to FRAME_RESIZE; a different aspect ratio here means they get stretched
print(f"Source resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

frame_seq=0
frames_skipped=0
skipped_now=0

prev_time=time.time()

# =====================================
# RUN LOG
# =====================================

LOG_FIELDS=["frame","t","fps","proc_ms","state","target_pts","motion_ok","motion_pts",
            "tx","ty","rot_deg","detect_calls","detect_ms","lost_frames","visible","stale",
            "yaw","calibrating","arrow_x","arrow_y","corners_cx","corners_cy","skipped"]

log_file=None
input_writer=None
snapshots=0
run_start=time.time()
recent=[]
last_state=None

if LOG_RUNS:
    run_dir=os.path.join("runs",time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(run_dir,"snapshots"),exist_ok=True)
    log_file=open(os.path.join(run_dir,"log.csv"),"w",newline="")
    log_writer=csv.DictWriter(log_file,fieldnames=LOG_FIELDS)
    log_writer.writeheader()
    if RECORD_INPUT:
        # MJPG/AVI stays readable even if the run is killed before the file is closed;
        # encoding runs on a background thread
        input_writer=pipeline.AsyncWriter(os.path.join(run_dir,"input.avi"),
                                          cv2.VideoWriter_fourcc(*"MJPG"),30,FRAME_RESIZE)
    print(f"Logging this run to {run_dir}")


# also runs on Ctrl+C, so the log and recording are always closed properly
def close_run():
    global log_file,input_writer
    if log_file is not None:
        log_file.close()
        log_file=None
        print(f"Run saved to {run_dir}")
    if input_writer is not None:
        input_writer.release()
        input_writer=None

atexit.register(close_run)

reconnects=0

# =====================================
# MAIN LOOP
# =====================================

while True:

    if grabber is None:

        ret,frame=cap.read()

        # the video file has ended
        if not ret:
            break

        frame=cv2.resize(frame,FRAME_RESIZE)

    else:

        new_frame,new_seq=grabber.read(frame_seq)

        if new_frame is None:

            if stop_requested():
                break

            # no frames: the stream died (DroidCam closed, Wi-Fi dropped); reconnect, then give up
            if time.time()-grabber.last_frame_time > STREAM_TIMEOUT:
                if reconnects >= STREAM_RECONNECTS:
                    print("Stream lost, giving up.")
                    break
                reconnects+=1
                print(f"No frames for {STREAM_TIMEOUT:.0f}s, reconnecting ({reconnects}/{STREAM_RECONNECTS})...")
                grabber.reconnect()
            continue

        # frames that arrived while the last one was processed are skipped (newest wins)
        skipped_now=new_seq-frame_seq-1
        frames_skipped+=skipped_now
        frame,frame_seq=new_frame,new_seq
        reconnects=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if input_writer is not None:
        input_writer.write(frame)

    frame_start=time.perf_counter()
    detect_calls,detect_ms=0,0.0

    frame_id+=1
    H=None
    fixed=False
    state="lost"
    arrow_dir=None
    target_behind=False

    # ---------------------------------
    # CAMERA MOTION
    # ---------------------------------

    small=cv2.resize(gray,None,fx=MOTION_SCALE,fy=MOTION_SCALE)

    M=None

    if prev_small is not None:
        M,motion_pts=estimate_camera_motion(prev_small,small,motion_pts)

    prev_small=small

    # Laplacian variance drops sharply on motion-blurred frames
    sharpness=cv2.Laplacian(small,cv2.CV_32F).var()
    blurry=sharp_ref is not None and sharpness<BLUR_RATIO*sharp_ref

    # move the last known target position along with the camera
    if corners_est is not None:
        if M is not None:
            corners_est=cv2.perspectiveTransform(corners_est,M)
        else:
            # the camera moved by an unknown amount: the estimate can no longer be trusted
            est_stale=True

    # the same camera motion, as a rotation (for the scene map / calibration)
    if R_cur is not None:
        R_step = calibration.rotation_from_motion(M,K_cam) if M is not None else None
        if R_step is not None:
            R_cur = R_step @ R_cur
        else:
            R_stale = True

    # a background detection that finished since the last frame, moved to this frame
    advance_detect_request(M)
    found=poll_detection(gray)
    good_new=None

    # ---------------------------------
    # OPTICAL FLOW TRACKING
    # ---------------------------------

    if prev_pts is not None:

        # start the search where camera motion says the points went, so a fast
        # move or shake does not throw the points out of LK's search window
        guess=cv2.perspectiveTransform(prev_pts,M) if M is not None else prev_pts.copy()

        next_pts,status,_=cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts,
            guess,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            **lk_params
        )

        good_new=next_pts[status==1]
        good_prev=prev_pts[status==1]
        good_ref=ref_pts[status==1]

        # ---------------------------------
        # FORWARD-BACKWARD CHECK
        # ---------------------------------

        if frame_id % FB_CHECK_INTERVAL == 0:

            back_pts,status_back,_=cv2.calcOpticalFlowPyrLK(
                gray,
                prev_gray,
                next_pts,
                prev_pts.copy(),
                flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                **lk_params
            )

            fb_error=np.linalg.norm(prev_pts-back_pts,axis=2)

            mask=(fb_error<1.5) & (status==1)

            good_new=next_pts[mask]
            good_prev=prev_pts[mask]
            good_ref=ref_pts[mask]

        # ---------------------------------
        # GEOMETRIC CONSISTENCY
        # ---------------------------------

        good_new,good_ref,good_prev = geometric_filter(good_prev,good_new,good_ref)

        # ---------------------------------
        # PERIODIC RE-ANCHOR (stops drift)
        # ---------------------------------

        # A fix taken from a partly visible or far target leaves few points, and the
        # corners extrapolated from them can be off; re-anchor sooner in that case.
        few_points = len(good_new) < 3*MIN_TRACKED_POINTS

        frames_since_anchor += 1

        # SIFT on a blurred frame mostly fails, so a re-anchor that falls due on one
        # waits for the next sharp frame instead of being skipped
        anchor_due = frames_since_anchor >= REDETECT_INTERVAL or (few_points and frames_since_anchor >= 3)

        if found is None and anchor_due and not blurry:

            frames_since_anchor = 0

            # only search around where the target is now: cheaper, and a small
            # far-away target fills more of the search window
            roi = roi_around(corners_est,gray.shape) if corners_est is not None else None

            found = request_detect(gray,roi,background=True)

        if found is None and len(good_new)<MIN_TRACKED_POINTS:

            prev_pts=None
            good_new=None

    # ---------------------------------
    # SEARCH (target not tracked)
    # ---------------------------------

    if prev_pts is None and found is None:

        lost_frames += 1

        # ---- scene map: where is the target, even when it is out of view? ----
        if scene_map is not None and not calibrating:

            # check the rotation against the keyframes right after a loss, then every
            # RELOC_INTERVAL frames (camera motion keeps it updated in between); if the
            # rotation is unknown, keep trying, backing off while it keeps failing
            # (e.g. this is not the calibrated place) so it does not eat the frame rate
            unknown = R_cur is None or R_stale or not map_ok
            reloc_every = min(3*2**reloc_fails,30) if unknown else RELOC_INTERVAL
            reloc_due = (lost_frames-1) % reloc_every == 0

            if reloc_due and not blurry:
                R_new = scene_map.relocalize(gray, None if unknown else R_cur)
                if R_new is not None:
                    if not map_ok:
                        print("Scene map matched: using it to locate the target")
                    R_cur,R_stale = R_new,False
                    map_ok,reloc_fails = True,0
                elif unknown:
                    reloc_fails += 1

            # only trust map predictions once the scene has actually been recognised
            if map_ok and R_cur is not None and not R_stale:
                corners_est = scene_map.project_target(R_cur,w_ref,h_ref)
                target_behind = corners_est is None
                est_stale = False

        # predictions can be wrong (drift, a bad relocalization): while they say the target
        # is out of view, still look over the whole frame now and then
        # Right after a loss, search now: the target is probably close and a fast recovery
        # matters most. In a long loss, full-frame searches go to the background: the live
        # freezes they caused happened while the target was not even in view.
        long_loss = lost_frames > SYNC_SEARCH_FRAMES

        safety_check = not blurry and lost_frames % SAFETY_DETECT_EVERY == 0

        if target_behind:

            # the map says it is behind the camera: nothing to search for
            if safety_check:
                found = request_detect(gray,background=long_loss)

        elif corners_est is not None and (lost_frames <= PREDICT_MAX_FRAMES or (scene_map is not None and not calibrating)):

            should_be_visible = visible_fraction(corners_est,gray.shape) > 0.2

            # SIFT on a smeared frame almost never succeeds and costs ~100 ms; and once a
            # loss drags on, searching every few frames is enough
            search_now = not blurry and (lost_frames <= LOST_FAST_FRAMES or lost_frames % LOST_DETECT_EVERY == 0)

            if search_now and (est_stale or search_misses >= 3):

                # the prediction is unreliable: look everywhere
                found = request_detect(gray,background=long_loss)

            elif search_now and should_be_visible:

                # camera motion tells us roughly where the target is: search only there,
                # and not at all while it is out of view
                roi = roi_around(corners_est,gray.shape)

                if roi is not None:
                    found = request_detect(gray,roi)

            elif safety_check:

                found = request_detect(gray,background=long_loss)

            if detect_completed and found is None and (est_stale or should_be_visible):
                search_misses += 1

        else:

            # no idea where it is: full-frame search with cooldown
            corners_est = None

            frames_since_detect += 1

            if frames_since_detect >= detect_interval:

                frames_since_detect = 0

                found = request_detect(gray,background=long_loss)

            # (a background search reports back on a later frame)
            if detect_completed:
                if found is not None:
                    detect_interval = DETECT_BASE_INTERVAL
                else:
                    detect_interval = min(detect_interval*2, DETECT_MAX_INTERVAL)

    if found is not None:
        good_ref,good_new = found
        good_ref = good_ref.reshape(-1,2)
        good_new = good_new.reshape(-1,2)
        good_prev = None

    # ---------------------------------
    # HOMOGRAPHY
    # ---------------------------------

    if good_new is not None:

        H_candidate,inliers=cv2.findHomography(
            good_ref.reshape(-1,1,2),
            good_new.reshape(-1,1,2),
            cv2.RANSAC,
            5.0
        )

        if valid_homography(H_candidate) and inliers is not None and sane_quad(H_candidate,gray.shape):

            # carry only points that agree with this pose into the next frame
            keep=inliers.ravel().astype(bool)
            good_new=good_new[keep]
            good_ref=good_ref[keep]
            if good_prev is not None:
                good_prev=good_prev[keep]

            corners=cv2.perspectiveTransform(ref_corners,H_candidate)

            # Smooth the projected corners rather than the matrix itself:
            # averaging two homographies element-wise is not a valid blend of poses.
            # corners_est was already moved by the camera motion, so smoothing only
            # removes jitter and does not make the overlay lag behind a moving camera.
            if corners_est is None or lost_frames > 0 or est_stale:
                # first sighting, re-found after a loss, or no trustworthy estimate: snap to it
                alpha = 0.0
            elif good_prev is None:
                # just re-anchored by SIFT: move most of the way to the corrected pose
                alpha = 0.3
            else:
                # Dynamic Anti-Jitter: Check how much the points actually moved
                movement = np.linalg.norm(good_new - good_prev, axis=1).mean()

                if movement < 0.8:
                    # If camera is basically still, LOCK the AR overlay completely (no jitter)
                    alpha = 1.0
                elif movement < 2.5:
                    # If moving slowly, apply HEAVY smoothing
                    alpha = 0.92
                else:
                    # If moving normally, use standard smooth
                    alpha = SMOOTH_ALPHA

            if corners_est is not None:
                corners = alpha*corners_est + (1-alpha)*corners

            corners_est=corners
            lost_frames=0
            est_stale=False
            search_misses=0
            fixed=True
            detect_interval=DETECT_BASE_INTERVAL
            full_searches=0

            # Only seed right after a sharp SIFT fix: reference coordinates of new points
            # come from the current pose, so seeding from a tracked (possibly slightly
            # drifted) pose would bake that error into every new point.
            if found is not None and not blurry:
                good_new,good_ref = seed_target_points(gray,H_candidate,good_new,good_ref)

            sharp_ref = sharpness if sharp_ref is None else 0.95*sharp_ref + 0.05*sharpness

            H=cv2.getPerspectiveTransform(
                ref_corners.reshape(4,2),
                corners_est.reshape(4,2).astype(np.float32)
            )

        prev_pts=good_new.reshape(-1,1,2).astype(np.float32)
        ref_pts=good_ref.reshape(-1,1,2).astype(np.float32)

    prev_gray=gray.copy()

    # ---------------------------------
    # SCENE MAP / CALIBRATION
    # ---------------------------------

    if fixed and (scene_map is not None or calibrate_requested):

        if calibrate_requested and not calibrating:
            # the view where the target is first tracked becomes the map's origin
            scene_map=calibration.SceneMap(K_cam,FRAME_RESIZE,H.copy(),MOTION_MASK_TOP)
            calibrating,calibrate_requested=True,False
            R_cur,R_stale=np.eye(3),False
            map_ok=True
            print("Calibration started: stay in place and turn the camera slowly left and right")

        # the tracked target pins down the camera rotation exactly
        R_t=scene_map.rotation_from_target(H)
        if R_t is not None:
            R_cur,R_stale=R_t,False

            # Now and then check that the map agrees with the target. It will not if the
            # target was moved, or if the camera moved away from where it was calibrated.
            if not calibrating and frame_id % MAP_CHECK_EVERY == 0 and not blurry:
                R_m=scene_map.relocalize(gray,R_t)
                if R_m is not None:
                    agrees=calibration.angle_between(R_m,R_t) < MAP_AGREE_DEG
                    if agrees != map_ok:
                        print("Scene map matched: using it to locate the target" if agrees else
                              "Scene map disagrees with the target (moved?): ignoring the map for now")
                    map_ok=agrees

    if calibrating:

        # lost track of the rotation mid-sweep (target out of view and motion failed):
        # recover it from the keyframes captured so far
        if R_stale and frame_id % 3 == 0 and not blurry:
            R_new=scene_map.relocalize(gray)
            if R_new is not None:
                R_cur,R_stale=R_new,False

        if R_cur is not None and not R_stale and not blurry:
            n_before=len(scene_map.rotations)
            R_cur=scene_map.maybe_add_keyframe(gray,R_cur)

            # between keyframes the frame-to-frame estimate drifts (it was ~15 degrees
            # off in a live sweep); pull it back to the nearest keyframe regularly
            if not fixed and len(scene_map.rotations)==n_before and frame_id % 10 == 0:
                R_new=scene_map.refine(gray,R_cur)
                if R_new is not None:
                    R_cur=R_new

        lo,hi=scene_map.sweep_span()
        if hi-lo >= calibration.SWEEP_TARGET_DEG:
            finish_calibration()

    # ---------------------------------
    # COAST THROUGH SHORT LOSSES
    # ---------------------------------

    # a blurred frame or a quick jerk loses the fix for a moment; keep the overlay
    # where camera motion says the target is instead of letting it blink out
    visible = visible_fraction(corners_est,gray.shape) if corners_est is not None else 0.0

    if fixed:
        state = "detect" if found is not None else "track"

    if H is None and corners_est is not None and not est_stale and lost_frames <= COAST_FRAMES and visible > 0.05:

        state="coast"

        H=cv2.getPerspectiveTransform(
            ref_corners.reshape(4,2),
            corners_est.reshape(4,2).astype(np.float32)
        )

    # ---------------------------------
    # OVERLAY
    # ---------------------------------

    if H is not None:

        area=cv2.contourArea(corners_est.reshape(-1,2).astype(np.float32))

        # only warp and blend inside the target's bounding box, not the whole frame
        x,y,w,h=cv2.boundingRect(corners_est.reshape(-1,1,2).astype(np.float32))
        x1,y1=max(x,0),max(y,0)
        x2,y2=min(x+w,frame.shape[1]),min(y+h,frame.shape[0])

        if area>MIN_OVERLAY_AREA and x2>x1 and y2>y1:

            # same homography, shifted so the box's corner is the origin
            T=np.array([[1,0,-x1],[0,1,-y1],[0,0,1]],np.float64) @ H
            size=(x2-x1,y2-y1)

            warped=cv2.warpPerspective(overlay,T,size)
            mask=cv2.warpPerspective(overlay_mask,T,size)

            np.copyto(frame[y1:y2,x1:x2],warped,where=mask[...,None]>0)

    elif corners_est is not None and not est_stale and visible < 0.05:

        state="offscreen"

        arrow_dir=draw_offscreen_arrow(frame,corners_est)

    elif target_behind:

        # behind the camera there is no projected quad, but the map still knows the direction
        state="offscreen"

        arrow_dir=draw_offscreen_arrow(frame,direction=scene_map.target_direction(R_cur,w_ref,h_ref))

    if calibrating or calibrate_requested:
        draw_calibration_hud(frame,scene_map,waiting=not calibrating)

    # ---------------------------------
    # FPS
    # ---------------------------------

    new_time=time.time()
    fps=1/max(new_time-prev_time,1e-6)
    prev_time=new_time

    cv2.putText(frame,f"FPS:{int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # ---------------------------------
    # RUN LOG
    # ---------------------------------

    if log_file is not None:

        log_writer.writerow({
            "frame":frame_id,
            "t":round(time.time()-run_start,3),
            "fps":round(fps,1),
            "proc_ms":round(1000*(time.perf_counter()-frame_start),1),
            "state":state,
            "target_pts":0 if prev_pts is None else len(prev_pts),
            "motion_ok":int(M is not None),
            "motion_pts":0 if motion_pts is None else len(motion_pts),
            "tx":"" if M is None else round(float(M[0,2]),1),
            "ty":"" if M is None else round(float(M[1,2]),1),
            "rot_deg":"" if M is None else round(float(np.degrees(np.arctan2(M[1,0],M[0,0]))),2),
            "detect_calls":detect_calls,
            "detect_ms":round(detect_ms,1),
            "lost_frames":lost_frames,
            "visible":round(float(visible),2),
            "stale":int(est_stale),
            "yaw":"" if R_cur is None or R_stale else round(calibration.yaw_of(R_cur),1),
            "calibrating":int(calibrating),
            "arrow_x":"" if arrow_dir is None else round(float(arrow_dir[0]),2),
            "arrow_y":"" if arrow_dir is None else round(float(arrow_dir[1]),2),
            "corners_cx":"" if not fixed else round(float(corners_est.reshape(-1,2)[:,0].mean()),1),
            "corners_cy":"" if not fixed else round(float(corners_est.reshape(-1,2)[:,1].mean()),1),
            "skipped":skipped_now
        })

        # snapshot whenever the overlay appears or disappears
        shown = state in ("track","detect","coast")
        if last_state is not None and shown != (last_state in ("track","detect","coast")) and snapshots < MAX_SNAPSHOTS:
            cv2.putText(frame,f"#{frame_id} {last_state}->{state}",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imwrite(os.path.join(run_dir,"snapshots",f"{frame_id:06d}_{state}.jpg"),frame)
            snapshots+=1
        last_state=state

        if not recent:
            window_start=time.time()

        recent.append((skipped_now,state))

        if len(recent)>=30:
            states=[s for _,s in recent]
            share=lambda *names: 100*sum(s in names for s in states)//len(states)
            # frames actually processed per second (averaging per-frame 1/dt overstates it),
            # and how many camera frames per second arrived but were skipped
            elapsed=max(time.time()-window_start,1e-6)
            print(f"[{time.time()-run_start:6.1f}s] fps {len(recent)/elapsed:4.0f} (+{sum(k for k,_ in recent)/elapsed:3.0f} skipped) | "
                  f"overlay {share('track','detect','coast'):3d}% (coast {share('coast'):3d}%) | "
                  f"offscreen {share('offscreen'):3d}% | lost {share('lost'):3d}%",flush=True)
            recent=[]
            log_file.flush()

    cv2.imshow("AR Output",frame)

    key=cv2.waitKey(1)&0xFF

    # 'c' starts a calibration sweep (once the target is tracked) or finishes it
    if key==ord('c'):
        if calibrating:
            finish_calibration()
        else:
            calibrate_requested = not calibrate_requested

    # 'q' in the video window, or closing the window
    if key==ord('q') or cv2.getWindowProperty("AR Output",cv2.WND_PROP_VISIBLE)<1:
        break

if grabber is not None:
    grabber.release()
else:
    cap.release()
cv2.destroyAllWindows()

if calibrating:
    finish_calibration()

if detector is not None and detect_stats["completed"]:
    print(f"Background detection: {detect_stats['completed']} finished, {detect_stats['found']} found the target, "
          f"{detect_stats['forwarded']} carried to the current frame "
          f"(median age {np.median(detect_stats['age']) if detect_stats['age'] else 0:.0f} frames)")

close_run()
