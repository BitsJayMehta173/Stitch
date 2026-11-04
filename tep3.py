import atexit
import cv2
import csv
import numpy as np
import os
import sys
import time

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
if len(sys.argv) > 1:
    URL = sys.argv[1]

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

# ---- stream ----
STREAM_TIMEOUT = 3.0        # seconds without frames before reconnecting to the stream
STREAM_RECONNECTS = 3       # give up after this many failed reconnects

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

def detect(gray,roi=None):

    global detect_calls,detect_ms

    t0=time.perf_counter()
    detect_calls+=1

    x0,y0 = 0,0

    if roi is not None:
        x0,y0,w,h = roi
        gray = gray[y0:y0+h,x0:x0+w]

    result=None

    for scale in SIFT_SCALES:

        found = detect_at_scale(gray,scale)

        if found is not None:
            ref_pts,frame_pts = found
            frame_pts += np.float32([x0,y0])
            result = ref_pts,frame_pts
            break

    detect_ms+=1000*(time.perf_counter()-t0)

    return result


def detect_at_scale(gray,scale):

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


# arrow at the frame edge pointing to where the off-screen target is

def draw_offscreen_arrow(frame,quad):

    h,w=frame.shape[:2]
    centre=np.float32([w/2,h/2])
    d=quad.reshape(-1,2).mean(axis=0)-centre
    n=np.linalg.norm(d)

    if n<1:
        return

    d/=n
    tip=centre+d*min(w,h)*0.45
    tail=centre+d*min(w,h)*0.30

    cv2.arrowedLine(frame,tuple(int(v) for v in tail),tuple(int(v) for v in tip),
                    (0,200,255),4,tipLength=0.4)


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


cap=open_source()

if not cap.isOpened():
    print(f"Could not open {URL}. Check the DroidCam URL or the video path.")
    exit()

# frames are resized to FRAME_RESIZE; a different aspect ratio here means they get stretched
print(f"Source resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

prev_time=time.time()

# =====================================
# RUN LOG
# =====================================

LOG_FIELDS=["frame","t","fps","proc_ms","state","target_pts","motion_ok","motion_pts",
            "tx","ty","rot_deg","detect_calls","detect_ms","lost_frames","visible","stale"]

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
        # MJPG/AVI stays readable even if the run is killed before the file is closed
        input_writer=cv2.VideoWriter(os.path.join(run_dir,"input.avi"),
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

last_frame_time=time.time()
reconnects=0

# =====================================
# MAIN LOOP
# =====================================

while True:

    ret,frame=cap.read()

    if not ret:
        # a video file has ended; a live stream just dropped a frame
        if IS_FILE or stop_requested():
            break

        # or the stream died (DroidCam closed, Wi-Fi dropped): reconnect, then give up
        if time.time()-last_frame_time > STREAM_TIMEOUT:
            if reconnects >= STREAM_RECONNECTS:
                print("Stream lost, giving up.")
                break
            reconnects+=1
            print(f"No frames for {STREAM_TIMEOUT:.0f}s, reconnecting ({reconnects}/{STREAM_RECONNECTS})...")
            cap.release()
            cap=open_source()
            last_frame_time=time.time()
        continue

    last_frame_time=time.time()
    reconnects=0

    frame=cv2.resize(frame,FRAME_RESIZE)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if input_writer is not None:
        input_writer.write(frame)

    frame_start=time.perf_counter()
    detect_calls,detect_ms=0,0.0

    frame_id+=1
    H=None
    fixed=False
    state="lost"

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

    found=None
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

        if anchor_due and not blurry:

            frames_since_anchor = 0

            # only search around where the target is now: cheaper, and a small
            # far-away target fills more of the search window
            roi = roi_around(corners_est,gray.shape) if corners_est is not None else None

            found = detect(gray,roi)

        if found is None and len(good_new)<MIN_TRACKED_POINTS:

            prev_pts=None
            good_new=None

    # ---------------------------------
    # SEARCH (target not tracked)
    # ---------------------------------

    if prev_pts is None and found is None:

        lost_frames += 1

        if corners_est is not None and lost_frames <= PREDICT_MAX_FRAMES:

            should_be_visible = visible_fraction(corners_est,gray.shape) > 0.2

            # SIFT on a smeared frame almost never succeeds and costs ~100 ms; and once a
            # loss drags on, searching every few frames is enough
            search_now = not blurry and (lost_frames <= LOST_FAST_FRAMES or lost_frames % LOST_DETECT_EVERY == 0)

            if search_now and (est_stale or search_misses >= 3):

                # the prediction is unreliable: look everywhere
                found = detect(gray)

            elif search_now and should_be_visible:

                # camera motion tells us roughly where the target is: search only there,
                # and not at all while it is out of view
                roi = roi_around(corners_est,gray.shape)

                if roi is not None:
                    found = detect(gray,roi)

            if search_now and found is None and (est_stale or should_be_visible):
                search_misses += 1

        else:

            # no idea where it is: full-frame search with cooldown
            corners_est = None

            frames_since_detect += 1

            if frames_since_detect >= detect_interval:

                frames_since_detect = 0

                found = detect(gray)

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

        if area>MIN_OVERLAY_AREA:

            size=(frame.shape[1],frame.shape[0])

            warped=cv2.warpPerspective(overlay,H,size)
            mask=cv2.warpPerspective(overlay_mask,H,size)
            mask_inv=cv2.bitwise_not(mask)

            frame_bg=cv2.bitwise_and(frame,frame,mask=mask_inv)
            overlay_fg=cv2.bitwise_and(warped,warped,mask=mask)

            frame=cv2.add(frame_bg,overlay_fg)

    elif corners_est is not None and not est_stale and visible < 0.05:

        state="offscreen"

        draw_offscreen_arrow(frame,corners_est)

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
            "stale":int(est_stale)
        })

        # snapshot whenever the overlay appears or disappears
        shown = state in ("track","detect","coast")
        if last_state is not None and shown != (last_state in ("track","detect","coast")) and snapshots < MAX_SNAPSHOTS:
            cv2.putText(frame,f"#{frame_id} {last_state}->{state}",(20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.imwrite(os.path.join(run_dir,"snapshots",f"{frame_id:06d}_{state}.jpg"),frame)
            snapshots+=1
        last_state=state

        recent.append((fps,state))

        if len(recent)>=30:
            states=[s for _,s in recent]
            share=lambda *names: 100*sum(s in names for s in states)//len(states)
            print(f"[{time.time()-run_start:6.1f}s] fps {np.mean([f for f,_ in recent]):4.0f} | "
                  f"overlay {share('track','detect','coast'):3d}% (coast {share('coast'):3d}%) | "
                  f"offscreen {share('offscreen'):3d}% | lost {share('lost'):3d}%",flush=True)
            recent=[]
            log_file.flush()

    cv2.imshow("AR Output",frame)

    # 'q' in the video window, or closing the window
    if stop_requested() or cv2.getWindowProperty("AR Output",cv2.WND_PROP_VISIBLE)<1:
        break

cap.release()
cv2.destroyAllWindows()
close_run()
