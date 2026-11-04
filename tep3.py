import cv2
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



# So the Problem right now is the images which has low keypoints and descriptors have slow processing power and recognition power

# Change the reference.jpg image to the object on which you want to overlay
REFERENCE_IMAGE = "reference.jpg"
# Change the Overlay.png image to the ScreenShot you want to show over the reference image 
OVERLAY_IMAGE = "overlay.png"
FRAME_RESIZE = (960,540)

MIN_TRACKED_POINTS = 20
FB_CHECK_INTERVAL = 5
SMOOTH_ALPHA = 0.85

# SIFT detection must agree with at least this many RANSAC inliers
MIN_DETECT_INLIERS = 20

# re-run SIFT every N frames while tracking to stop optical flow drift
REDETECT_INTERVAL = 15

# Downscale factor for SIFT detection (e.g., 0.5 means half resolution)
SIFT_SCALE = 0.5

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

kp_ref,des_ref = sift.detectAndCompute(reference,None)

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

prev_pts=None
ref_pts=None
prev_gray=None
prev_corners=None

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

def detect(gray):

    # 1. Downscale the frame to drastically reduce SIFT processing time
    small_gray = cv2.resize(gray, (0, 0), fx=SIFT_SCALE, fy=SIFT_SCALE)
    
    # 2. Run SIFT on the smaller image
    kp_frame,des_frame=sift.detectAndCompute(small_gray,None)

    if des_frame is None or len(des_frame)<2:
        return None

    matches=flann.knnMatch(des_ref,des_frame,k=2)

    good=[]

    for pair in matches:
        if len(pair)==2 and pair[0].distance<0.7*pair[1].distance:
            good.append(pair[0])

    if len(good)<30:
        return None

    ref_pts=np.float32(
        [kp_ref[m.queryIdx].pt for m in good]
    ).reshape(-1,1,2)

    # 3. Rescale the points back to original resolution so optical flow and homography work properly
    frame_pts=np.float32(
        [kp_frame[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    frame_pts /= SIFT_SCALE

    # 4. Ratio test alone lets false positives through; require a consistent homography
    H,mask=cv2.findHomography(ref_pts,frame_pts,cv2.RANSAC,5.0)

    if not valid_homography(H) or mask is None:
        return None

    mask=mask.ravel().astype(bool)

    if mask.sum()<MIN_DETECT_INLIERS:
        return None

    return ref_pts[mask],frame_pts[mask]


# =====================================
# STREAM
# =====================================

def stop_requested():
    return cv2.waitKey(1)&0xFF==ord('q')


cap=cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

if not cap.isOpened():
    print(f"Could not open {URL}. Check the DroidCam URL or the video path.")
    exit()

prev_time=time.time()

# =====================================
# MAIN LOOP
# =====================================

while True:

    ret,frame=cap.read()

    if not ret:
        # a video file has ended; a live stream just dropped a frame
        if IS_FILE or stop_requested():
            break
        continue

    frame=cv2.resize(frame,FRAME_RESIZE)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    frame_id+=1
    H=None

    # ---------------------------------
    # DETECTION MODE (no object tracked)
    # ---------------------------------

    if prev_pts is None:

        frames_since_detect += 1

        found = None

        if frames_since_detect >= detect_interval:

            frames_since_detect = 0

            found = detect(gray)

            if found is not None:
                detect_interval = DETECT_BASE_INTERVAL
            else:
                detect_interval = min(detect_interval*2, DETECT_MAX_INTERVAL)

        if found is None:

            cv2.imshow("AR Output",frame)
            if stop_requested():
                break
            continue

        # render on the frame the target was found in instead of skipping it
        good_ref,good_new = found
        good_ref = good_ref.reshape(-1,2)
        good_new = good_new.reshape(-1,2)
        good_prev = good_new

    else:

        # ---------------------------------
        # OPTICAL FLOW TRACKING
        # ---------------------------------

        next_pts,status,_=cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            prev_pts,
            None,
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
                None,
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

        if frame_id % REDETECT_INTERVAL == 0:

            found = detect(gray)

            if found is not None:
                good_ref,good_new = found
                good_ref = good_ref.reshape(-1,2)
                good_new = good_new.reshape(-1,2)
                good_prev = None

        if len(good_new)<MIN_TRACKED_POINTS:

            prev_pts=None
            prev_corners=None

            cv2.imshow("AR Output",frame)
            if stop_requested():
                break
            continue

    # ---------------------------------
    # HOMOGRAPHY
    # ---------------------------------

    H_candidate,inliers=cv2.findHomography(
        good_ref.reshape(-1,1,2),
        good_new.reshape(-1,1,2),
        cv2.RANSAC,
        5.0
    )

    if valid_homography(H_candidate) and inliers is not None:

        # carry only points that agree with this pose into the next frame
        keep=inliers.ravel().astype(bool)
        good_new=good_new[keep]
        good_ref=good_ref[keep]
        if good_prev is not None:
            good_prev=good_prev[keep]

        corners=cv2.perspectiveTransform(ref_corners,H_candidate)

        # Smooth the projected corners rather than the matrix itself:
        # averaging two homographies element-wise is not a valid blend of poses.
        if prev_corners is None:
            alpha = 0.0
        elif good_prev is None:
            # just re-anchored by SIFT: ease towards the corrected pose
            alpha = SMOOTH_ALPHA
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

        if prev_corners is not None:
            corners = alpha*prev_corners + (1-alpha)*corners

        prev_corners=corners

        H=cv2.getPerspectiveTransform(
            ref_corners.reshape(4,2),
            corners.reshape(4,2).astype(np.float32)
        )

    prev_pts=good_new.reshape(-1,1,2).astype(np.float32)
    ref_pts=good_ref.reshape(-1,1,2).astype(np.float32)

    prev_gray=gray.copy()

    # ---------------------------------
    # OVERLAY
    # ---------------------------------

    if H is not None:

        area=cv2.contourArea(prev_corners.astype(np.float32))

        if area>2000:

            size=(frame.shape[1],frame.shape[0])

            warped=cv2.warpPerspective(overlay,H,size)
            mask=cv2.warpPerspective(overlay_mask,H,size)
            mask_inv=cv2.bitwise_not(mask)

            frame_bg=cv2.bitwise_and(frame,frame,mask=mask_inv)
            overlay_fg=cv2.bitwise_and(warped,warped,mask=mask)

            frame=cv2.add(frame_bg,overlay_fg)

    # ---------------------------------
    # FPS
    # ---------------------------------

    new_time=time.time()
    fps=1/max(new_time-prev_time,1e-6)
    prev_time=new_time

    cv2.putText(frame,f"FPS:{int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("AR Output",frame)

    if stop_requested():
        break

cap.release()
cv2.destroyAllWindows()
