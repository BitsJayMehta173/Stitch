import cv2
import numpy as np
import time

# =====================================
# CONFIGURATION
# =====================================



# CHANGE THE URL FIRST  
# Use the Camera stream URL you see on your phone in DroidCam for Android Install It First In Your Device
# after you install DroidCam you will get Wifi Ip if your pc and mobile is both connected in same wifi you can copy the URL and keep it here this way you can get the live feed but below you also have to change the refence image and overlay image

# ================================
# ================================
URL = "http://X.X.X.X:Y/video"



# So the Problem right now is the images which has low keypoints and descriptors have slow processing power and recognition power

# Change the reference.jpg image to the object on which you want to overlay
REFERENCE_IMAGE = "reference.jpg"
# Change the Overlay.png image to the ScreenShot you want to show over the reference image 
OVERLAY_IMAGE = "overlay.png"
FRAME_RESIZE = (960,540)

MIN_TRACKED_POINTS = 20
FB_CHECK_INTERVAL = 5
SMOOTH_ALPHA = 0.85

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
prev_H=None

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

def geometric_filter(prev_pts,new_pts,ref_pts):

    motion = new_pts - prev_pts
    median_motion = np.median(motion,axis=0)

    diff = np.linalg.norm(motion - median_motion,axis=1)

    mask = diff < 5.0

    return new_pts[mask], ref_pts[mask]


# =====================================
# DETECTION FUNCTION
# =====================================

def detect(gray):

    global prev_pts,ref_pts

    kp_frame,des_frame=sift.detectAndCompute(gray,None)

    if des_frame is None:
        return False

    matches=flann.knnMatch(des_ref,des_frame,k=2)

    good=[]

    for m,n in matches:
        if m.distance<0.7*n.distance:
            good.append(m)

    if len(good)<30:
        return False

    ref_pts=np.float32(
        [kp_ref[m.queryIdx].pt for m in good]
    ).reshape(-1,1,2)

    prev_pts=np.float32(
        [kp_frame[m.trainIdx].pt for m in good]
    ).reshape(-1,1,2)

    return True


# =====================================
# STREAM
# =====================================

cap=cv2.VideoCapture(URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE,1)

prev_time=time.time()

# =====================================
# MAIN LOOP
# =====================================

while True:

    ret,frame=cap.read()

    if not ret:
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

        if frames_since_detect < detect_interval:

            cv2.imshow("AR Output",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            continue

        frames_since_detect = 0

        success = detect(gray)

        if success:

            detect_interval = DETECT_BASE_INTERVAL
            prev_gray = gray.copy()
            continue

        else:

            detect_interval = min(detect_interval*2, DETECT_MAX_INTERVAL)

            cv2.imshow("AR Output",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            continue


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

    good_new,good_ref = geometric_filter(good_prev,good_new,good_ref)

    if len(good_new)<MIN_TRACKED_POINTS:

        prev_pts=None
        continue

    # ---------------------------------
    # HOMOGRAPHY
    # ---------------------------------

    H_candidate,_=cv2.findHomography(
        good_ref.reshape(-1,1,2),
        good_new.reshape(-1,1,2),
        cv2.RANSAC,
        5.0
    )

    if valid_homography(H_candidate):

        if prev_H is None:
            H=H_candidate
        else:
            H=SMOOTH_ALPHA*prev_H+(1-SMOOTH_ALPHA)*H_candidate

        prev_H=H

    prev_pts=good_new.reshape(-1,1,2)
    ref_pts=good_ref.reshape(-1,1,2)

    prev_gray=gray.copy()

    # ---------------------------------
    # OVERLAY
    # ---------------------------------

    if H is not None:

        corners=np.float32([
            [0,0],
            [w_ref,0],
            [w_ref,h_ref],
            [0,h_ref]
        ]).reshape(-1,1,2)

        proj=cv2.perspectiveTransform(corners,H)

        area=cv2.contourArea(proj)

        if area>2000:

            warped=cv2.warpPerspective(
                overlay,
                H,
                (frame.shape[1],frame.shape[0])
            )

            mask=(warped.sum(axis=2)>0).astype(np.uint8)*255
            mask_inv=cv2.bitwise_not(mask)

            frame_bg=cv2.bitwise_and(frame,frame,mask=mask_inv)
            overlay_fg=cv2.bitwise_and(warped,warped,mask=mask)

            frame=cv2.add(frame_bg,overlay_fg)

    # ---------------------------------
    # FPS
    # ---------------------------------

    new_time=time.time()
    fps=1/(new_time-prev_time)
    prev_time=new_time

    cv2.putText(frame,f"FPS:{int(fps)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("AR Output",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()