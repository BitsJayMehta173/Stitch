import cv2
import numpy as np
import time

# ================================
# CONFIGURATION
# ================================

# ================================
# ================================

# CHANGE THE URL FIRST  
# Use the Camera stream URL you see on your phone in DroidCam for Android Install It First In Your Device

# ================================
# ================================
URL = "http://X.X.X.X:Y/video"


# Change the reference.jpg image to the object on which you want to overlay
REFERENCE_IMAGE = "reference.jpg"
# Change the Overlay.png image to the ScreenShot you want to show over the reference image 
OVERLAY_IMAGE = "overlay.png"

# ================================
# AR INITIALIZATION (FROM YOUR NOTEBOOK)
# ================================
reference = cv2.imread(REFERENCE_IMAGE, 0)
overlay = cv2.imread(OVERLAY_IMAGE)

# SIFT Setup
sift = cv2.SIFT_create(nfeatures=1500)
kp_ref, des_ref = sift.detectAndCompute(reference, None)
h_ref, w_ref = reference.shape
overlay = cv2.resize(overlay, (w_ref, h_ref))

# FLANN Matcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Memory & CLAHE
prev_H = None
reuse_count = 0
max_reuse = 5
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# ================================
# STREAM ACCESS
# ================================
print(f"Connecting to stream at {URL}...")
cap = cv2.VideoCapture(URL)

# Optimization: Reduce internal buffer to ensure we don't see "old" frames
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: Could not open the video stream. Check if DroidCam is running.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped frame, retrying...")
        continue

    # 1. Image Enhancement
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = clahe.apply(gray)

    # 2. SIFT Processing
    kp_frame, des_frame = sift.detectAndCompute(gray, None)
    H = None

    if des_frame is not None:
        matches = flann.knnMatch(des_ref, des_frame, k=2)
        # Lowes Ratio Test
        good = [m for m, n in matches if m.distance < 0.65 * n.distance]

        if len(good) > 12:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            
            H_cand, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H_cand is not None and mask is not None:
                if mask.ravel().sum() > 20: # Inlier check
                    H = H_cand
                    prev_H = H_cand
                    reuse_count = 0

    # 3. Homography Persistence
    if H is None and prev_H is not None and reuse_count < max_reuse:
        H = prev_H
        reuse_count += 1

    # 4. Warping and Rendering
    if H is not None:
        try:
            warped = cv2.warpPerspective(overlay, H, (frame.shape[1], frame.shape[0]))
            
            # Create transparency mask
            gray_overlay = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, o_mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(o_mask)
            
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            overlay_fg = cv2.bitwise_and(warped, warped, mask=o_mask)
            frame = cv2.add(frame_bg, overlay_fg)
        except Exception as e:
            print(f"Warp error: {e}")

    # 5. Application Output Window
    cv2.imshow("Real-Time AR Output", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()