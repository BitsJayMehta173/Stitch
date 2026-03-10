# Our Main Motive is Digital Garbage Digital AR/VR Social Media 
# People Can Visit the digital Garbage People Throws at a Perfectly Fine Place In the World Which Will be REcognized By many Different Means Like Location, GPS, Gyroscope Map, GTA MAP TYPE SHIT WORLD BRODAA,,,,Suppose EveryOne Will Be Allowed to throw A Single Cocacola Can On A specific Place Once the Application Is Launched Just In the Rememberance of the App Launch You Might be able to see people throwing cans digitally, these are just metaphor for people thats my negative thinking people have to think positive themselves I dont know what you might use it for,, but the future is different for sure but it starts with thissss.


import cv2
import numpy as np

# -------------------------------
# Load reference and overlay
# -------------------------------

reference = cv2.imread("reference.jpg", 0)

overlay = cv2.imread("overlay.png")

sift = cv2.SIFT_create(nfeatures=1500)

kp_ref, des_ref = sift.detectAndCompute(reference, None)

h_ref, w_ref = reference.shape

overlay = cv2.resize(overlay, (w_ref, h_ref))


# -------------------------------
# Feature matcher
# -------------------------------

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


# -------------------------------
# Video input / output
# -------------------------------

cap = cv2.VideoCapture("video.mp4")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "output.mp4",
    fourcc,
    8,                 # reduced fps
    (width, height)
)


# -------------------------------
# Homography reuse system
# -------------------------------

prev_H = None
reuse_count = 0
max_reuse = 5


# -------------------------------
# Preprocessing setup
# -------------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


# -------------------------------
# Process video frames
# -------------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # preprocessing filters
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray = clahe.apply(gray)

    kp_frame, des_frame = sift.detectAndCompute(gray, None)

    H = None

    if des_frame is not None:

        matches = flann.knnMatch(des_ref, des_frame, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)

        if len(good) > 12:

            src_pts = np.float32(
                [kp_ref[m.queryIdx].pt for m in good]
            ).reshape(-1,1,2)

            dst_pts = np.float32(
                [kp_frame[m.trainIdx].pt for m in good]
            ).reshape(-1,1,2)

            H_candidate, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                5.0
            )

            if H_candidate is not None and mask is not None:

                inliers = mask.ravel().sum()

                if inliers > 20:

                    H = H_candidate
                    prev_H = H_candidate
                    reuse_count = 0


    # reuse previous homography if detection failed
    if H is None and prev_H is not None and reuse_count < max_reuse:

        H = prev_H
        reuse_count += 1


    # apply overlay
    if H is not None:

        warped = cv2.warpPerspective(
            overlay,
            H,
            (frame.shape[1], frame.shape[0])
        )

        gray_overlay = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        _, overlay_mask = cv2.threshold(
            gray_overlay,
            1,
            255,
            cv2.THRESH_BINARY
        )

        mask_inv = cv2.bitwise_not(overlay_mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

        overlay_fg = cv2.bitwise_and(warped, warped, mask=overlay_mask)

        frame = cv2.add(frame_bg, overlay_fg)


    out.write(frame)


# -------------------------------
# Release resources
# -------------------------------

cap.release()
out.release()

print("Processing complete. Output saved as output.mp4")