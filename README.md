# Stitch Mobile AR/VR

### Our Main Motive is Digital Garbage Digital AR/VR Social Media 

### AR/VR Customized Virtual Social Media Where Your Surroundings will Be the Wall Feed Which was What Facebook Visualized As. You Can See the Live Feeds People Put Up In The Wall, Buildings, Trees, Nature and Anywhere You Like... One Common Practice We will follow is We will keep a Can Point Where we will throw Virtual Cocacola Can and Pile Up Just in the remeberance of the Launch(SooooN)


![Project Walkthrough](./giff.gif)


# Updates

#### Run tep2.py for now
#### where you first need to install DroidCam on your mobile device and Your PC 
#### Then use the IP address of the Mobile camera to access the live feed 
#### Then you can change the overlay.png and reference.jpg to your choice to experience the mobile AR/VR of your own.

# Overview

This project demonstrates a computer vision–based augmented reality screen projection system. Using feature detection and homography estimation, the application can project a digital screen (such as a laptop display screenshot) onto a real-world surface captured by a camera.

The system detects a predefined reference surface using SIFT feature detection and feature matching, estimates the surface orientation through homography transformation, and overlays a digital screen onto that surface in real time or across video frames.

The goal is to simulate the experience of AR/VR floating screens using only computer vision techniques.

Concept

Modern AR and VR systems allow digital screens to appear fixed in physical space. This project recreates a simplified version of that behavior using classical computer vision.

The pipeline works as follows:

A reference image (for example, a book, table, or card) is used as the target surface.

The system detects distinctive visual features on that surface using SIFT.

In each camera frame, those features are matched with the reference image.

A homography matrix is computed to determine the perspective transformation between the reference surface and the camera frame.

A digital screen image (e.g., a laptop screenshot) is warped and projected onto the detected surface.

The processed frames are reconstructed into a video stream.

This makes the digital screen appear attached to the physical surface, even as the camera moves.

Intended Application

The long-term goal of this project is to enable lightweight screen sharing between devices using augmented reality concepts.

A practical example:

A laptop shares its screen as an image stream.

A mobile phone camera points at a physical surface such as a notebook or table.

The system detects that surface and projects the laptop screen onto it, making it appear like a floating AR display.

This creates an experience similar to AR/VR spatial screens, where digital content appears anchored to real-world objects.

Potential use cases include:

AR-style remote screen viewing

Portable secondary displays

Collaborative workspace visualization

Educational AR demonstrations

Prototyping spatial computing interfaces

Key Technologies

The system is built using classical computer vision techniques:

SIFT (Scale-Invariant Feature Transform) for feature detection

FLANN-based feature matching

RANSAC-based homography estimation

Perspective warping for screen projection

Frame-by-frame video reconstruction

These components allow the application to maintain spatial consistency of the projected screen across camera motion.

Future Improvements

Possible future extensions include:

real-time camera processing

feature tracking instead of frame-by-frame detection

multi-surface projection

improved AR stability

integration with mobile camera streams

wireless screen streaming from laptop to phone
