"""Background threads for tep3.py, so the tracking loop never waits on slow work.

OpenCV releases the GIL while decoding, encoding and running SIFT, so these threads run
truly in parallel with the main loop.
"""
import queue
import threading
import time

import cv2


class FrameGrabber:
    """Reads a live stream on its own thread and keeps only the newest frame.

    Decoding a 1280x720 DroidCam frame takes ~12 ms; doing it here takes it off the main
    loop, and always handing out the newest frame keeps latency low when the loop is slow."""

    def __init__(self, open_source, size):
        self.open_source = open_source
        self.size = size
        self.cond = threading.Condition()
        self.frame = None
        self.seq = 0
        self.last_frame_time = time.time()
        self._start()

    def _start(self):
        self.cap = self.open_source()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def opened(self):
        return self.cap.isOpened()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            frame = cv2.resize(frame, self.size)
            with self.cond:
                self.frame = frame
                self.seq += 1
                self.last_frame_time = time.time()
                self.cond.notify_all()

    def read(self, last_seq, timeout=0.2):
        """Newest frame newer than last_seq: (frame, seq), or (None, last_seq) on timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq > last_seq, timeout)
            if self.seq > last_seq:
                return self.frame, self.seq
        return None, last_seq

    def reconnect(self):
        self.release()
        self._start()
        self.last_frame_time = time.time()

    def release(self):
        self.running = False
        self.thread.join(timeout=5)
        self.cap.release()


class AsyncWriter:
    """cv2.VideoWriter on a background thread (~9 ms per frame off the main loop).

    Frames are dropped, not queued forever, if encoding falls behind."""

    def __init__(self, path, fourcc, fps, size, max_queue=90):
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        self.queue = queue.Queue(max_queue)
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def write(self, frame):
        try:
            # the caller keeps drawing on its frame, so store a copy
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            self.writer.write(frame)

    def release(self):
        self.queue.put(None)
        self.thread.join()
        self.writer.release()


class DetectWorker:
    """Runs one detection at a time on a background thread.

    detect(gray, roi, request) runs on the worker. submit() hands over a frame; take()
    returns (request, result) once it has finished.
    The request is the caller's own object, so it can carry what is needed to move the
    result from the frame it was computed on to the current frame."""

    def __init__(self, detect):
        self.detect = detect
        self.cond = threading.Condition()
        self.pending = None
        self.finished = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def busy(self):
        with self.cond:
            return self.pending is not None

    def submit(self, request, gray, roi):
        with self.cond:
            if self.pending is not None:
                return False
            self.pending = (request, gray, roi)
            self.cond.notify_all()
            return True

    def take(self):
        with self.cond:
            done, self.finished = self.finished, None
            return done

    def in_flight(self):
        """The request being worked on or waiting to be taken, if any."""
        with self.cond:
            if self.pending is not None:
                return self.pending[0]
            return None if self.finished is None else self.finished[0]

    def _run(self):
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.pending is not None)
                request, gray, roi = self.pending
            result = self.detect(gray, roi, request)
            with self.cond:
                self.finished = (request, result)
                self.pending = None
