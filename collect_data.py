"""Collect hand-landmark training samples from the webcam.

Keys:  1 open palm | 2 closed fist | 3 thumbs up | 4 peace (optional extra)
       0 pause | Z discard unsaved samples | Q save and quit
Samples are written to data/gestures.csv whenever you pause or quit.
"""
import argparse
import csv
import os
import time
from collections import Counter

import cv2
import mediapipe as mp

from config import DATA_CSV, GESTURES, SAMPLES_TARGET
from features import csv_header, extract

KEY_TO_NAME = {ord(v[0]): name for name, v in GESTURES.items()}


def existing_counts():
    if not DATA_CSV.exists():
        return Counter()
    with open(DATA_CSV, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return Counter(row[0] for row in reader if row)


def flush(buffer):
    if not buffer:
        return 0
    DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DATA_CSV.exists() or DATA_CSV.stat().st_size == 0
    with open(DATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(csv_header())
        writer.writerows(buffer)
    n = len(buffer)
    buffer.clear()
    return n


def put(frame, text, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--rate", type=float, default=15.0, help="max samples per second")
    args = ap.parse_args()

    saved = existing_counts()
    session = Counter()
    buffer = []
    recording = None
    last_t = 0.0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam could not be opened. Try --camera 1 or close other apps using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    drawing = mp.solutions.drawing_utils
    hands_mod = mp.solutions.hands
    print("Press 1/2/3 (4 optional) to record, 0 to pause, Z to discard unsaved, Q to quit.")

    with hands_mod.Hands(max_num_hands=1, min_detection_confidence=0.7,
                         min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ext = extract(result)

            if ext is not None:
                feats, hand = ext
                drawing.draw_landmarks(frame, hand, hands_mod.HAND_CONNECTIONS)
                now = time.time()
                if recording and now - last_t >= 1.0 / args.rate:
                    buffer.append([recording] + [f"{v:.6f}" for v in feats])
                    session[recording] += 1
                    last_t = now

            status = f"RECORDING: {GESTURES[recording][1]}" if recording else "PAUSED"
            put(frame, status, (15, 30), (0, 0, 255) if recording else (0, 200, 255), 0.8)
            if ext is None:
                put(frame, "No hand detected", (15, 58), (0, 165, 255))
            y = 95
            for name, (key, disp, _, _) in GESTURES.items():
                total = saved[name] + session[name]
                color = (0, 220, 0) if total >= SAMPLES_TARGET else (255, 255, 255)
                put(frame, f"[{key}] {disp}: {total}/{SAMPLES_TARGET}", (15, y), color, 0.55)
                y += 26
            put(frame, "0 pause | Z discard unsaved | Q save+quit", (15, frame.shape[0] - 15), scale=0.5)

            cv2.imshow("GestureSlide - collect data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("0"):
                recording = None
                n = flush(buffer)
                if n:
                    print(f"Saved {n} samples.")
            elif key == ord("z"):
                for row in buffer:
                    session[row[0]] -= 1
                buffer.clear()
                print("Discarded unsaved samples.")
            elif key in KEY_TO_NAME:
                if recording:
                    flush(buffer)
                recording = KEY_TO_NAME[key]

    n = flush(buffer)
    if n:
        print(f"Saved {n} samples.")
    cap.release()
    cv2.destroyAllWindows()
    totals = existing_counts()
    print("Dataset totals:", dict(totals))
    if totals and max(totals.values()) > 1.5 * min(totals.values()):
        print("Warning: classes are unbalanced - collect more of the smaller class.")


if __name__ == "__main__":
    main()
