"""GestureSlide - real-time gesture-controlled presentation.

  python gestureslide.py              preview only (no keys sent)
  python gestureslide.py --control    send keys to the active presentation

Window keys: Q quit | C toggle control | P toggle guarded hand pointer | L landmarks
"""
import argparse
import csv
import json
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

from config import (DEFAULT_COOLDOWN, DEFAULT_STABLE, DEFAULT_THRESHOLD, GESTURES,
                    GUARD_KEYWORDS, LABELS_PATH, LOG_DIR, MODEL_PATH)
from features import extract

ACTIONS = {name: (key, desc) for name, (_, _, key, desc) in GESTURES.items()}
DISPLAY = {name: disp for name, (_, disp, _, _) in GESTURES.items()}
SWIPE_ACTIONS = {
    "swipe_right": ("right", "Swipe right - next slide"),
    "swipe_left": ("left", "Swipe left - previous slide"),
}


def rule_predict(hand):
    """Recognize the four basic poses from MediaPipe landmarks without a trained model.

    This fallback makes a fresh checkout usable for a live preview. It is a
    simple geometric classifier; collect/train data for the learned model
    when more robust recognition is needed.
    """
    p = hand.landmark
    extended = [p[tip].y < p[pip].y - 0.015
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
    fingers_up = sum(extended)
    thumb_up = p[4].y < p[0].y - 0.12

    gesture = None
    if all(extended):
        gesture = "open_palm"
    elif extended[0] and extended[1] and not extended[2] and not extended[3]:
        gesture = "peace"
    elif fingers_up == 0 and thumb_up:
        gesture = "thumbs_up"
    elif fingers_up == 0:
        gesture = "fist"

    classes = list(GESTURES)
    probs = np.full(len(classes), 0.0, dtype=np.float32)
    if gesture is not None:
        probs[classes.index(gesture)] = 1.0
    return probs


def put(frame, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def bar(frame, x, y, w, h, frac, color):
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * frac), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)


def window_allowed(keywords):
    """Window guard: only allow key presses when a presentation window is active."""
    try:
        import pygetwindow as gw
        win = gw.getActiveWindow()
        if win is None:
            return False, ""
        title = win.title or ""
        return any(k in title.lower() for k in keywords), title
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)  # fail closed when the active window cannot be checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", action="store_true", help="send keyboard commands")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--stable", type=int, default=DEFAULT_STABLE, help="frames that must agree")
    ap.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    ap.add_argument("--repeat", action="store_true",
                    help="allow the same gesture to repeat while held (default: must re-arm)")
    ap.add_argument("--no-guard", action="store_true", help="disable active-window safety guard")
    args = ap.parse_args()

    rule_mode = not MODEL_PATH.exists() or not LABELS_PATH.exists()
    if rule_mode:
        model = None
        classes = list(GESTURES)
        print("No trained model found; using live geometric gesture rules.")
    else:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        classes = json.loads(LABELS_PATH.read_text())["classes"]

    pag = None
    control = args.control
    if control:
        try:
            import pyautogui
            pyautogui.PAUSE = 0
            pag = pyautogui
        except Exception as e:  # noqa: BLE001
            print("PyAutoGUI unavailable, running in preview mode:", e)
            control = False

    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.csv"
    log_f = open(log_path, "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["time", "gesture", "confidence", "key", "result"])

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam could not be opened. Try --camera 1 or close apps using the camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    hands_mod = mp.solutions.hands
    drawing = mp.solutions.drawing_utils

    history = deque(maxlen=args.stable)
    motion_history = deque()
    last_fire = 0.0
    last_swipe = 0.0
    last_fired = None
    armed = True
    last_seen = 0.0
    banner, banner_t = "", 0.0
    counts = Counter()
    detected_counts = Counter()
    confidence_totals = Counter()
    outcome_counts = Counter()
    slide = 0
    show_lm = True
    pointer_on = False
    pointer_pos = None
    pointer_frames = 0
    swipe_enabled = False
    start = time.time()
    session_started = datetime.now()
    fps, prev_t = 0.0, time.time()

    with hands_mod.Hands(max_num_hands=1, min_detection_confidence=0.7,
                         min_tracking_confidence=0.6) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-3))
            prev_t = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            ext = extract(result)
            probs = None
            name, conf = None, 0.0
            pointer_active = False
            swipe_event = None

            if ext is None:
                history.clear()
                motion_history.clear()
                if now - last_seen > 0.4:
                    armed = True  # hand left the view -> re-arm
            else:
                feats, hand = ext
                last_seen = now
                if show_lm:
                    drawing.draw_landmarks(frame, hand, hands_mod.HAND_CONNECTIONS)
                probs = (rule_predict(hand) if rule_mode else
                         model(feats[None, :], training=False).numpy()[0])
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
                if conf >= args.threshold and conf > 0:
                    name = classes[idx]
                    if name != last_fired:
                        armed = True  # a different confident gesture re-arms
                history.append(name)

                if pointer_on and control and pag is not None:
                    if args.no_guard:
                        allowed = True
                    else:
                        allowed, _ = window_allowed(GUARD_KEYWORDS)
                    if allowed:
                        pointer_active = True
                        screen_w, screen_h = pag.size()
                        tip = hand.landmark[8]
                        target = (int(max(0, min(screen_w - 1, tip.x * screen_w))),
                                  int(max(0, min(screen_h - 1, tip.y * screen_h))))
                        if pointer_pos is None:
                            pointer_pos = target
                        else:
                            pointer_pos = (int(0.7 * pointer_pos[0] + 0.3 * target[0]),
                                           int(0.7 * pointer_pos[1] + 0.3 * target[1]))
                        pag.moveTo(*pointer_pos, duration=0)
                        pointer_frames += 1

                motion_history.append((now, hand.landmark[0].x))
                while motion_history and now - motion_history[0][0] > 0.40:
                    motion_history.popleft()
                if (swipe_enabled and len(motion_history) >= 4
                        and now - last_swipe >= max(0.8, args.cooldown)
                        and now - last_fire >= args.cooldown):
                    delta_x = motion_history[-1][1] - motion_history[0][1]
                    if abs(delta_x) >= 0.22:
                        swipe_name = "swipe_right" if delta_x > 0 else "swipe_left"
                        swipe_key, swipe_desc = SWIPE_ACTIONS[swipe_name]
                        swipe_score = min(1.0, abs(delta_x) / 0.30)
                        swipe_event = (swipe_name, swipe_key, swipe_desc, swipe_score)
                        last_swipe = now
                        motion_history.clear()

            # ---- stable-prediction window ----
            run = 0
            if history and history[-1] is not None:
                for g in reversed(history):
                    if g == history[-1]:
                        run += 1
                    else:
                        break
            progress = run / args.stable
            stable = run >= args.stable
            cooling = (now - last_fire) < args.cooldown

            action_event = swipe_event
            pose_navigation_suppressed = (
                swipe_enabled and history and history[-1] in {"open_palm", "fist"}
            )
            if (action_event is None and stable and not cooling and (armed or args.repeat)
                    and not pose_navigation_suppressed):
                gesture = history[-1]
                key, desc = ACTIONS.get(gesture, (None, gesture))
                action_event = (gesture, key, desc, conf)

            if action_event is not None:
                gesture, key, desc, event_conf = action_event
                detected_counts[gesture] += 1
                confidence_totals[gesture] += event_conf
                result_txt = "preview"
                if key and control and pag is not None:
                    if args.no_guard:
                        allowed, title = True, ""
                    else:
                        allowed, title = window_allowed(GUARD_KEYWORDS)
                    if allowed:
                        pag.press(key)
                        result_txt = "sent"
                    else:
                        result_txt = "blocked"
                        banner = "BLOCKED: open your slideshow window"
                if result_txt != "blocked":
                    banner = f"{desc.upper()}" + ("" if result_txt == "sent" else "  (preview)")
                    counts[gesture] += 1
                    if result_txt == "sent":
                        if key == "right":
                            slide += 1
                        elif key == "left":
                            slide = max(0, slide - 1)
                        elif key == "f5":
                            slide = 1
                log.writerow([datetime.now().isoformat(timespec="seconds"), gesture,
                              f"{event_conf:.3f}", key or "", result_txt])
                log_f.flush()
                outcome_counts[result_txt] += 1
                banner_t = now
                last_fire, last_fired, armed = now, gesture, False
                history.clear()

            # ---- HUD ----
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 118), (20, 20, 20), -1)
            frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
            mode = "CONTROL ON" if control else ("RULE PREVIEW" if rule_mode else "PREVIEW")
            put(frame, mode, (15, 28), (0, 0, 255) if control else (0, 200, 255), 0.7, 2)
            put(frame, f"FPS {fps:4.1f}", (w - 110, 28), scale=0.55)
            label = DISPLAY.get(name, name) if name else ("No hand" if ext is None else "Uncertain")
            put(frame, label, (15, 62), (0, 255, 0) if name else (180, 180, 180), 0.9, 2)
            put(frame, f"conf {conf:.0%}", (15, 88), scale=0.55)
            bar(frame, 110, 76, 150, 12, conf, (0, 200, 0))
            put(frame, "hold", (275, 88), scale=0.5)
            bar(frame, 320, 76, 120, 12, 1.0 if cooling else progress, (0, 140, 255) if not cooling else (90, 90, 90))
            put(frame, f"{'cooldown' if cooling else 'ready' if armed or args.repeat else 're-arm: drop hand'}",
                (455, 88), scale=0.5, color=(200, 200, 200))
            elapsed = int(now - start)
            put(frame, f"Timer {elapsed // 60:02d}:{elapsed % 60:02d}   Slide ~{slide}", (15, 110), scale=0.55)
            pointer_label = ("POINTER ON" if pointer_active else
                             "POINTER BLOCKED" if pointer_on else "POINTER OFF")
            pointer_color = (0, 255, 255) if pointer_active else (0, 0, 255) if pointer_on else (180, 180, 180)
            put(frame, pointer_label, (w - 310, 110), pointer_color, 0.45)
            swipe_label = "SWIPE ON" if swipe_enabled else "SWIPE OFF"
            put(frame, swipe_label, (w - 465, 110), (0, 255, 255) if swipe_enabled else (180, 180, 180), 0.45)

            if probs is not None:  # live probability bars per class
                for i, c in enumerate(classes):
                    y = 140 + i * 26
                    put(frame, DISPLAY.get(c, c)[:14], (w - 235, y + 12), scale=0.45)
                    bar(frame, w - 130, y, 115, 12, float(probs[i]), (255, 180, 0))

            if now - banner_t < 1.0 and banner:
                color = (0, 0, 255) if banner.startswith("BLOCKED") else (0, 255, 255)
                put(frame, banner, (w // 2 - 15 * len(banner) // 2 - 40, h // 2), color, 1.0, 3)
            put(frame, "Q quit | C control | P pointer | S swipe navigation | L landmarks",
                (15, h - 15), scale=0.5)

            cv2.imshow("GestureSlide", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("l"):
                show_lm = not show_lm
            elif key == ord("c"):
                if pag is None:
                    try:
                        import pyautogui
                        pyautogui.PAUSE = 0
                        pag = pyautogui
                    except Exception:  # noqa: BLE001
                        pag = None
                control = (not control) and pag is not None
                if not control:
                    pointer_on = False
                    pointer_pos = None
            elif key == ord("p"):
                if control and pag is not None:
                    pointer_on = not pointer_on
                    pointer_pos = None
                    banner = "HAND POINTER ON" if pointer_on else "HAND POINTER OFF"
                else:
                    banner = "PRESS C TO ENABLE GUARDED CONTROL FIRST"
                banner_t = time.time()
            elif key == ord("s"):
                swipe_enabled = not swipe_enabled
                motion_history.clear()
                history.clear()
                banner = "SWIPE NAVIGATION ON" if swipe_enabled else "SWIPE NAVIGATION OFF"
                banner_t = time.time()

    cap.release()
    cv2.destroyAllWindows()
    log_f.close()
    total = int(time.time() - start)
    print(f"\nSession {total // 60:02d}:{total % 60:02d} | commands: {dict(counts) or 'none'}")
    print(f"Log saved: {log_path}")
    report_path = log_path.with_suffix(".json")
    report = {
        "started_at": session_started.isoformat(timespec="seconds"),
        "duration_seconds": total,
        "recognition_mode": "geometric_rules" if rule_mode else "trained_model",
        "gesture_detections": dict(detected_counts),
        "mean_confidence": {
            name: round(confidence_totals[name] / count, 3)
            for name, count in detected_counts.items()
        },
        "command_outcomes": dict(outcome_counts),
        "estimated_slide": slide,
        "hand_pointer_frames": pointer_frames,
        "air_swipe_actions": detected_counts["swipe_left"] + detected_counts["swipe_right"],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Session report saved: {report_path}")


if __name__ == "__main__":
    main()
