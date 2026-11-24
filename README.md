# GestureSlide
AI-based hand-gesture controlled presentation system (classroom deep-learning prototype).

## Tech stack
| Layer | Tool |
|---|---|
| Language / IDE | Python 3.11 (64-bit), VS Code |
| Hand landmarks | MediaPipe Hands (21 landmarks x,y,z = 63 values) |
| Classifier | TensorFlow / Keras MLP (128-64-32, Dropout, Softmax) |
| Video / HUD | OpenCV |
| Evaluation | scikit-learn (stratified split, precision/recall, confusion matrix), matplotlib |
| Data | NumPy, pandas, CSV |
| Slide control | PyAutoGUI (+ PyGetWindow safety guard on Windows) |

MediaPipe only extracts landmarks. The trainable deep-learning model is the Keras MLP (not a CNN).

## Files
```
GestureSlide/
|-- data/gestures.csv        # created by collect_data.py
|-- model/                   # pretrained .task + optional trained model, labels, metrics, plots
|-- logs/                    # session_*.csv and session_*.json written by the live app
|-- config.py                # paths, gesture classes, key mapping, defaults
|-- features.py              # landmark normalisation + augmentation (shared)
|-- collect_data.py
|-- train_model.py
|-- gestureslide.py
|-- requirements.txt
|-- setup.bat  run_collect.bat  run_train.bat  run_app.bat  run_app_control.bat
`-- README.md
```

## Run (Windows)
1. Double-click `setup.bat` (creates `.venv`, installs requirements) - or follow the PDF steps manually.
2. `run_collect.bat` - press **1** palm, **2** fist, **3** thumbs up (**4** peace, optional). Aim for 100-150 each, vary angle/distance/lighting. **0** pause, **Z** discard unsaved, **Q** quit.
3. `run_train.bat` - trains, prints test accuracy/precision/recall/confusion matrix, saves model + plots.
4. `run_app.bat` - preview only. If no trained model exists yet, the live preview uses geometric MediaPipe rules for palm, fist, thumbs-up and peace; collect/train a model for more robust recognition.
5. `run_app_control.bat` - sends keys to the slideshow (open it first and keep it active).

| Gesture | Key | Action |
|---|---|---|
| Open palm | Right | Next slide |
| Closed fist | Left | Previous slide |
| Thumbs up | F5 | Start slideshow |
| Peace sign (optional extra) | B | Blank / unblank screen |

## Additional features (beyond the guide)
- **Active-window safety guard** - keys are sent only when a PowerPoint / slideshow window is active; otherwise shows "BLOCKED". Disable with `--no-guard`.
- **Re-arm logic** - a gesture fires once, then you must drop your hand or switch gesture before it fires again (prevents runaway slide skipping). `--repeat` restores auto-repeat.
- **Scale/position/handedness-invariant features** - wrist-centred, size-normalised, left hand mirrored, so left- and right-handed users and different distances work.
- **Data augmentation** - rotation, scale jitter and noise on training data only; the test set stays untouched.
- **Optional 4th gesture** - peace sign for blank screen; the model adapts to however many classes are in the CSV.
- **Live HUD** - confidence bar, hold-progress bar, cooldown/ready state, per-class probability bars, FPS, presenter timer, estimated slide counter.
- **Hand pointer** - press `C` to enable guarded control, then `P` to move the pointer with your index fingertip while a presentation window is active. It turns off with control and never clicks.
- **Air swipe navigation** - press `S`, then swipe your hand left or right to move backward or forward. Palm/fist navigation is paused while swipe mode is on; thumbs-up and peace actions remain available.
- **Runtime toggles** - `C` control on/off, `P` hand pointer, `S` swipe navigation, `L` landmarks on/off, `Q` quit.
- **Session report** - `logs/session_*.csv` records commands; a matching JSON report summarizes run time, gesture counts, mean confidence, outcomes, estimated slide, pointer use, and air swipes.
- **Report artefacts** - `confusion_matrix.png`, `training_curves.png`, `metrics.json` for your report slides.

## Tuning flags
`python gestureslide.py --control --threshold 0.9 --stable 8 --cooldown 1.5 --camera 1`

## Notes
- Do not quote an accuracy before running training on your own data.
- PyAutoGUI fail-safe stays on: slam the mouse into a screen corner to abort key sending.
- Keep a backup screen recording of the demo in case classroom webcam/projector misbehaves.

