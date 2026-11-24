"""Train the TensorFlow/Keras MLP gesture classifier from data/gestures.csv.

Outputs: model/gestures.keras, model/labels.json, model/metrics.json,
         model/confusion_matrix.png, model/training_curves.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

from config import DATA_CSV, LABELS_PATH, METRICS_PATH, MODEL_PATH
from features import NUM_FEATURES, augment

SEED = 42


def build_model(n_classes: int) -> keras.Model:
    return keras.Sequential([
        keras.Input(shape=(NUM_FEATURES,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="gestureslide_mlp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    if not DATA_CSV.exists():
        raise SystemExit("data/gestures.csv not found. Run collect_data.py first.")
    df = pd.read_csv(DATA_CSV)
    classes = sorted(df["label"].unique())
    counts = df["label"].value_counts().to_dict()
    print("Samples per class:", counts)
    if len(classes) < 2:
        raise SystemExit("Need at least 2 gesture classes. Collect more data.")
    if min(counts.values()) < 30:
        raise SystemExit("Fewer than 30 samples in a class. Collect 100-150 per gesture.")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = np.array([classes.index(l) for l in df["label"]])

    # Stratified train / test split, then a validation slice from the training part
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=SEED)
    if not args.no_augment:
        X_fit, y_fit = augment(X_fit, y_fit, copies=4, seed=SEED)  # augment training data only
    print(f"Train {len(X_fit)} | Val {len(X_val)} | Test {len(X_te)} (held-out, never augmented)")

    weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_fit)
    model = build_model(len(classes))
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    hist = model.fit(
        X_fit, y_fit, validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=dict(enumerate(weights)),
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                                 restore_best_weights=True)],
        verbose=2,
    )

    # ---- Evaluation on held-out test data ----
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred)
    print("\n=== TEST RESULTS (held-out data) ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred, target_names=classes, zero_division=0))
    print("Confusion matrix:\n", cm)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps({"classes": classes}, indent=2))
    METRICS_PATH.write_text(json.dumps({
        "test_accuracy": float(acc), "samples_per_class": counts,
        "report": report, "confusion_matrix": cm.tolist(), "classes": classes,
    }, indent=2))

    # ---- Plots ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion matrix (acc {acc:.2%})")
    fig.tight_layout(); fig.savefig(MODEL_PATH.parent / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(hist.history["accuracy"], label="train"); axes[0].plot(hist.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(hist.history["loss"], label="train"); axes[1].plot(hist.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend()
    fig.tight_layout(); fig.savefig(MODEL_PATH.parent / "training_curves.png", dpi=150); plt.close(fig)

    print(f"\nSaved: {MODEL_PATH}, {LABELS_PATH}, metrics.json, confusion_matrix.png, training_curves.png")
    print("Report the accuracy measured above from YOUR data; do not quote any other figure.")


if __name__ == "__main__":
    main()
