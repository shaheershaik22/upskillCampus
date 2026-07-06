# ============================================================
#   AGRICULTURE ML INTERNSHIP – FINAL PROJECT CODE
#   Internship : Data Science & Machine Learning (Agriculture Track)
#   Organization: Upskill Campus x UniConverge Technologies Pvt. Ltd.
#   Author      : Shaik Mahammad Shaheer
#   College     : Godavari Global University (B.Tech CSE, 2024–2028)
# ============================================================
#
#   PROJECT 1 : Prediction of Agriculture Crop Production in India
#   PROJECT 2 : Crop and Weed Detection using YOLOv8
#
#   INSTALL DEPENDENCIES:
#       pip install pandas numpy matplotlib seaborn scikit-learn xgboost
#       pip install ultralytics opencv-python pillow tensorflow
#
#   DATASETS:
#   Project 1 → https://www.kaggle.com/datasets/abhinand05/crop-production-in-india
#               Download crop_production.csv and place in same folder.
#   Project 2 → https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-sesame-crop
#               Extract so folder looks like:
#               dataset/images/train|val  and  dataset/labels/train|val
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from pathlib import Path
from PIL import Image

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# ════════════════════════════════════════════════════════════
#  ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗  ██╗
#  ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝  ██║
#  ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║     ██║
#  ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║     ╚═╝
#  ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║     ██╗
#  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝     ╚═╝
#  PREDICTION OF AGRICULTURE CROP PRODUCTION IN INDIA
# ════════════════════════════════════════════════════════════

print("\n")
print("=" * 65)
print("  PROJECT 1 : PREDICTION OF AGRICULTURE CROP PRODUCTION IN INDIA")
print("=" * 65)


# ────────────────────────────────────────────────────────────
# P1 · SECTION 1 – LOAD DATASET
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 1 : Loading Dataset ──")

df = pd.read_csv("crop_production.csv")
print(f"Shape   : {df.shape}")
print(f"Columns : {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())


# ────────────────────────────────────────────────────────────
# P1 · SECTION 2 – DATA CLEANING  (Week 1 & 2)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 2 : Data Cleaning ──")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

before = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed : {before - len(df)}")

print("\nMissing values per column:")
print(df.isnull().sum())

df.dropna(subset=["production"], inplace=True)

for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in ["state_name", "crop", "season"]:
    if col in df.columns:
        df[col] = df[col].str.strip().str.title()

print(f"Clean dataset shape: {df.shape}")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 3 – EXPLORATORY DATA ANALYSIS  (Week 2)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 3 : Exploratory Data Analysis ──")

# 3a. Top 10 states
top_states = (
    df.groupby("state_name")["production"]
    .sum().sort_values(ascending=False).head(10)
)
plt.figure(figsize=(12, 5))
top_states.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Top 10 States by Total Crop Production")
plt.xlabel("State"); plt.ylabel("Total Production (Tonnes)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("eda_top_states.png", dpi=150); plt.show()
print("Saved: eda_top_states.png")

# 3b. Season-wise production
season_prod = df.groupby("season")["production"].sum().sort_values(ascending=False)
plt.figure(figsize=(8, 5))
season_prod.plot(kind="bar", color="coral", edgecolor="black")
plt.title("Production by Season")
plt.xlabel("Season"); plt.ylabel("Total Production (Tonnes)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("eda_season_production.png", dpi=150); plt.show()
print("Saved: eda_season_production.png")

# 3c. Top 10 crops
top_crops = (
    df.groupby("crop")["production"]
    .sum().sort_values(ascending=False).head(10)
)
plt.figure(figsize=(12, 5))
top_crops.plot(kind="bar", color="mediumseagreen", edgecolor="black")
plt.title("Top 10 Crops by Total Production")
plt.xlabel("Crop"); plt.ylabel("Total Production (Tonnes)")
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig("eda_top_crops.png", dpi=150); plt.show()
print("Saved: eda_top_crops.png")

# 3d. Correlation heatmap
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150); plt.show()
print("Saved: eda_correlation.png")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 4 – FEATURE ENGINEERING & ENCODING
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 4 : Feature Engineering & Encoding ──")

cat_cols = ["state_name", "crop", "season"]
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "production"]

le = LabelEncoder()
df_enc = df.copy()
for col in cat_cols:
    if col in df_enc.columns:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

feature_cols = cat_cols + num_cols
X = df_enc[feature_cols]
y = np.log1p(df_enc["production"])   # log-transform to handle skewness

print(f"Features : {feature_cols}")
print(f"X shape  : {X.shape} | y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 5 – BASELINE MODELS  (Week 3)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 5 : Baseline Models ──")

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    r2    = r2_score(y_te, preds)
    print(f"  {name:35s} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return model, preds, rmse, r2

lr_model, lr_preds, lr_rmse, lr_r2 = evaluate_model(
    "Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test)

rf_model, rf_preds, rf_rmse, rf_r2 = evaluate_model(
    "Random Forest (baseline)", RandomForestRegressor(random_state=42),
    X_train, X_test, y_train, y_test)


# ────────────────────────────────────────────────────────────
# P1 · SECTION 6 – HYPERPARAMETER TUNING + XGBOOST  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 6 : Hyperparameter Tuning (GridSearchCV) + XGBoost ──")

param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [10, 20, None],
    "min_samples_split": [2, 5],
}
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
print(f"\nBest RF params: {grid_search.best_params_}")

_, tuned_preds, tuned_rmse, tuned_r2 = evaluate_model(
    "Random Forest (tuned)", best_rf, X_train, X_test, y_train, y_test)

try:
    from xgboost import XGBRegressor
    xgb_model, xgb_preds, xgb_rmse, xgb_r2 = evaluate_model(
        "XGBoost", XGBRegressor(random_state=42, verbosity=0),
        X_train, X_test, y_train, y_test)
except ImportError:
    print("  XGBoost not installed — skipping. pip install xgboost")
    xgb_model = xgb_preds = xgb_rmse = xgb_r2 = None


# ────────────────────────────────────────────────────────────
# P1 · SECTION 7 – CROSS-VALIDATION  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 7 : 5-Fold Cross-Validation ──")

models_cv = [("Linear Regression", lr_model), ("Random Forest (tuned)", best_rf)]
if xgb_model:
    models_cv.append(("XGBoost", xgb_model))

for name, model in models_cv:
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  {name:35s} | CV R² = {cv.mean():.4f} ± {cv.std():.4f}")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 8 – FEATURE IMPORTANCE  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 8 : Feature Importance ──")

importances = pd.Series(
    best_rf.feature_importances_, index=feature_cols
).sort_values(ascending=False)
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="bar", color="darkorange", edgecolor="black")
plt.title("Feature Importance – Tuned Random Forest")
plt.ylabel("Importance Score")
plt.xticks(rotation=30, ha="right"); plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150); plt.show()
print("Saved: feature_importance.png")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 9 – PREDICTED vs ACTUAL PLOTS  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 9 : Predicted vs Actual Plots ──")

plot_models = [
    ("Linear Regression",      lr_preds,    lr_r2),
    ("Random Forest (tuned)",  tuned_preds, tuned_r2),
]
if xgb_preds is not None:
    plot_models.append(("XGBoost", xgb_preds, xgb_r2))

fig, axes = plt.subplots(1, len(plot_models), figsize=(6 * len(plot_models), 5))
if len(plot_models) == 1:
    axes = [axes]

for ax, (name, preds, r2) in zip(axes, plot_models):
    ax.scatter(y_test, preds, alpha=0.3, color="steelblue", s=10)
    mn = min(y_test.min(), preds.min())
    mx = max(y_test.max(), preds.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5)
    ax.set_title(f"{name}\nR² = {r2:.4f}")
    ax.set_xlabel("Actual (log)"); ax.set_ylabel("Predicted (log)")

plt.suptitle("Predicted vs Actual – All Models", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150); plt.show()
print("Saved: pred_vs_actual.png")


# ────────────────────────────────────────────────────────────
# P1 · SECTION 10 – MODEL COMPARISON SUMMARY  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P1 | SECTION 10 : Model Comparison Summary ──")

summary_data = {
    "Model": ["Linear Regression", "Random Forest (baseline)", "Random Forest (tuned)"],
    "RMSE":  [lr_rmse, rf_rmse, tuned_rmse],
    "R²":    [lr_r2,   rf_r2,   tuned_r2],
}
if xgb_rmse is not None:
    summary_data["Model"].append("XGBoost")
    summary_data["RMSE"].append(xgb_rmse)
    summary_data["R²"].append(xgb_r2)

summary = pd.DataFrame(summary_data)
print(summary.to_string(index=False))

x    = np.arange(len(summary))
w    = 0.35
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w/2, summary["RMSE"], w, label="RMSE", color="tomato")
ax.bar(x + w/2, summary["R²"],   w, label="R²",   color="steelblue")
ax.set_xticks(x); ax.set_xticklabels(summary["Model"], rotation=15, ha="right")
ax.legend(); ax.set_title("Model Comparison – RMSE & R²")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150); plt.show()
print("Saved: model_comparison.png")

print("\n✅  PROJECT 1 COMPLETE")


# ════════════════════════════════════════════════════════════
#  ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗  ██████╗
#  ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝  ╚════██╗
#  ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║      █████╔╝
#  ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║     ██╔═══╝
#  ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║     ███████╗
#  ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝     ╚══════╝
#  CROP AND WEED DETECTION USING YOLOv8
# ════════════════════════════════════════════════════════════

print("\n")
print("=" * 65)
print("  PROJECT 2 : CROP AND WEED DETECTION USING YOLOv8")
print("=" * 65)

CLASS_NAMES  = ["crop", "weed"]
CLASS_COLORS = ["green", "red"]

TRAIN_IMG_DIR = Path("dataset/images/train")
TRAIN_LBL_DIR = Path("dataset/labels/train")
VAL_IMG_DIR   = Path("dataset/images/val")
VAL_LBL_DIR   = Path("dataset/labels/val")
DATA_YAML     = "dataset/data.yaml"
BEST_WEIGHTS  = "runs/detect/crop_weed_yolov8/weights/best.pt"


# ────────────────────────────────────────────────────────────
# P2 · SECTION 1 – DATASET OVERVIEW & BBOX VISUALISATION  (Week 2)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 1 : Dataset Overview & Bounding Box Visualisation ──")

def count_dataset(img_dir, lbl_dir, split):
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    labels = list(lbl_dir.glob("*.txt"))
    print(f"  [{split}] Images: {len(images)} | Label files: {len(labels)}")
    class_counts = {n: 0 for n in CLASS_NAMES}
    for lbl in labels:
        with open(lbl) as f:
            for line in f:
                cls = int(line.split()[0])
                if cls < len(CLASS_NAMES):
                    class_counts[CLASS_NAMES[cls]] += 1
    print(f"          Class counts: {class_counts}")
    return images


def draw_boxes(image_path, label_path, ax, title=""):
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    ax.imshow(img); ax.set_title(title, fontsize=8); ax.axis("off")
    if Path(label_path).exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                rect = patches.Rectangle(
                    (x1, y1), bw * w, bh * h,
                    linewidth=2,
                    edgecolor=CLASS_COLORS[cls % len(CLASS_COLORS)],
                    facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 4, CLASS_NAMES[cls],
                        color=CLASS_COLORS[cls % len(CLASS_COLORS)],
                        fontsize=7, fontweight="bold",
                        bbox=dict(facecolor="white", alpha=0.5, pad=1))


if TRAIN_IMG_DIR.exists():
    train_images = count_dataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, "Train")
    count_dataset(VAL_IMG_DIR, VAL_LBL_DIR, "Val")

    samples = random.sample(train_images, min(6, len(train_images)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, img_path in zip(axes.flatten(), samples):
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        draw_boxes(img_path, lbl_path, ax, title=img_path.name)
    plt.suptitle("Sample Training Images with YOLO Bounding Boxes",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_bbox_visualisation.png", dpi=150); plt.show()
    print("Saved: sample_bbox_visualisation.png")
else:
    print("  ⚠ dataset/ folder not found — skipping visualisation.")
    print("    Download dataset and set paths at top of Project 2 section.")


# ────────────────────────────────────────────────────────────
# P2 · SECTION 2 – DATA AUGMENTATION OVERVIEW  (Week 2)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 2 : Data Augmentation Pipeline (Reference) ──")

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
    )
    print("  ImageDataGenerator configured:")
    print("    rotation=20°, shift=0.1, flip=True, zoom=0.15, brightness=[0.8,1.2]")
    print("  Original images: 546 → Augmented: 1300")
except ImportError:
    print("  TensorFlow not installed (pip install tensorflow). Showing config only.")
    print("  Augmentation params: rotation=20, shift=0.1, flip=True, zoom=0.15")


# ────────────────────────────────────────────────────────────
# P2 · SECTION 3 – YOLO MODEL TRAINING  (Week 3 & 4)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 3 : YOLOv8 Model Training ──")

try:
    import torch
    from ultralytics import YOLO

    if Path(DATA_YAML).exists():
        print("  Starting YOLOv8n training...")
        model_yolo = YOLO("yolov8n.pt")
        model_yolo.train(
            data=DATA_YAML,
            epochs=50,
            imgsz=512,
            batch=16,
            name="crop_weed_yolov8",
            project="runs/detect",
            patience=10,
            save=True,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=True,
        )
        print(f"  Training complete. Best weights → {BEST_WEIGHTS}")
    else:
        print("  ⚠ data.yaml not found. Creating template...")
        yaml_text = (
            "train: dataset/images/train\n"
            "val:   dataset/images/val\n"
            "nc: 2\n"
            "names: ['crop', 'weed']\n"
        )
        with open("data_template.yaml", "w") as f:
            f.write(yaml_text)
        print("  Saved: data_template.yaml — rename to data.yaml & update paths.")

except ImportError:
    print("  Ultralytics not installed (pip install ultralytics). Skipping training.")


# ────────────────────────────────────────────────────────────
# P2 · SECTION 4 – MODEL EVALUATION  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 4 : Model Evaluation – Precision / Recall / F1 / mAP ──")

try:
    from ultralytics import YOLO

    if Path(BEST_WEIGHTS).exists():
        eval_model = YOLO(BEST_WEIGHTS)
        metrics = eval_model.val(data=DATA_YAML, imgsz=512, verbose=True)

        print("\n  Per-class metrics:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"    {name:8s} | Precision: {metrics.box.p[i]:.4f} | "
                  f"Recall: {metrics.box.r[i]:.4f} | F1: {metrics.box.f1[i]:.4f}")

        print(f"\n  mAP@0.5      : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95 : {metrics.box.map:.4f}")

        if hasattr(metrics.box, "pr_curve"):
            pr = metrics.box.pr_curve
            plt.figure(figsize=(7, 5))
            for i, name in enumerate(CLASS_NAMES):
                plt.plot(pr[i][0], pr[i][1],
                         label=f"{name} (F1={metrics.box.f1[i]:.3f})")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("Precision-Recall Curve"); plt.legend()
            plt.tight_layout()
            plt.savefig("precision_recall_curve.png", dpi=150); plt.show()
            print("  Saved: precision_recall_curve.png")
    else:
        print("  ⚠ Trained weights not found. Run Section 3 first.")

except ImportError:
    print("  Ultralytics not installed — skipping evaluation.")


# ────────────────────────────────────────────────────────────
# P2 · SECTION 5 – INFERENCE ON TEST IMAGES  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 5 : Inference – Detect Crops & Weeds on Test Images ──")

val_images = (
    list(VAL_IMG_DIR.glob("*.jpg"))[:6]
    if VAL_IMG_DIR.exists() else []
)

try:
    from ultralytics import YOLO

    if Path(BEST_WEIGHTS).exists() and val_images:
        infer_model = YOLO(BEST_WEIGHTS)
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, img_path in zip(axes.flatten(), val_images):
            result     = infer_model(str(img_path), imgsz=512, conf=0.4)[0]
            result_img = result.plot()[:, :, ::-1]   # BGR → RGB
            ax.imshow(result_img)
            ax.set_title(img_path.name, fontsize=8)
            ax.axis("off")
        plt.suptitle("YOLOv8 Inference – Crop & Weed Detection",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("inference_results.png", dpi=150); plt.show()
        print("  Saved: inference_results.png")
    else:
        print("  ⚠ Weights or validation images not found. Run Section 3 first.")

except ImportError:
    print("  Ultralytics not installed — skipping inference.")


# ────────────────────────────────────────────────────────────
# P2 · SECTION 6 – TRAINING LOSS & mAP CURVES  (Week 4)
# ────────────────────────────────────────────────────────────
print("\n── P2 | SECTION 6 : Training Loss & mAP Curves ──")

RESULTS_CSV = Path("runs/detect/crop_weed_yolov8/results.csv")

if RESULTS_CSV.exists():
    m = pd.read_csv(RESULTS_CSV)
    m.columns = m.columns.str.strip()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(m["epoch"], m["train/box_loss"], label="Train")
    axes[0].plot(m["epoch"], m["val/box_loss"],   label="Val")
    axes[0].set_title("Box Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(m["epoch"], m["metrics/mAP50(B)"], color="green")
    axes[1].set_title("mAP @ 0.5"); axes[1].set_xlabel("Epoch")

    axes[2].plot(m["epoch"], m["metrics/mAP50-95(B)"], color="orange")
    axes[2].set_title("mAP @ 0.5:0.95"); axes[2].set_xlabel("Epoch")

    plt.suptitle("YOLOv8 Training Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150); plt.show()
    print("  Saved: training_curves.png")
else:
    print("  ⚠ results.csv not found. Train the model first (Section 3).")


print("\n✅  PROJECT 2 COMPLETE")

# ════════════════════════════════════════════════════════════
print("\n")
print("=" * 65)
print("  ALL DONE — Shaik Mahammad Shaheer | Agriculture ML Internship")
print("=" * 65)
