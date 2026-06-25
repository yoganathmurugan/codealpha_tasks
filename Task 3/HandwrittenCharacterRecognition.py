import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ── GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATASET_TYPE = 'mnist'

# ┌── Image dimensions ────────────────────────────────────────────────────────┐
IMG_SIZE  = 28
CHANNELS  = 1

# ┌── Training hyperparameters ────────────────────────────────────────────────┐
BATCH_SIZE     = 128
EPOCHS         = 30
LEARNING_RATE  = 0.001

# ┌── Paths ───────────────────────────────────────────────────────────────────┐
MODEL_PATH  = 'codealpha_task3_cnn.h5'
OUTPUT_DIR  = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ┌── Reproducibility ─────────────────────────────────────────────────────────┐
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Print startup banner
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("    CODEALPHA  —  TASK 3: HANDWRITTEN CHARACTER RECOGNITION")
print("    CNN Deep Learning  |  MNIST Dataset")
print("=" * 65)
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Dataset     : {DATASET_TYPE.upper()}")
print(f"  Image Size  : {IMG_SIZE}×{IMG_SIZE} grayscale")
print(f"  Batch Size  : {BATCH_SIZE}  |  Max Epochs : {EPOCHS}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_type: str):
    """
    Load training and testing images.

    MNIST  → 60,000 training + 10,000 testing images, digits 0–9
    EMNIST → 112,800 training + 18,800 testing images, 47 classes

    Returns:
        X_train, y_train  — training images & labels
        X_test,  y_test   — testing  images & labels
        num_classes        — how many unique classes
        class_names        — human-readable class labels list
    """
    print("\n" + "─" * 65)
    print(f"[STEP 1]  Loading {dataset_type.upper()} Dataset...")

    # ── MNIST (default) ───────────────────────────────────────────────────────
    if dataset_type == 'mnist':

        # Keras downloads and caches MNIST automatically the first time
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        num_classes  = 10
        class_names  = [str(i) for i in range(10)]   # ['0','1',...,'9']

    # ── EMNIST (optional extension) ───────────────────────────────────────────
    elif dataset_type == 'emnist':

        try:
            import tensorflow_datasets as tfds  # pip install tensorflow-datasets
        except ImportError:
            raise ImportError(
                "\n  EMNIST needs an extra package.\n"
                "  Run:  pip install tensorflow-datasets\n"
                "  Or set DATASET_TYPE = 'mnist' to start quickly.\n"
            )

        print("  Loading EMNIST Balanced — first run may take a few minutes…")

        (ds_train, ds_test), info = tfds.load(
            'emnist/balanced',
            split        = ['train', 'test'],
            as_supervised= True,
            with_info    = True
        )

        num_classes = 47
        class_names = (
            list('0123456789') +
            list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') +
            list('abdefghnqrt')
        )

        imgs_tr, lbs_tr = zip(*tfds.as_numpy(ds_train))
        imgs_te, lbs_te = zip(*tfds.as_numpy(ds_test))

        X_train = np.array(imgs_tr)[:, :, :, 0]
        X_test  = np.array(imgs_te)[:, :, :, 0]
        y_train = np.array(lbs_tr)
        y_test  = np.array(lbs_te)
        X_train = np.transpose(X_train, (0, 2, 1))
        X_test  = np.transpose(X_test,  (0, 2, 1))

    else:
        raise ValueError(f"Unknown dataset '{dataset_type}'. Use 'mnist' or 'emnist'.")

    # Print dataset summary
    print(f"  ✓  Training images : {X_train.shape[0]:>7,}   shape → {X_train.shape}")
    print(f"  ✓  Testing  images : {X_test.shape[0]:>7,}   shape → {X_test.shape}")
    print(f"  ✓  Classes         : {num_classes}")
    print(f"  ✓  Class names     : {class_names}")

    return X_train, y_train, X_test, y_test, num_classes, class_names


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ── PREPROCESS DATA
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(X_train, y_train, X_test, y_test, num_classes):
    """
    Prepare raw pixel data for the neural network.

    THREE OPERATIONS:
    ────────────────────────────────────────────────────────────────
    1. NORMALISE  :  Pixel values  0–255  →  floats  0.0–1.0
       WHY?  Neural networks train faster and more stably with small values.

    2. RESHAPE    :  (N, 28, 28)  →  (N, 28, 28, 1)
       WHY?  Conv2D layers expect a "channels" dimension at the end.
             Grayscale = 1 channel.  RGB would be 3.

    3. ONE-HOT    :  integer 5  →  [0,0,0,0,0,1,0,0,0,0]
       WHY?  Categorical crossentropy loss needs this vector format.
             The model outputs one probability per class,
             so labels must match that shape.
    """
    print("\n" + "─" * 65)
    print("[STEP 2]  Preprocessing Data…")

    # 1 ── Normalise
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0

    # 2 ── Reshape  →  add channel dimension
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, CHANNELS)
    X_test  = X_test.reshape(-1,  IMG_SIZE, IMG_SIZE, CHANNELS)

    # 3 ── One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat  = to_categorical(y_test,  num_classes)

    print(f"  ✓  X_train : {X_train.shape}  pixel range [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"  ✓  X_test  : {X_test.shape}")
    print(f"  ✓  y_train : {y_train_cat.shape}  (one-hot encoded)")

    return X_train, y_train_cat, X_test, y_test_cat


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ── VISUALISE SAMPLE TRAINING IMAGES
# ─────────────────────────────────────────────────────────────────────────────

def visualize_samples(X_train, y_train_raw, class_names, n: int = 25):
    """
    Display a 5×5 grid of random training images to verify data loaded correctly.
    Each image shows its true label below the picture.
    """
    print("\n[INFO]  Generating sample data grid…")

    indices = np.random.choice(len(X_train), n, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(
        f'25 Random Training Samples — {DATASET_TYPE.upper()}',
        fontsize=15, fontweight='bold', y=1.01
    )

    for i, idx in enumerate(indices):
        ax = axes[i // 5, i % 5]
        ax.imshow(X_train[idx].squeeze(), cmap='gray_r')
        ax.set_title(f'Label: {class_names[y_train_raw[idx]]}', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_sample_images.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ── BUILD THE CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

def build_cnn_model(input_shape: tuple, num_classes: int) -> keras.Model:
    
    print("\n" + "─" * 65)
    print("[STEP 3]  Building CNN Architecture…")

    model = models.Sequential(name="CodeAlpha_HandwrittenCNN")

    # ── BLOCK 1 — learn simple features: edges, dots, lines ──────────────────
    model.add(layers.Conv2D(
        filters     = 32,         # 32 learnable filters
        kernel_size = (3, 3),     # Each filter covers a 3×3 pixel patch
        activation  = 'relu',     # Negative → 0; positive → kept as-is
        padding     = 'same',     # Output same spatial size as input
        input_shape = input_shape,
        name        = 'conv1a'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv1b'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2), name='pool1'))   # 28×28 → 14×14
    model.add(layers.Dropout(0.25, name='drop1'))

    # ── BLOCK 2 — learn medium features: curves, partial digit shapes ─────────
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2a'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv2b'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2), name='pool2'))   # 14×14 → 7×7
    model.add(layers.Dropout(0.25, name='drop2'))

    # ── BLOCK 3 — learn high-level features: near-complete digit shapes ───────
    model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv3'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2), name='pool3'))   # 7×7 → 3×3
    model.add(layers.Dropout(0.25, name='drop3'))

    # ── FLATTEN — 3D (3,3,128) → 1D (1152,) ─────────────────────────────────
    model.add(layers.Flatten(name='flatten'))

    # ── FULLY CONNECTED — combine all features, make final decision ──────────
    model.add(layers.Dense(256, activation='relu', name='dense1'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.50, name='drop4'))   # Heavier dropout near output

    model.add(layers.Dense(128, activation='relu', name='dense2'))
    model.add(layers.Dropout(0.30, name='drop5'))

    # ── OUTPUT — one neuron per class, softmax converts to probabilities ──────
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))

    # ── Compile — set loss function, optimiser, and metric ───────────────────
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss      = 'categorical_crossentropy',  # Standard for multi-class problems
        metrics   = ['accuracy']
    )

    model.summary()
    print(f"\n  ✓  Total parameters : {model.count_params():,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ── DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def create_data_generators(X_train, y_train_cat):
   
    print("\n[INFO]  Setting up Data Augmentation…")

    datagen = ImageDataGenerator(
        rotation_range     = 10,
        width_shift_range  = 0.1,
        height_shift_range = 0.1,
        zoom_range         = 0.1,
        shear_range        = 0.1,
        fill_mode          = 'nearest',
        validation_split   = 0.1
    )

    datagen.fit(X_train)

    # Training stream — 90 % of X_train
    train_gen = datagen.flow(
        X_train, y_train_cat,
        batch_size = BATCH_SIZE,
        subset     = 'training',
        seed       = SEED
    )

    # Validation stream — remaining 10 %
    val_gen = datagen.flow(
        X_train, y_train_cat,
        batch_size = BATCH_SIZE,
        subset     = 'validation',
        seed       = SEED
    )

    print(f"  ✓  Training   batches / epoch : {len(train_gen)}")
    print(f"  ✓  Validation batches / epoch : {len(val_gen)}")

    return train_gen, val_gen


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ── TRAIN THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model: keras.Model, train_gen, val_gen):
   
    print("\n" + "─" * 65)
    print("[STEP 4]  Training the CNN Model…")
    print(f"  Max epochs : {EPOCHS}  |  Early stopping patience : 7")

    callbacks = [
        EarlyStopping(
            monitor            = 'val_loss',
            patience           = 7,
            restore_best_weights = True,
            verbose            = 1
        ),
        ModelCheckpoint(
            filepath       = MODEL_PATH,
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 0.5,       # New LR = old LR × 0.5
            patience = 4,
            min_lr   = 1e-7,
            verbose  = 1
        )
    ]

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = EPOCHS,
        callbacks       = callbacks,
        verbose         = 1
    )

    print(f"\n  ✓  Training complete!  Best model → {MODEL_PATH}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── EVALUATE THE MODEL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test_cat, y_test_raw, class_names):

    print("\n" + "─" * 65)
    print("[STEP 5]  Evaluating on Test Set…")

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"  ✓  Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  ✓  Test Loss     : {test_loss:.4f}")

    # Get predicted class index for each image
    y_pred_probs   = model.predict(X_test, verbose=0)     
    y_pred_classes = np.argmax(y_pred_probs, axis=1)     

    print("\n  ── Classification Report ")
    print(classification_report(
        y_test_raw,
        y_pred_classes,
        target_names = class_names,
        digits       = 4
    ))

    return y_pred_classes, y_pred_probs


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ── PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history):
 
    print("\n[INFO]  Plotting training history…")

    epochs_ran = range(1, len(history.history['accuracy']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CNN Training History', fontsize=16, fontweight='bold')

    # Accuracy
    axes[0].plot(epochs_ran, history.history['accuracy'],
                 'b-o', ms=3, lw=2, label='Train Accuracy')
    axes[0].plot(epochs_ran, history.history['val_accuracy'],
                 'r-s', ms=3, lw=2, label='Val Accuracy')
    axes[0].set_title('Accuracy per Epoch', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0.85, 1.01])

    # Loss
    axes[1].plot(epochs_ran, history.history['loss'],
                 'b-o', ms=3, lw=2, label='Train Loss')
    axes[1].plot(epochs_ran, history.history['val_loss'],
                 'r-s', ms=3, lw=2, label='Val Loss')
    axes[1].set_title('Loss per Epoch', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 ── SAMPLE PREDICTIONS GRID
# ─────────────────────────────────────────────────────────────────────────────

def plot_sample_predictions(model, X_test, y_test_raw, class_names, n: int = 25):
  
    print("\n[INFO]  Visualising sample predictions…")

    indices     = np.random.choice(len(X_test), n, replace=False)
    samples     = X_test[indices]
    true_labels = y_test_raw[indices]

    preds       = model.predict(samples, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1) * 100

    fig, axes = plt.subplots(5, 5, figsize=(14, 14))
    fig.suptitle(
        'Test Set Predictions  —  Green ✓ Correct  |  Red ✗ Wrong',
        fontsize=14, fontweight='bold'
    )

    for i in range(n):
        ax    = axes[i // 5, i % 5]
        true  = class_names[true_labels[i]]
        pred  = class_names[pred_labels[i]]
        color = 'green' if true == pred else 'red'

        ax.imshow(samples[i].squeeze(), cmap='gray_r')
        ax.set_title(
            f'True: {true}  |  Pred: {pred}\n({confidences[i]:.1f}%)',
            color=color, fontsize=9, fontweight='bold'
        )
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_sample_predictions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 ── CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names):
 
    print("\n[INFO]  Plotting confusion matrix…")

    cm         = confusion_matrix(y_true, y_pred)
    cm_pct     = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Confusion Matrix', fontsize=15, fontweight='bold')

    # Left: raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title('Prediction Counts',  fontsize=12)
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Right: recall % per class
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[1])
    axes[1].set_title('Recall % per Class (row-normalised)', fontsize=12)
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '04_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 ── MISCLASSIFIED EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

def plot_misclassified(X_test, y_test_raw, y_pred, class_names, n: int = 20):
   
    print("\n[INFO]  Displaying misclassified images…")

    wrong_idx = np.where(y_pred != y_test_raw)[0]
    total_wrong = len(wrong_idx)
    print(f"  Total misclassified : {total_wrong} / {len(y_test_raw)}")

    if total_wrong == 0:
        print("  Perfect score — no errors to display!")
        return

    n       = min(n, total_wrong)
    samples = np.random.choice(wrong_idx, n, replace=False)
    cols    = 5
    rows    = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    fig.suptitle(
        f'Misclassified Examples  ({total_wrong} total errors)',
        fontsize=14, fontweight='bold', color='darkred'
    )

    for i, idx in enumerate(samples):
        ax = axes[i]
        ax.imshow(X_test[idx].squeeze(), cmap='gray_r')
        ax.set_title(
            f'True: {class_names[y_test_raw[idx]]}\n'
            f'Pred: {class_names[y_pred[idx]]}',
            color='red', fontsize=9
        )
        ax.axis('off')

    for j in range(n, len(axes)):   # Hide unused subplot slots
        axes[j].axis('off')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '05_misclassified.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 ── CLASS DISTRIBUTION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(y_train_raw, y_test_raw, class_names):
   
    print("\n[INFO]  Plotting class distribution…")

    train_counts = [np.sum(y_train_raw == i) for i in range(len(class_names))]
    test_counts  = [np.sum(y_test_raw  == i) for i in range(len(class_names))]

    x = np.arange(len(class_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, train_counts, w, label='Training',   color='steelblue',  alpha=0.8)
    ax.bar(x + w/2, test_counts,  w, label='Testing',    color='darkorange', alpha=0.8)

    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Number of Images')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '06_class_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 15 ── PREDICT ON A CUSTOM IMAGE (optional)
# ─────────────────────────────────────────────────────────────────────────────

def predict_custom_image(image_path: str, class_names: list):
    
    try:
        import cv2
    except ImportError:
        print("  OpenCV not installed.  Run: pip install opencv-python")
        return

    print(f"\n[CUSTOM PREDICT]  Loading image: {image_path}")

    if not os.path.exists(image_path):
        print(f"  ✗  File not found: {image_path}")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"  ✗  Saved model not found: {MODEL_PATH}  — train first.")
        return

    saved_model = keras.models.load_model(MODEL_PATH)

    # ── Load & preprocess ─────────────────────────────────────────────────────
    img     = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    
    if np.mean(resized) > 127:
        resized = cv2.bitwise_not(resized)

    normed  = resized.astype('float32') / 255.0
    inp     = normed.reshape(1, IMG_SIZE, IMG_SIZE, CHANNELS)

    # ── Predict ───────────────────────────────────────────────────────────────
    probs      = saved_model.predict(inp, verbose=0)[0]
    pred_class = np.argmax(probs)
    confidence = probs[pred_class] * 100

    print(f"  ✓  Predicted  : '{class_names[pred_class]}'")
    print(f"  ✓  Confidence : {confidence:.2f}%")
    print(f"  Top 3 predictions:")
    for rank, idx in enumerate(np.argsort(probs)[::-1][:3]):
        print(f"    {rank+1}. '{class_names[idx]}'  →  {probs[idx]*100:.2f}%")

    # ── Visualise ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(resized, cmap='gray_r')
    axes[0].set_title(
        f"Predicted: '{class_names[pred_class]}'  ({confidence:.1f}%)",
        fontsize=13, fontweight='bold'
    )
    axes[0].axis('off')

    colors = ['green' if i == pred_class else 'steelblue' for i in range(len(class_names))]
    axes[1].bar(class_names, probs, color=colors)
    axes[1].set_title('Confidence per Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'custom_prediction.png'), dpi=150)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 16 ── FINAL SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_final_summary(test_acc: float):
    print("\n" + "=" * 65)
    print("    CODEALPHA  —  TASK 3  COMPLETE")
    print("=" * 65)
    print(f"  ✓  Final Test Accuracy : {test_acc * 100:.2f}%")
    print(f"  ✓  Model saved at      : {MODEL_PATH}")
    print(f"  ✓  Output plots        : ./{OUTPUT_DIR}/")
    print()
    print("  Files generated:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"      ├── {fname}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 17 ── MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, num_classes, class_names = \
        load_dataset(DATASET_TYPE)

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    X_train, y_train_cat, X_test, y_test_cat = preprocess_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw, num_classes
    )

    # Keep raw integer labels for evaluation functions
    y_test_labels = y_test_raw.copy()

    # ── Step 3: Visualise samples ────────────────────────────────────────────
    visualize_samples(X_train, y_train_raw, class_names)

    # ── Step 4: Build CNN ─────────────────────────────────────────────────────
    model = build_cnn_model(
        input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS),
        num_classes = num_classes
    )

    # ── Step 5: Data augmentation generators ─────────────────────────────────
    train_gen, val_gen = create_data_generators(X_train, y_train_cat)

    # ── Step 6: Train ─────────────────────────────────────────────────────────
    history = train_model(model, train_gen, val_gen)

    # ── Step 7: Evaluate ──────────────────────────────────────────────────────
    y_pred_classes, y_pred_probs = evaluate_model(
        model, X_test, y_test_cat, y_test_labels, class_names
    )

    # Fetch final accuracy for summary
    _, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)

    # ── Step 8: Training curves ───────────────────────────────────────────────
    plot_training_history(history)

    # ── Step 9: Sample predictions grid ──────────────────────────────────────
    plot_sample_predictions(model, X_test, y_test_labels, class_names)

    # ── Step 10: Confusion matrix ─────────────────────────────────────────────
    plot_confusion_matrix(y_test_labels, y_pred_classes, class_names)

    # ── Step 11: Misclassified ────────────────────────────────────────────────
    plot_misclassified(X_test, y_test_labels, y_pred_classes, class_names)

    # ── Step 12: Class distribution ───────────────────────────────────────────
    plot_class_distribution(y_train_raw, y_test_labels, class_names)

    # ── Step 13: Final summary ────────────────────────────────────────────────
    print_final_summary(test_acc)

if __name__ == "__main__":
    main()