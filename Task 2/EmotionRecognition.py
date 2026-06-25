import os
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Matplotlib 3.7+ compatibility: restore get_cmap if missing
if not hasattr(cm, 'get_cmap'):
    cm.get_cmap = plt.get_cmap

import librosa
import librosa.display

from tqdm import tqdm

from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models   import Sequential
from tensorflow.keras.layers   import (
    Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical

print("✅ All libraries imported.")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  — edit DATASET_PATH to match your setup
# ─────────────────────────────────────────────────────────────

DATASET_PATH = "RAVDESS"   # folder that contains Actor_01 … Actor_24

# All generated outputs will be saved inside this folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def output_path(filename):
    return os.path.join(OUTPUT_DIR, filename)

RANDOM_SEED  = 42
MAX_PAD_LEN  = 174
DURATION     = 3
SAMPLE_RATE  = 22050
N_MFCC       = 40
N_CHROMA     = 12
N_MELS       = 128

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────
# RAVDESS EMOTION MAP
# ─────────────────────────────────────────────────────────────

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}


EMOTION_COLORS = {
    'neutral'  : '#90A4AE',
    'calm'     : '#4FC3F7',
    'happy'    : '#FFD54F',
    'sad'      : '#5C6BC0',
    'angry'    : '#EF5350',
    'fearful'  : '#AB47BC',
    'disgust'  : '#66BB6A',
    'surprised': '#FF7043'
}

# ═══════════════════════════════════════════════════════════════
# PART A — FEATURE EXTRACTION  (same as base code)
# ═══════════════════════════════════════════════════════════════

def extract_features(file_path):
    """
    Extract MFCC + Chroma + Mel features from a .wav file.
    Returns shape (180, 174) or None on error.
    """
    try:
        audio, sr = librosa.load(
            file_path, sr=SAMPLE_RATE,
            duration=DURATION, res_type='kaiser_fast'
        )

        def pad_trim(arr, length):
            if arr.shape[1] < length:
                arr = np.pad(arr, ((0, 0), (0, length - arr.shape[1])))
            return arr[:, :length]

        mfcc   = pad_trim(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC), MAX_PAD_LEN)
        chroma = pad_trim(librosa.feature.chroma_stft(
                              S=np.abs(librosa.stft(audio)), sr=sr, n_chroma=N_CHROMA), MAX_PAD_LEN)
        mel    = pad_trim(librosa.power_to_db(
                              librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS),
                              ref=np.max), MAX_PAD_LEN)

        return np.vstack([mfcc, chroma, mel])   # (180, 174)

    except Exception as e:
        print(f"  ⚠ {os.path.basename(file_path)}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# PART B — LOAD RAVDESS  +  COLLECT ONE SAMPLE PER EMOTION
# ═══════════════════════════════════════════════════════════════

def load_ravdess_data(data_path):

    audio_files = glob.glob(
        os.path.join(data_path, '**', '*.wav'), recursive=True
    )
    if not audio_files:
        return None, None, {}

    print(f"  Found {len(audio_files)} files. Extracting features...\n")

    features_list = []
    labels_list   = []
    sample_files  = {}          # one representative file per emotion

    for fp in tqdm(audio_files, desc="  Processing"):
        parts = os.path.basename(fp).replace('.wav', '').split('-')
        if len(parts) < 3:
            continue
        code = parts[2]
        if code not in EMOTION_MAP:
            continue
        emotion = EMOTION_MAP[code]

        feat = extract_features(fp)
        if feat is not None:
            features_list.append(feat)
            labels_list.append(emotion)
            if emotion not in sample_files:
                sample_files[emotion] = fp   # remember the first file per emotion

    X = np.array(features_list)
    y = np.array(labels_list)

    print(f"\n  Loaded {len(X)} samples.")
    unique, counts = np.unique(y, return_counts=True)
    for e, c in zip(unique, counts):
        print(f"    {e:10s}: {c} samples")

    return X, y, sample_files


# ═══════════════════════════════════════════════════════════════
# PART C — PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def preprocess_data(X, y):
    X = X.transpose(0, 2, 1)                             # (N, 174, 180)
    n, t, f = X.shape
    scaler  = StandardScaler()
    X       = scaler.fit_transform(X.reshape(-1, f)).reshape(n, t, f)

    le      = LabelEncoder()
    y_int   = le.fit_transform(y)
    n_cls   = len(le.classes_)
    y_hot   = to_categorical(y_int, n_cls)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_hot, test_size=0.20,
        random_state=RANDOM_SEED, stratify=y_int
    )

    print(f"  Train: {X_train.shape[0]}   Test: {X_test.shape[0]}")
    print(f"  Classes: {list(le.classes_)}")
    return X_train, X_test, y_train, y_test, le, scaler, n_cls


# ═══════════════════════════════════════════════════════════════
# PART D — CNN + LSTM MODEL
# ═══════════════════════════════════════════════════════════════

def build_model(input_shape, n_classes):
    m = Sequential([
        Conv1D(256, 5, padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),

        Conv1D(128, 5, padding='same', activation='relu'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),

        Conv1D(64,  3, padding='same', activation='relu'),
        BatchNormalization(), MaxPooling1D(2), Dropout(0.3),

        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64,  return_sequences=False), Dropout(0.3),

        Dense(128, activation='relu'), BatchNormalization(), Dropout(0.4),
        Dense(64,  activation='relu'), Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy', metrics=['accuracy'])
    m.summary()
    return m


def train_model(model, X_train, X_test, y_train, y_test):
    cbs = [
        EarlyStopping(monitor='val_accuracy', patience=15,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(output_path('best_emotion_model.h5'), monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-7, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100, batch_size=32, callbacks=cbs, verbose=1
    )
    return history


# ═══════════════════════════════════════════════════════════════
# PART E — CHART FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# ─── Helper: load raw audio for one sample file ──────────────
def _load_audio(path):
    return librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, res_type='kaiser_fast')


# ─────────────────────────────────────────────────────────────
# CHART 1 — Emotion Sample Count Bar Chart
# ─────────────────────────────────────────────────────────────
def chart_emotion_distribution(y):
    
    emotions, counts = np.unique(y, return_counts=True)
    colors  = [EMOTION_COLORS.get(e, '#888') for e in emotions]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(emotions, counts, color=colors, edgecolor='white', height=0.6)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Number of Audio Samples', fontsize=11)
    ax.set_title('RAVDESS — Sample Count per Emotion', fontsize=13, fontweight='bold', pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path('chart1_emotion_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart1_emotion_distribution.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 2 — Per-Emotion MFCC Heatmap Grid
# ─────────────────────────────────────────────────────────────
def chart_mfcc_heatmaps(sample_files):

    emotions = sorted(sample_files.keys())
    n        = len(emotions)
    ncols    = 4
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    fig.suptitle('MFCC Heatmap — One Sample per Emotion',
                 fontsize=14, fontweight='bold', y=1.01)

    axes_flat = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for idx, emotion in enumerate(emotions):
        ax   = axes_flat[idx]
        path = sample_files[emotion]
        audio, sr = _load_audio(path)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        img  = librosa.display.specshow(mfcc, x_axis='time', sr=sr,
                                        ax=ax, cmap='magma')
        ax.set_title(f'{emotion.upper()}',
                     color=EMOTION_COLORS.get(emotion, '#333'),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('MFCC', fontsize=8)
        fig.colorbar(img, ax=ax, format='%+.0f')

    # Hide any unused subplot cells
    for idx in range(len(emotions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart2_mfcc_heatmaps_per_emotion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart2_mfcc_heatmaps_per_emotion.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 3 — Per-Emotion Chroma Heatmap Grid
# ─────────────────────────────────────────────────────────────
def chart_chroma_heatmaps(sample_files):
    
    emotions  = sorted(sample_files.keys())
    n         = len(emotions)
    ncols     = 4
    nrows     = (n + ncols - 1) // ncols
    note_labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    fig.suptitle('Chroma Heatmap — Pitch Class Energy per Emotion',
                 fontsize=14, fontweight='bold', y=1.01)

    axes_flat = axes.flatten()

    for idx, emotion in enumerate(emotions):
        ax    = axes_flat[idx]
        audio, sr = _load_audio(sample_files[emotion])
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=N_CHROMA)

        img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma',
                                       sr=sr, ax=ax, cmap='YlOrRd')
        ax.set_title(f'{emotion.upper()}',
                     color=EMOTION_COLORS.get(emotion, '#333'),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_yticks(range(N_CHROMA))
        ax.set_yticklabels(note_labels, fontsize=7)
        fig.colorbar(img, ax=ax)

    for idx in range(len(emotions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart3_chroma_heatmaps_per_emotion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart3_chroma_heatmaps_per_emotion.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 4 — Per-Emotion Mel Spectrogram Heatmap Grid
# ─────────────────────────────────────────────────────────────
def chart_mel_heatmaps(sample_files):

    emotions = sorted(sample_files.keys())
    n        = len(emotions)
    ncols    = 4
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.8))
    fig.suptitle('Mel Spectrogram Heatmap — Frequency Energy per Emotion',
                 fontsize=14, fontweight='bold', y=1.01)

    axes_flat = axes.flatten()

    for idx, emotion in enumerate(emotions):
        ax    = axes_flat[idx]
        audio, sr = _load_audio(sample_files[emotion])
        mel   = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        img = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel',
                                       sr=sr, ax=ax, cmap='inferno')
        ax.set_title(f'{emotion.upper()}',
                     color=EMOTION_COLORS.get(emotion, '#333'),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Hz', fontsize=8)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')

    for idx in range(len(emotions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart4_mel_heatmaps_per_emotion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart4_mel_heatmaps_per_emotion.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 5 — Per-Emotion Waveform Grid
# ─────────────────────────────────────────────────────────────
def chart_waveforms(sample_files):

    emotions = sorted(sample_files.keys())
    n        = len(emotions)
    ncols    = 4
    nrows    = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 2.8))
    fig.suptitle('Audio Waveform — Raw Signal per Emotion',
                 fontsize=14, fontweight='bold', y=1.01)

    axes_flat = axes.flatten()

    for idx, emotion in enumerate(emotions):
        ax    = axes_flat[idx]
        audio, sr = _load_audio(sample_files[emotion])
        times = np.linspace(0, len(audio) / sr, num=len(audio))
        color = EMOTION_COLORS.get(emotion, '#607D8B')

        ax.plot(times, audio, color=color, linewidth=0.6, alpha=0.85)
        ax.fill_between(times, audio, alpha=0.2, color=color)
        ax.axhline(0, color='white', linewidth=0.4, linestyle='--')
        ax.set_facecolor('#1a1a2e')
        ax.set_title(f'{emotion.upper()}',
                     color=color, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.set_ylim(-1, 1)
        ax.spines[['top', 'right']].set_visible(False)

    for idx in range(len(emotions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.patch.set_facecolor('#0f0f1a')
    plt.tight_layout()
    plt.savefig(output_path('chart5_waveforms_per_emotion.png'), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {output_path('chart5_waveforms_per_emotion.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 6 — Mean MFCC Comparison Across All Emotions
# ─────────────────────────────────────────────────────────────
def chart_mfcc_mean_comparison(sample_files):

    fig, ax = plt.subplots(figsize=(13, 5))

    for emotion in sorted(sample_files.keys()):
        audio, sr = _load_audio(sample_files[emotion])
        mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mean_mfcc = mfcc.mean(axis=1)   # average over time → shape (40,)

        color = EMOTION_COLORS.get(emotion, '#888')
        ax.plot(range(1, N_MFCC + 1), mean_mfcc,
                label=emotion, color=color, linewidth=2, marker='o',
                markersize=3, alpha=0.85)

    ax.set_title('Mean MFCC per Coefficient — All Emotions Compared',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('MFCC Coefficient Index', fontsize=11)
    ax.set_ylabel('Mean Coefficient Value', fontsize=11)
    ax.legend(title='Emotion', bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=9, framealpha=0.7)
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart6_mfcc_mean_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart6_mfcc_mean_comparison.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 7 — Per-Emotion Energy Violin Plot
# ─────────────────────────────────────────────────────────────
def chart_energy_violin(y, X_raw):

    energy = np.sqrt(np.mean(X_raw[:, :N_MFCC, :] ** 2, axis=(1, 2)))

    df = pd.DataFrame({'emotion': y, 'energy': energy})
    order = sorted(df['emotion'].unique())
    palette = [EMOTION_COLORS.get(e, '#888') for e in order]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.violinplot(data=df, x='emotion', y='energy', order=order,
                   palette=palette, inner='quartile',
                   linewidth=1.2, ax=ax)

    ax.set_title('Audio Energy Distribution per Emotion (MFCC RMS proxy)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Emotion', fontsize=11)
    ax.set_ylabel('RMS Energy', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path('chart7_energy_violin_per_emotion.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart7_energy_violin_per_emotion.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 8 — Training Accuracy & Loss Curves
# ─────────────────────────────────────────────────────────────
def chart_training_history(history):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CNN + LSTM Training History — Emotion Recognition',
                 fontsize=13, fontweight='bold')

    epochs = range(1, len(history.history['accuracy']) + 1)

    # Accuracy
    axes[0].plot(epochs, history.history['accuracy'],
                 label='Train', color='#1565C0', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'],
                 label='Validation', color='#E53935', linewidth=2, linestyle='--')
    axes[0].fill_between(epochs, history.history['accuracy'],
                          history.history['val_accuracy'], alpha=0.08, color='#1565C0')
    axes[0].set_title('Accuracy per Epoch', fontsize=11)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)
    axes[0].spines[['top', 'right']].set_visible(False)

    # Loss
    axes[1].plot(epochs, history.history['loss'],
                 label='Train', color='#1565C0', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'],
                 label='Validation', color='#E53935', linewidth=2, linestyle='--')
    axes[1].fill_between(epochs, history.history['loss'],
                          history.history['val_loss'], alpha=0.08, color='#E53935')
    axes[1].set_title('Loss per Epoch', fontsize=11)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart8_training_history.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart8_training_history.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 9 — Confusion Matrix Heatmap (post-training)
# ─────────────────────────────────────────────────────────────
def chart_confusion_matrix(model, X_test, y_test, label_encoder):

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    classes = label_encoder.classes_

    # Normalise to percentages for easier reading
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Confusion Matrix — Emotion Predictions', fontsize=13, fontweight='bold')

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='white', ax=axes[0])
    axes[0].set_title('Raw Counts', fontsize=11)
    axes[0].set_ylabel('True Emotion')
    axes[0].set_xlabel('Predicted Emotion')
    axes[0].tick_params(axis='x', rotation=40)

    # Normalised percentages
    annot = np.array([[f"{v:.1f}%" for v in row] for row in cm_pct])
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Greens',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='white', ax=axes[1],
                vmin=0, vmax=100)
    axes[1].set_title('Normalised (%)', fontsize=11)
    axes[1].set_ylabel('True Emotion')
    axes[1].set_xlabel('Predicted Emotion')
    axes[1].tick_params(axis='x', rotation=40)

    plt.tight_layout()
    plt.savefig(output_path('chart9_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart9_confusion_matrix.png')}")

    # Print classification report
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    return np.diag(cm_pct)   # per-class accuracy (diagonal values)


# ─────────────────────────────────────────────────────────────
# CHART 10 — Per-Class Accuracy Bar Chart (from confusion matrix diagonal)
# ─────────────────────────────────────────────────────────────
def chart_per_class_accuracy(per_class_acc, label_encoder):

    classes = label_encoder.classes_
    colors  = [EMOTION_COLORS.get(c, '#888') for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(classes, per_class_acc, color=colors,
                   edgecolor='white', height=0.55)

    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=np.mean(per_class_acc), color='black', linestyle='--',
               linewidth=1.5, label=f"Mean: {np.mean(per_class_acc):.1f}%")
    ax.set_xlabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Emotion Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path('chart10_per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart10_per_class_accuracy.png')}")


# ─────────────────────────────────────────────────────────────
# CHART 11 — Live Prediction Probability Bar Chart
# ─────────────────────────────────────────────────────────────
def predict_and_chart(file_path, model, label_encoder, scaler):

    print(f"\n  Predicting: {os.path.basename(file_path)}")

    feat = extract_features(file_path)
    if feat is None:
        print("  Could not extract features.")
        return

    # Preprocess single sample
    feat = feat.T[np.newaxis, :, :]                            # (1, 174, 180)
    n, t, f = feat.shape
    feat = scaler.transform(feat.reshape(-1, f)).reshape(n, t, f)

    probs   = model.predict(feat, verbose=0)[0]                # (n_classes,)
    classes = label_encoder.classes_
    pred    = classes[np.argmax(probs)]
    colors  = ['#EF5350' if c == pred else EMOTION_COLORS.get(c, '#888')
               for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Prediction for: {os.path.basename(file_path)}',
                 fontsize=12, fontweight='bold')

    # ── Left: Probability bars ────────────────────────────
    bars = axes[0].bar(classes, probs * 100, color=colors,
                       edgecolor='white', width=0.55)
    for bar, prob in zip(bars, probs * 100):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{prob:.1f}%", ha='center', fontsize=9, fontweight='bold')
    axes[0].set_title(f'Model Prediction: {pred.upper()}', fontsize=12,
                      color='#EF5350', fontweight='bold')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Confidence (%)')
    axes[0].set_ylim(0, 115)
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].spines[['top', 'right']].set_visible(False)

    # ── Right: Waveform of the predicted file ─────────────
    audio, sr = _load_audio(file_path)
    times = np.linspace(0, len(audio) / sr, len(audio))
    pred_color = EMOTION_COLORS.get(pred, '#607D8B')
    axes[1].plot(times, audio, color=pred_color, linewidth=0.7, alpha=0.9)
    axes[1].fill_between(times, audio, alpha=0.2, color=pred_color)
    axes[1].set_facecolor('#1a1a2e')
    axes[1].set_title(f'Waveform — predicted as {pred.upper()}',
                      color=pred_color, fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_ylim(-1, 1)
    axes[1].spines[['top', 'right']].set_visible(False)

    fig.patch.set_facecolor('#f5f5f5')
    plt.tight_layout()
    plt.savefig(output_path('chart11_prediction_result.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {output_path('chart11_prediction_result.png')}")
    print(f"\n  ★ Predicted: {pred.upper()}  ({probs.max() * 100:.1f}% confidence)")

    return pred, probs.max()


# ═══════════════════════════════════════════════════════════════
# MAIN — FULL PIPELINE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ─── STAGE 1: LOAD DATA ─────────────────────────────────
    print("─" * 55)
    print("[1/6]  Loading RAVDESS dataset...")
    print("─" * 55)

    X, y, sample_files = load_ravdess_data(DATASET_PATH)

    if X is None:
        print("\n  ❌  Dataset not found!")
        print("  Steps:")
        print("  1. Open → https://zenodo.org/record/1188976")
        print("  2. Download  Audio_Speech_Actors_01-24.zip")
        print("  3. Extract to a folder")
        print("  4. Set  DATASET_PATH  at the top of this script")
        raise SystemExit(1)

    # ─── STAGE 2: PRE-TRAINING CHARTS ───────────────────────
    print("\n─" * 28)
    print("[2/6]  Generating pre-training charts...")
    print("─" * 55)

    print("  Chart 1 — Emotion distribution...")
    chart_emotion_distribution(y)

    print("  Chart 2 — MFCC heatmaps per emotion...")
    chart_mfcc_heatmaps(sample_files)

    print("  Chart 3 — Chroma heatmaps per emotion...")
    chart_chroma_heatmaps(sample_files)

    print("  Chart 4 — Mel Spectrogram heatmaps per emotion...")
    chart_mel_heatmaps(sample_files)

    print("  Chart 5 — Waveforms per emotion...")
    chart_waveforms(sample_files)

    print("  Chart 6 — Mean MFCC comparison...")
    chart_mfcc_mean_comparison(sample_files)

    print("  Chart 7 — Energy violin plot...")
    chart_energy_violin(y, X)

    # ─── STAGE 3: PREPROCESS ────────────────────────────────
    print("\n─" * 28)
    print("[3/6]  Preprocessing data...")
    print("─" * 55)

    X_train, X_test, y_train, y_test, \
        le, scaler, n_classes = preprocess_data(X, y)

    # ─── STAGE 4: BUILD MODEL ───────────────────────────────
    print("\n─" * 28)
    print("[4/6]  Building CNN + LSTM model...")
    print("─" * 55)

    model = build_model((X_train.shape[1], X_train.shape[2]), n_classes)

    # ─── STAGE 5: TRAIN ─────────────────────────────────────
    print("\n─" * 28)
    print("[5/6]  Training model...")
    print("─" * 55)

    history = train_model(model, X_train, X_test, y_train, y_test)

    # ─── STAGE 6: POST-TRAINING CHARTS ──────────────────────
    print("\n─" * 28)
    print("[6/6]  Generating post-training charts...")
    print("─" * 55)

    print("  Chart 8  — Training history...")
    chart_training_history(history)

    print("  Chart 9  — Confusion matrix...")
    per_class_acc = chart_confusion_matrix(model, X_test, y_test, le)

    print("  Chart 10 — Per-emotion accuracy...")
    chart_per_class_accuracy(per_class_acc, le)

    # ─── OPTIONAL: CHART 11 — LIVE PREDICTION ───────────────
    # Uncomment and point to any .wav file to see prediction chart:
    test_file = "RAVDESS/Actor_01/03-01-03-01-01-01-01.wav"
    predict_and_chart(test_file, model, le, scaler)

    # ─── SAVE MODEL ARTEFACTS ───────────────────────────────
    model.save(output_path('emotion_recognition_model.h5'))
    np.save(output_path('label_classes.npy'), le.classes_)
    with open(output_path('scaler.pkl'), 'wb') as fh:
        pickle.dump(scaler, fh)

    # ─── FINAL SUMMARY ──────────────────────────────────────
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 62)
    print("  ✅  COMPLETE!")
    print(f"  ★  Test Accuracy  : {test_acc * 100:.2f}%")
    print("─" * 62)
    print("  Saved Model Files:")
    print(f"    {output_path('emotion_recognition_model.h5')}")
    print(f"    {output_path('label_classes.npy')}")
    print(f"    {output_path('scaler.pkl')}")
    print("─" * 62)
    print("  Saved Chart Files inside output/ folder:")
    charts = [
        f"{output_path('chart1_emotion_distribution.png')}    ← sample counts",
        output_path('chart2_mfcc_heatmaps_per_emotion.png'),
        output_path('chart3_chroma_heatmaps_per_emotion.png'),
        output_path('chart4_mel_heatmaps_per_emotion.png'),
        output_path('chart5_waveforms_per_emotion.png'),
        output_path('chart6_mfcc_mean_comparison.png'),
        output_path('chart7_energy_violin_per_emotion.png'),
        output_path('chart8_training_history.png'),
        output_path('chart9_confusion_matrix.png'),
        output_path('chart10_per_class_accuracy.png'),
        f"{output_path('chart11_prediction_result.png')}      ← prediction chart",
    ]
    for c in charts:
        print(f" {c}")
    print("=" * 62)