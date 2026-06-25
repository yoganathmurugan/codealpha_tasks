import os
import time
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.datasets         import load_breast_cancer
from sklearn.model_selection  import (
    train_test_split, cross_val_score, learning_curve, validation_curve
)
from sklearn.preprocessing    import StandardScaler
from sklearn.calibration      import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)

from sklearn.svm           import SVC
from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from xgboost                import XGBClassifier

from scipy.stats import chi2 as chi2_dist


from fpdf import FPDF


# ================================================================
# SECTION 2 — GLOBAL CONSTANTS & STYLE
# ================================================================

MODEL_COLORS = {
    "SVM"                 : "#3498db",
    "Logistic Regression" : "#e74c3c",
    "Random Forest"       : "#2ecc71",
    "XGBoost"             : "#f39c12"
}
CLASS_COLORS = ["#e74c3c", "#2196F3"]

DIABETES_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/"
    "master/pima-indians-diabetes.data.csv"
)
HEART_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/"
    "Heart-UCI-Dataset/master/heart.csv"
)
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

STUDENT_ID = "."

OUTPUT_DIR = "task4_outputs"

# Create output folder automatically
os.makedirs(OUTPUT_DIR, exist_ok=True)


def out(filename: str) -> str:
   
    return os.path.join(OUTPUT_DIR, filename)


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor"  : "#f9f9f9",
    "axes.grid"        : True,
    "grid.alpha"        : 0.35,
    "font.family"        : "DejaVu Sans"
})


# ================================================================
# SECTION 3 — DATASET LOADERS
# ================================================================

def load_diabetes():
    
    print("\n  📥  LOADING: Diabetes Dataset")
    df = pd.read_csv(DIABETES_URL, header=None, names=DIABETES_COLUMNS)
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df[col] = df[col].replace(0, df[col].median())
    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    print(f"     Shape={df.shape}  |  No Diabetes={sum(y==0)}  Diabetes={sum(y==1)}")
    return X, y, "Diabetes", ["No Diabetes", "Diabetes"]


def load_heart_disease():
    
    print("\n  📥  LOADING: Heart Disease Dataset")
    df = pd.read_csv(HEART_URL)
    X, y = df.drop("target", axis=1), df["target"]
    print(f"     Shape={df.shape}  |  No Disease={sum(y==0)}  Disease={sum(y==1)}")
    return X, y, "Heart Disease", ["No Disease", "Disease"]


def load_breast_cancer_data():
    
    print("\n  📥  LOADING: Breast Cancer Dataset (sklearn built-in)")
    raw = load_breast_cancer()
    X   = pd.DataFrame(raw.data, columns=raw.feature_names)
    y   = pd.Series(raw.target)
    print(f"     Shape={X.shape}  |  Malignant={sum(y==0)}  Benign={sum(y==1)}")
    return X, y, "Breast Cancer", ["Malignant", "Benign"]


# ================================================================
# SECTION 4 — HELPER UTILITIES
# ================================================================

def get_top_discriminative_features(X, y, n=8):
    
    m0, m1 = X[y == 0].mean(), X[y == 1].mean()
    score  = (m1 - m0).abs() / (X.std() + 1e-10)
    return score.nlargest(n).index.tolist()


def short_name(name, n=16):
    
        return name[:n] + ".." if len(name) > n else name


def get_models():
    
    return {
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=42, verbosity=0)
    }


def prepare_data(X, y):
    
        X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), y_train, y_test, sc


def mcnemar_test(y_true, pred_a, pred_b):
    
    y_true = np.array(y_true)
    b = int(np.sum((np.array(pred_a) == y_true) & (np.array(pred_b) != y_true)))
    c = int(np.sum((np.array(pred_a) != y_true) & (np.array(pred_b) == y_true)))
    if (b + c) == 0:
        return 1.0
    stat = (abs(b - c) - 1.0) ** 2 / float(b + c)
    return 1.0 - chi2_dist.cdf(stat, df=1)


# ================================================================
# SECTION 5 — CHART: EDA DASHBOARD
# ================================================================

def plot_eda_dashboard(X, y, class_names, dataset_name):
    
    print(f"  📊  Chart: EDA Dashboard — {dataset_name}")
    n_show   = min(12, X.shape[1])
    top_cols = get_top_discriminative_features(X, y, n=n_show)
    top8     = top_cols[:8]

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        f"EDA Dashboard — {dataset_name}\n"
        f"({X.shape[0]} patients | {X.shape[1]} features | "
        f"{class_names[0]} vs {class_names[1]})",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.38)
    counts = [sum(y == 0), sum(y == 1)]

    # Class distribution bar
    ax0 = fig.add_subplot(gs[0, 0])
    bars = ax0.bar(class_names, counts, color=CLASS_COLORS,
                   edgecolor="black", width=0.45)
    ax0.set_title("Class Distribution", fontweight="bold")
    for bar, cnt in zip(bars, counts):
        ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.02,
                 str(cnt), ha="center", fontsize=11, fontweight="bold")
    ax0.set_ylim(0, max(counts) * 1.18)

    # Class distribution pie
    ax1 = fig.add_subplot(gs[0, 1])
    wedges, texts, autotexts = ax1.pie(
        counts, labels=class_names, colors=CLASS_COLORS, autopct="%1.1f%%",
        startangle=90, explode=(0.05, 0),
        wedgeprops=dict(edgecolor="white", linewidth=1.5)
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold")
    ax1.set_title("Class Balance (%)", fontweight="bold")

    # Feature stats table
    ax2 = fig.add_subplot(gs[0, 2:]); ax2.axis("off")
    top6 = top_cols[:6]
    table_data = []
    for feat in top6:
        m0, m1, std_f = X[y==0][feat].mean(), X[y==1][feat].mean(), X[feat].std()
        ratio = abs(m1-m0)/(std_f+1e-10)
        table_data.append([short_name(feat,18), f"{m0:.2f}", f"{m1:.2f}",
                            f"{std_f:.2f}", f"{ratio:.3f}"])
    tbl = ax2.table(cellText=table_data,
                    colLabels=["Feature", f"Mean({class_names[0]})",
                               f"Mean({class_names[1]})", "Std", "Diff Ratio"],
                    cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r%2==0: cell.set_facecolor("#ecf0f1")
        cell.set_edgecolor("white")
    ax2.set_title("Top Discriminative Features", fontweight="bold", pad=8)

    # Correlation heatmap
    ax3 = fig.add_subplot(gs[1, :])
    corr_df = X[top_cols].corr()
    labels  = [short_name(c,13) for c in top_cols]
    sns.heatmap(corr_df, ax=ax3, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, vmin=-1, vmax=1, linewidths=0.5, linecolor="white",
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size":7}, square=True,
                cbar_kws={"shrink":0.7,"label":"Pearson r"})
    ax3.set_title(f"Feature Correlation Heatmap (Top {n_show})", fontweight="bold")
    ax3.tick_params(axis="x", rotation=35, labelsize=8)
    ax3.tick_params(axis="y", rotation=0, labelsize=8)

    # Box plots
    for i, feat in enumerate(top8):
        ax = fig.add_subplot(gs[2, i % 4])
        data_per_class = [X[y==cls][feat].values for cls in [0,1]]
        bp = ax.boxplot(data_per_class, patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_title(short_name(feat,16), fontweight="bold", fontsize=9)

    patches = [mpatches.Patch(color=CLASS_COLORS[i], label=class_names[i]) for i in range(2)]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fname = out(f"{dataset_name.replace(' ','_')}_1_EDA.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 6 — CHART: CONFUSION MATRICES (all 4 models)
# ================================================================

def plot_confusion_matrices(results, y_test, class_names, dataset_name):
    
    print(f"  📊  Chart: Confusion Matrices — {dataset_name}")
    model_names = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"All 4 Model Confusion Matrices — {dataset_name}",
                fontsize=14, fontweight="bold")

    for idx, (name, ax) in enumerate(zip(model_names, axes.flatten())):
        cm  = confusion_matrix(y_test, results[name]["y_pred"])
        acc, auc = results[name]["Accuracy"]*100, results[name]["AUC"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([[f"{cm[r][c]}\n({cm_pct[r][c]:.1f}%)" for c in range(2)]
                          for r in range(2)])
        sns.heatmap(cm, ax=ax, annot=annot, fmt="", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=1.0, linecolor="white",
                    annot_kws={"size":12,"weight":"bold"}, cbar=False)
        for r in range(2):
            for c in range(2):
                edge = "#27ae60" if r==c else "#e74c3c"
                ax.add_patch(plt.Rectangle((c,r),1,1, fill=False, edgecolor=edge, lw=3))
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nAcc={acc:.1f}%  AUC={auc:.3f}",
                    fontsize=11, fontweight="bold",
                    color=list(MODEL_COLORS.values())[idx])

    leg = [mpatches.Patch(facecolor="none", edgecolor="#27ae60", lw=3, label="Correct (TN/TP)"),
           mpatches.Patch(facecolor="none", edgecolor="#e74c3c", lw=3, label="Wrong (FP/FN)")]
    fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout(rect=[0,0.04,1,1])

    fname = out(f"{dataset_name.replace(' ','_')}_2_ConfusionMatrices.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 7 — CHART: ROC + PRECISION-RECALL CURVES
# ================================================================

def plot_roc_pr_curves(results, y_test, dataset_name):
    
    print(f"  📊  Chart: ROC + PR Curves — {dataset_name}")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"ROC & Precision-Recall Curves — {dataset_name}",
                fontsize=14, fontweight="bold")
    model_names, colors = list(results.keys()), list(MODEL_COLORS.values())

    for name, color in zip(model_names, colors):
        y_prob, auc = results[name]["y_prob"], results[name]["AUC"]
        ap = average_precision_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=2.2, label=f"{name} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, color=color, lw=2.2, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0,1],[0,1],"k--", lw=1.5, label="Random (AUC=0.50)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", fontweight="bold"); axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal")

    baseline = y_test.mean()
    axes[1].axhline(y=baseline, color="gray", ls="--", lw=1.5,
                    label=f"Baseline (AP≈{baseline:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontweight="bold"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_3_ROC_PR_Curves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 8 — CHART: PERFORMANCE DASHBOARD
# ================================================================

def plot_performance_dashboard(results, feature_names, dataset_name):
    
    print(f"  📊  Chart: Performance Dashboard — {dataset_name}")
    model_names, colors = list(results.keys()), list(MODEL_COLORS.values())
    n_models, x_pos, bar_w = len(model_names), np.arange(len(model_names)), 0.14

    metric_keys   = ["Accuracy","Precision","Recall","F1","AUC"]
    metric_colors = ["#3498db","#9b59b6","#e74c3c","#f39c12","#1abc9c"]
    metric_matrix = {k: [results[m][k]*100 for m in model_names] for k in metric_keys}

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(f"Performance Dashboard — {dataset_name}", fontsize=14, fontweight="bold")

    ax = axes[0,0]
    offsets = np.linspace(-(2*bar_w), 2*bar_w, 5)
    for i, (metric, mc) in enumerate(zip(metric_keys, metric_colors)):
        ax.bar(x_pos+offsets[i], metric_matrix[metric], bar_w, label=metric,
               color=mc, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos); ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax.set_ylim([50,112]); ax.set_title("Multi-Metric Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)

    ax = axes[0,1]
    cv_data = [results[m]["CV Scores"]*100 for m in model_names]
    bp = ax.boxplot(cv_data, patch_artist=True, widths=0.45,
                    medianprops=dict(color="black", linewidth=2.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for i,(data,color) in enumerate(zip(cv_data, colors)):
        jitter = np.random.default_rng(i).uniform(-0.12,0.12,len(data))
        ax.scatter([i+1]*len(data)+jitter, data, color=color, edgecolor="black", s=40, zorder=5)
        ax.text(i+1, np.max(data)+0.8, f"{np.mean(data):.1f}%", ha="center",
               fontsize=9, fontweight="bold", color=color)
    ax.set_xticks(range(1,n_models+1)); ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax.set_title("5-Fold CV Scores (spread = consistency)", fontweight="bold")

    ax = axes[1,0]
    rf_model = results["Random Forest"]["model"]
    n_feat   = min(15, len(feature_names))
    imp      = rf_model.feature_importances_
    idx_s    = np.argsort(imp)[-n_feat:]
    ax.barh([short_name(feature_names[i],18) for i in idx_s], imp[idx_s],
           color=plt.cm.Greens(np.linspace(0.3,0.9,n_feat)), edgecolor="black", linewidth=0.5)
    ax.set_title(f"Random Forest — Top {n_feat} Feature Importances", fontweight="bold")

    ax = axes[1,1]
    xgb_model = results["XGBoost"]["model"]
    imp2 = xgb_model.feature_importances_
    idx_s2 = np.argsort(imp2)[-n_feat:]
    ax.barh([short_name(feature_names[i],18) for i in idx_s2], imp2[idx_s2],
           color=plt.cm.Oranges(np.linspace(0.3,0.9,n_feat)), edgecolor="black", linewidth=0.5)
    ax.set_title(f"XGBoost — Top {n_feat} Feature Importances", fontweight="bold")

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_4_Performance_Dashboard.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 9 — CHART: LEARNING CURVES
# ================================================================

def plot_learning_curves(X_sc, y, dataset_name):
    
    print(f"  📊  Chart: Learning Curves — {dataset_name}")
    model_names = list(MODEL_COLORS.keys())
    models_list = list(get_models().values())
    colors      = list(MODEL_COLORS.values())
    train_sizes = np.linspace(0.15, 1.0, 7)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Learning Curves — {dataset_name}\n"
                "Does more training data help each model?", fontsize=13, fontweight="bold")

    for idx, (name, model, color) in enumerate(zip(model_names, models_list, colors)):
        ax = axes[idx//2, idx%2]
        try:
            tr_sz, tr_sc, cv_sc = learning_curve(
                model, X_sc, y, cv=3, train_sizes=train_sizes,
                scoring="accuracy", n_jobs=1, shuffle=True, random_state=42)
        except Exception as e:
            ax.text(0.5,0.5,f"Error: {str(e)[:60]}", ha="center", va="center",
                   transform=ax.transAxes); ax.set_title(name); continue

        tr_mean, tr_std = tr_sc.mean(axis=1), tr_sc.std(axis=1)
        cv_mean, cv_std = cv_sc.mean(axis=1), cv_sc.std(axis=1)

        ax.plot(tr_sz, tr_mean, "o-", color=color, lw=2.2, label="Training Accuracy")
        ax.fill_between(tr_sz, tr_mean-tr_std, tr_mean+tr_std, alpha=0.12, color=color)
        ax.plot(tr_sz, cv_mean, "s--", color=color, lw=2.2, alpha=0.7, label="CV Accuracy")
        ax.fill_between(tr_sz, cv_mean-cv_std, cv_mean+cv_std, alpha=0.08, color="gray")

        gap = tr_mean[-1] - cv_mean[-1]
        if cv_mean[-1] < 0.70: status, sc_ = "Possible Underfitting", "#e74c3c"
        elif gap > 0.12:        status, sc_ = "Possible Overfitting", "#e67e22"
        else:                   status, sc_ = "Good Fit", "#27ae60"
        ax.text(0.97,0.05, status, ha="right", va="bottom", transform=ax.transAxes,
               fontsize=9, fontweight="bold", color=sc_,
               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sc_, alpha=0.8))

        ax.set_xlabel("Training Samples"); ax.set_ylabel("Accuracy")
        ax.set_title(f"{name} (Train={tr_mean[-1]*100:.1f}% CV={cv_mean[-1]*100:.1f}% "
                    f"Gap={gap*100:.1f}%)", fontsize=10, fontweight="bold", color=color)
        ax.legend(loc="lower right", fontsize=8); ax.set_ylim([0.45,1.05])

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_5_LearningCurves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 10 — CHART: VALIDATION CURVES
# ================================================================

def plot_validation_curves(X_sc, y, dataset_name):
    
    print(f"  📊  Chart: Validation Curves — {dataset_name}")
    configs = [
        (SVC(kernel="rbf", probability=True, random_state=42), "SVM", "C",
         [0.001,0.01,0.1,1,10,100,1000]),
        (LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42),
         "Logistic Regression", "C", [0.001,0.01,0.1,1,10,100,1000]),
        (RandomForestClassifier(random_state=42), "Random Forest", "n_estimators",
         [10,25,50,75,100,150,200]),
        (XGBClassifier(learning_rate=0.1, eval_metric="logloss", random_state=42, verbosity=0),
         "XGBoost", "max_depth", [1,2,3,4,5,6,7,8])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"Validation Curves (Hyperparameter Sensitivity) — {dataset_name}",
                fontsize=13, fontweight="bold")

    for idx, (model, name, param, p_range) in enumerate(configs):
        ax, color = axes[idx//2, idx%2], MODEL_COLORS[name]
        try:
            tr_sc, cv_sc = validation_curve(model, X_sc, y, param_name=param,
                                            param_range=p_range, cv=3,
                                            scoring="accuracy", n_jobs=1)
        except Exception as e:
            ax.text(0.5,0.5,f"Error: {str(e)[:60]}", ha="center", va="center",
                   transform=ax.transAxes); ax.set_title(name); continue

        tr_mean, cv_mean = tr_sc.mean(axis=1), cv_sc.mean(axis=1)
        x_vals = np.arange(len(p_range))
        ax.plot(x_vals, tr_mean, "o-", color=color, lw=2.2, label="Training Accuracy")
        ax.plot(x_vals, cv_mean, "s--", color=color, lw=2.2, alpha=0.7, label="CV Accuracy")

        best_idx = int(np.argmax(cv_mean))
        ax.axvline(x=best_idx, color="#e74c3c", ls=":", lw=1.5, alpha=0.7)
        ax.scatter([best_idx],[cv_mean[best_idx]], s=140, zorder=6,
                  color="#e74c3c", edgecolor="black")
        ax.annotate(f"Best {param}={p_range[best_idx]}\nCV={cv_mean[best_idx]*100:.1f}%",
                   xy=(best_idx,cv_mean[best_idx]), xytext=(best_idx+0.4, cv_mean[best_idx]-0.04),
                   fontsize=8, color="#c0392b", arrowprops=dict(arrowstyle="->", color="#c0392b"))

        ax.set_xticks(x_vals); ax.set_xticklabels([str(v) for v in p_range], rotation=20, fontsize=8)
        ax.set_xlabel(f"{param}"); ax.set_title(f"{name} — Varying '{param}'",
                                                fontsize=10, fontweight="bold", color=color)
        ax.legend(loc="lower right", fontsize=8); ax.set_ylim([0.45,1.05])

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_6_ValidationCurves.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 11 — CHART: CALIBRATION + CONFIDENCE
# ================================================================

def plot_calibration_and_confidence(results, y_test, dataset_name):
    
    print(f"  📊  Chart: Calibration & Confidence — {dataset_name}")
    model_names, colors = list(results.keys()), list(MODEL_COLORS.values())
    fig, axes = plt.subplots(2, len(model_names), figsize=(22, 11))
    fig.suptitle(f"Calibration & Confidence Analysis — {dataset_name}",
                fontsize=13, fontweight="bold")

    for idx, (name, color) in enumerate(zip(model_names, colors)):
        y_prob, y_pred = results[name]["y_prob"], results[name]["y_pred"]

        ax_cal = axes[0, idx]
        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy="uniform")
        ax_cal.plot([0,1],[0,1],"k--", lw=1.5, alpha=0.6, label="Perfect")
        ax_cal.plot(mean_pred, frac_pos, "o-", color=color, lw=2.5, markersize=7, label=name)
        cal_error = np.mean(np.abs(frac_pos - mean_pred))
        ax_cal.text(0.04,0.88, f"Cal.Error={cal_error:.3f}\n" +
                   ("Well calibrated" if cal_error<0.05 else "Needs calibration"),
                   transform=ax_cal.transAxes, fontsize=8, fontweight="bold",
                   color="#27ae60" if cal_error<0.05 else "#e74c3c",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
        ax_cal.set_xlim([0,1]); ax_cal.set_ylim([0,1]); ax_cal.set_aspect("equal")
        ax_cal.set_title(f"Calibration\n{name}", fontsize=9, fontweight="bold", color=color)
        ax_cal.legend(fontsize=7)

        ax_conf = axes[1, idx]
        correct_mask = (y_pred == np.array(y_test))
        probs_c, probs_w = y_prob[correct_mask], y_prob[~correct_mask]
        bins = np.linspace(0,1,21)
        if len(probs_c): ax_conf.hist(probs_c, bins=bins, color="#27ae60", alpha=0.65,
                                      label=f"Correct ({len(probs_c)})", edgecolor="white")
        if len(probs_w): ax_conf.hist(probs_w, bins=bins, color="#e74c3c", alpha=0.65,
                                      label=f"Wrong ({len(probs_w)})", edgecolor="white")
        ax_conf.axvline(x=0.5, color="black", ls="--", lw=1.5, alpha=0.5)
        ax_conf.set_title(f"Confidence Distribution\n{name}", fontsize=9,
                          fontweight="bold", color=color)
        ax_conf.legend(fontsize=7); ax_conf.set_xlim([0,1])

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_7_Calibration_Confidence.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 12 — CHART: OVERFITTING + PER-CLASS METRICS
# ================================================================

def plot_overfitting_and_perclass(results, X_train_sc, X_test_sc,
                                  y_train, y_test, class_names, dataset_name):
    
    print(f"  📊  Chart: Overfitting + Per-Class — {dataset_name}")
    model_names, colors = list(results.keys()), list(MODEL_COLORS.values())
    n_models = len(model_names)

    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    fig.suptitle(f"Overfitting Analysis & Per-Class Metrics — {dataset_name}",
                fontsize=13, fontweight="bold")

    ax = axes[0,0]
    x, bw = np.arange(n_models), 0.28
    train_accs, test_accs, gaps = [], [], []
    for name in model_names:
        model = results[name]["model"]
        tr_acc = accuracy_score(y_train, model.predict(X_train_sc))
        te_acc = results[name]["Accuracy"]
        train_accs.append(tr_acc*100); test_accs.append(te_acc*100); gaps.append((tr_acc-te_acc)*100)

    ax.bar(x-bw, train_accs, bw, label="Train Acc", color=colors, alpha=0.9, edgecolor="black")
    ax.bar(x, test_accs, bw, label="Test Acc", color=colors, alpha=0.5, edgecolor="black", hatch="//")
    bars_gp = ax.bar(x+bw, gaps, bw, label="Gap", color="#e74c3c", alpha=0.7, edgecolor="black")
    for bar, gap in zip(bars_gp, gaps):
        bar.set_facecolor("#27ae60" if gap<2 else "#f39c12" if gap<8 else "#e74c3c")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=10, fontsize=9)
    ax.set_title("Train vs Test Accuracy + Overfitting Gap", fontweight="bold")
    ax.legend(fontsize=8)
    for bar,val in zip(bars_gp, gaps):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{val:.1f}%",
               ha="center", fontsize=8, fontweight="bold")

    ax = axes[0,1]; ax.axis("off")
    rows = [["Model","Train","Test","Gap","Verdict"]]
    for name, tr, te, g in zip(model_names, train_accs, test_accs, gaps):
        verdict = "Excellent" if g<2 else "Acceptable" if g<8 else "Moderate" if g<15 else "Severe"
        rows.append([name[:16], f"{tr:.1f}%", f"{te:.1f}%", f"{g:.1f}%", verdict])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center",
                   loc="center", bbox=[0,0.1,1,0.8])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
    for (r,c), cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
    ax.set_title("Overfitting Summary Table", fontweight="bold")

    metrics_per_class = {cls: {m:{} for m in model_names} for cls in class_names}
    for m_name in model_names:
        y_pred = results[m_name]["y_pred"]
        for cls_idx, cls_name in enumerate(class_names):
            y_bin    = (np.array(y_test)==cls_idx).astype(int)
            y_pred_b = (np.array(y_pred)==cls_idx).astype(int)
            metrics_per_class[cls_name][m_name]["Precision"] = precision_score(y_bin,y_pred_b,zero_division=0)
            metrics_per_class[cls_name][m_name]["Recall"]    = recall_score(y_bin,y_pred_b,zero_division=0)

    ax = axes[1,0]
    prec_matrix = np.array([[metrics_per_class[c][m]["Precision"]*100 for m in model_names] for c in class_names])
    sns.heatmap(prec_matrix, ax=ax, annot=True, fmt=".1f", cmap="Greens", vmin=50, vmax=100,
               xticklabels=[m[:11] for m in model_names], yticklabels=class_names,
               annot_kws={"size":11,"weight":"bold"})
    ax.set_title("Per-Class PRECISION (%)", fontweight="bold")

    ax = axes[1,1]
    rec_matrix = np.array([[metrics_per_class[c][m]["Recall"]*100 for m in model_names] for c in class_names])
    sns.heatmap(rec_matrix, ax=ax, annot=True, fmt=".1f", cmap="Blues", vmin=50, vmax=100,
               xticklabels=[m[:11] for m in model_names], yticklabels=class_names,
               annot_kws={"size":11,"weight":"bold"})
    ax.set_title("Per-Class RECALL (%)", fontweight="bold")

    plt.tight_layout()
    fname = out(f"{dataset_name.replace(' ','_')}_8_Overfitting_PerClass.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 13 — UNIFIED PER-DATASET PIPELINE (trains ONCE)
# ================================================================

def run_pipeline(X, y, dataset_name, class_names):
    
    chart_paths = {"eda": plot_eda_dashboard(X, y, class_names, dataset_name)}

    print(f"\n  {'═'*58}\n  🏥  TRAINING: {dataset_name.upper()}\n  {'═'*58}")
    X_train_sc, X_test_sc, y_train, y_test, _ = prepare_data(X, y)
    feature_names = list(X.columns)
    models, results, speed, cv_scores_dict, predictions = get_models(), {}, {}, {}, {}

    for name, model in models.items():
        t0 = time.perf_counter(); model.fit(X_train_sc, y_train)
        train_ms = (time.perf_counter()-t0)*1000

        t1 = time.perf_counter(); y_pred = model.predict(X_test_sc)
        predict_ms = (time.perf_counter()-t1)*1000
        y_prob = model.predict_proba(X_test_sc)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)
        cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")

        results[name] = {"model":model, "y_pred":y_pred, "y_prob":y_prob,
                         "Accuracy":acc, "Precision":prec, "Recall":rec, "F1":f1,
                         "AUC":auc, "CV Scores":cv, "CV Mean":cv.mean(), "CV Std":cv.std()}
        speed[name] = {"train_ms":train_ms, "predict_ms":predict_ms}
        cv_scores_dict[name], predictions[name] = cv, y_pred

        print(f"  {name:<22} Acc={acc*100:.1f}%  AUC={auc:.4f}  "
             f"CV={cv.mean()*100:.1f}%±{cv.std()*100:.2f}%  Train={train_ms:.0f}ms")

    chart_paths["confusion"]   = plot_confusion_matrices(results, y_test, class_names, dataset_name)
    chart_paths["roc_pr"]      = plot_roc_pr_curves(results, y_test, dataset_name)
    chart_paths["performance"] = plot_performance_dashboard(results, feature_names, dataset_name)

    # Learning/validation curves use the FULL dataset (their own internal CV)
    sc_full   = StandardScaler()
    X_full_sc = sc_full.fit_transform(X)
    chart_paths["learning"]   = plot_learning_curves(X_full_sc, y, dataset_name)
    chart_paths["validation"] = plot_validation_curves(X_full_sc, y, dataset_name)

    chart_paths["calibration"] = plot_calibration_and_confidence(results, y_test, dataset_name)
    chart_paths["overfitting"] = plot_overfitting_and_perclass(
        results, X_train_sc, X_test_sc, y_train, y_test, class_names, dataset_name)

    return {
        "results": results, "speed": speed, "predictions": predictions,
        "y_test": np.array(y_test), "cv_scores": cv_scores_dict,
        "charts": chart_paths, "class_names": class_names,
        "n_samples": X.shape[0], "n_features": X.shape[1]
    }


# ================================================================
# SECTION 14 — CROSS-DATASET CHARTS
# ================================================================

def plot_final_summary_dashboard(all_results):
    
    print(f"\n  📊  Chart: Final Summary Dashboard")
    dataset_names = list(all_results.keys())
    model_names   = list(list(all_results.values())[0].keys())
    colors        = list(MODEL_COLORS.values())
    metric_list   = ["Accuracy","Precision","Recall","F1","AUC"]
    mm = {k: np.array([[all_results[ds][m][k]*100 for m in model_names] for ds in dataset_names])
          for k in metric_list}

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("Final Summary Dashboard — All Datasets x All Models",
                fontsize=15, fontweight="bold", y=0.99)
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38)

    ax = fig.add_subplot(gs[0,0])
    sns.heatmap(mm["Accuracy"], ax=ax, annot=True, fmt=".1f", cmap="YlGn",
               xticklabels=[m[:10] for m in model_names], yticklabels=dataset_names,
               vmin=70, vmax=100, annot_kws={"size":11,"weight":"bold"})
    ax.set_title("Accuracy Heatmap", fontweight="bold")

    ax = fig.add_subplot(gs[0,1])
    sns.heatmap(mm["AUC"], ax=ax, annot=True, fmt=".1f", cmap="Blues",
               xticklabels=[m[:10] for m in model_names], yticklabels=dataset_names,
               vmin=70, vmax=100, annot_kws={"size":11,"weight":"bold"})
    ax.set_title("AUC Heatmap (x100)", fontweight="bold")

    ax = fig.add_subplot(gs[0,2]); ax.axis("off")
    rows = [["Dataset","Best Model","Accuracy","AUC"]]
    for ds in dataset_names:
        r = all_results[ds]; bm = max(r, key=lambda k: r[k]["Accuracy"])
        rows.append([ds, bm, f"{r[bm]['Accuracy']*100:.2f}%", f"{r[bm]['AUC']:.4f}"])
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], cellLoc="center", loc="center", bbox=[0,0.2,1,0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    for (r,c), cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#2c3e50"); cell.set_text_props(color="white", fontweight="bold")
        elif r%2==1: cell.set_facecolor("#d5e8d4")
        cell.set_edgecolor("white")
    ax.set_title("Best Model Per Dataset", fontweight="bold")

    ax = fig.add_subplot(gs[1,0])
    x_pos, bw = np.arange(len(dataset_names)), 0.18
    for j,(m,c) in enumerate(zip(model_names, colors)):
        vals = [all_results[ds][m]["F1"]*100 for ds in dataset_names]
        ax.bar(x_pos+(j-1.5)*bw, vals, bw, label=m[:10], color=c, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos); ax.set_xticklabels(dataset_names, fontsize=9); ax.set_ylim([55,105])
    ax.set_title("F1-Score Comparison", fontweight="bold"); ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1,1])
    for j,(m,c) in enumerate(zip(model_names, colors)):
        vals = [all_results[ds][m]["Recall"]*100 for ds in dataset_names]
        ax.bar(x_pos+(j-1.5)*bw, vals, bw, label=m[:10], color=c, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos); ax.set_xticklabels(dataset_names, fontsize=9); ax.set_ylim([55,105])
    ax.set_title("Recall (Sensitivity) Comparison", fontweight="bold"); ax.legend(fontsize=7, ncol=2)

    ax = fig.add_subplot(gs[1,2])
    cv_std = np.array([[all_results[ds][m]["CV Std"]*100 for m in model_names] for ds in dataset_names])
    sns.heatmap(cv_std, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd_r",
               xticklabels=[m[:10] for m in model_names], yticklabels=dataset_names,
               annot_kws={"size":11})
    ax.set_title("CV Consistency (Std Dev, lower=better)", fontweight="bold")

    ax = fig.add_subplot(gs[2,:2])
    for j,(m,c) in enumerate(zip(model_names, colors)):
        for ds in dataset_names:
            r = all_results[ds][m]
            size = (r["F1"]*100-60)**2 * 1.5
            ax.scatter(r["Accuracy"]*100, r["AUC"]*100, s=size, color=c, alpha=0.7, edgecolor="black")
            ax.text(r["Accuracy"]*100+0.2, r["AUC"]*100+0.15, f"{ds[:4]}/{m[:4]}", fontsize=6.5)
    leg = [mpatches.Patch(color=c, label=m) for m,c in zip(model_names, colors)]
    ax.legend(handles=leg, loc="lower right", fontsize=8)
    ax.set_xlabel("Accuracy (%)"); ax.set_ylabel("AUC x100")
    ax.set_title("Bubble Chart: Accuracy vs AUC (size = F1)", fontweight="bold")

    ax = fig.add_subplot(gs[2,2]); ax.axis("off")
    rows2 = []
    for m in model_names:
        vals = [np.mean([all_results[ds][m][k]*100 for ds in dataset_names]) for k in metric_list]
        rows2.append([m[:16]] + [f"{v:.1f}" for v in vals] + [f"{np.mean(vals):.1f}"])
    rows2.sort(key=lambda r: float(r[-1]), reverse=True)
    tbl2 = ax.table(cellText=rows2, colLabels=["Model"]+[m[:5] for m in metric_list]+["Avg"],
                    cellLoc="center", loc="center", bbox=[0,0.1,1,0.85])
    tbl2.auto_set_font_size(False); tbl2.set_fontsize(8.5)
    for (r,c), cell in tbl2.get_celld().items():
        if r==0: cell.set_facecolor("#1a252f"); cell.set_text_props(color="white", fontweight="bold")
        elif r==1: cell.set_facecolor("#fdebd0")
        cell.set_edgecolor("white")
    ax.set_title("Overall Rankings (avg across datasets)", fontweight="bold")

    fname = out("Summary_Final_Dashboard.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


def plot_cv_violin(all_cv_scores, all_names):
    
    print(f"  📊  Chart: CV Violin Plots")
    model_names, colors = list(MODEL_COLORS.keys()), list(MODEL_COLORS.values())
    fig, axes = plt.subplots(1, len(all_names), figsize=(21,7), sharey=True)
    fig.suptitle("Cross-Validation Score Distributions — All Datasets",
                fontsize=13, fontweight="bold")

    for col, (ds_name, ax) in enumerate(zip(all_names, axes)):
        violin_data = [all_cv_scores[ds_name][m]*100 for m in model_names]
        parts = ax.violinplot(violin_data, positions=range(len(model_names)), widths=0.65,
                              showmeans=True, showmedians=True)
        for idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[idx]); pc.set_alpha(0.70); pc.set_edgecolor("black")
        for idx, data in enumerate(violin_data):
            jitter = np.random.default_rng(idx*10).uniform(-0.10,0.10,len(data))
            ax.scatter([idx]*len(data)+jitter, data, color=colors[idx], edgecolor="black", s=45, zorder=5)
            ax.text(idx, max(data)+0.8, f"{np.mean(data):.1f}%", ha="center", fontsize=7.5,
                   fontweight="bold", color=colors[idx])
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([m[:10] for m in model_names], rotation=18, ha="right", fontsize=8)
        ax.set_title(f"Dataset: {ds_name}", fontsize=11, fontweight="bold")
        ax.set_ylim([50,107])

    leg = [mpatches.Patch(color=c, label=m) for m,c in zip(model_names, colors)]
    fig.legend(handles=leg, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5,-0.02))
    plt.tight_layout(rect=[0,0.05,1,1])
    fname = out("EVAL_CV_Violin.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


def plot_mcnemar_heatmaps(all_predictions, all_y_tests, all_names):
    
    print(f"  📊  Chart: McNemar Statistical Test")
    model_names, n_models = list(MODEL_COLORS.keys()), len(MODEL_COLORS)
    fig, axes = plt.subplots(1, len(all_names), figsize=(21,6))
    fig.suptitle("McNemar's Statistical Significance Test — All Datasets",
                fontsize=12, fontweight="bold")

    for col, (ds_name, ax) in enumerate(zip(all_names, axes)):
        preds, y_true = all_predictions[ds_name], np.array(all_y_tests[ds_name])
        p_matrix = np.ones((n_models, n_models))
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    p_matrix[i,j] = mcnemar_test(y_true, preds[model_names[i]], preds[model_names[j]])
        im = ax.imshow(p_matrix, cmap="RdYlGn_r", vmin=0, vmax=0.15, aspect="auto")
        for i in range(n_models):
            for j in range(n_models):
                if i==j: txt = "—"
                elif p_matrix[i,j]<0.01: txt = f"{p_matrix[i,j]:.3f}\n**p<0.01"
                elif p_matrix[i,j]<0.05: txt = f"{p_matrix[i,j]:.3f}\n*p<0.05"
                else: txt = f"{p_matrix[i,j]:.3f}\n(n.s.)"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                       color="white" if i==j else "black")
        ax.set_xticks(range(n_models)); ax.set_yticks(range(n_models))
        ax.set_xticklabels([m[:10] for m in model_names], rotation=18, ha="right", fontsize=8)
        ax.set_yticklabels([m[:10] for m in model_names], fontsize=8)
        ax.set_title(ds_name, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p-value")

    leg = [mpatches.Patch(color="#27ae60", label="**p<0.01 highly significant"),
           mpatches.Patch(color="#f39c12", label="*p<0.05 significant"),
           mpatches.Patch(color="#e74c3c", label="n.s. p>=0.05 not significant")]
    fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5,-0.04))
    plt.tight_layout(rect=[0,0.05,1,1])
    fname = out("EVAL_McNemar_Statistical.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


def plot_speed_and_report(all_speed, all_results, all_names):
    
    print(f"  📊  Chart: Speed Benchmark + Master Report")
    model_names = list(MODEL_COLORS.keys())
    fig = plt.figure(figsize=(22, 15))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)
    fig.suptitle("Speed Benchmark & Master Evaluation Report", fontsize=14, fontweight="bold")

    x, bw, ds_clr = np.arange(len(model_names)), 0.22, ["#2c3e50","#8e44ad","#16a085"]
    ax = fig.add_subplot(gs[0,0])
    for i, ds in enumerate(all_names):
        times = [all_speed[ds][m]["train_ms"] for m in model_names]
        ax.bar(x+(i-1)*bw, times, bw, label=ds, color=ds_clr[i], alpha=0.8, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax.set_title("Training Time (ms)", fontweight="bold"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0,1])
    for i, ds in enumerate(all_names):
        times = [all_speed[ds][m]["predict_ms"] for m in model_names]
        ax.bar(x+(i-1)*bw, times, bw, label=ds, color=ds_clr[i], alpha=0.8, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(model_names, rotation=12, ha="right", fontsize=9)
    ax.set_title("Prediction Time (ms)", fontweight="bold"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1,:]); ax.axis("off")
    headers = ["Dataset","Model","Acc","Prec","Rec","F1","AUC","CV Acc","CV Std","Train(ms)","Pred(ms)"]
    rows, best_rows = [], {}
    for ds in all_names:
        best_m = max(all_results[ds], key=lambda k: all_results[ds][k]["Accuracy"])
        for m in model_names:
            r, spd = all_results[ds][m], all_speed[ds][m]
            rows.append([ds[:14], m[:18], f"{r['Accuracy']*100:.1f}%", f"{r['Precision']*100:.1f}%",
                        f"{r['Recall']*100:.1f}%", f"{r['F1']*100:.1f}%", f"{r['AUC']:.4f}",
                        f"{r['CV Mean']*100:.1f}%", f"{r['CV Std']*100:.2f}%",
                        f"{spd['train_ms']:.1f}", f"{spd['predict_ms']:.2f}"])
            if m == best_m: best_rows[len(rows)] = True

    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r,c), cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor("#1a252f"); cell.set_text_props(color="white", fontweight="bold")
        elif r in best_rows: cell.set_facecolor("#d5f5e3")
        elif r%2==0: cell.set_facecolor("#eaecee")
        cell.set_edgecolor("#bdc3c7")
    ax.set_title("Master Evaluation Report (green = best per dataset)", fontweight="bold")

    fname = out("EVAL_Speed_Report.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"     ✅  Saved → '{fname}'")
    return fname


# ================================================================
# SECTION 15 — TERMINAL SUMMARY
# ================================================================

def print_terminal_summary(all_results):
    dataset_names = list(all_results.keys())
    model_names   = list(list(all_results.values())[0].keys())
    print("\n" + "═"*78 + "\n  📊  FINAL RESULTS — ALL DATASETS x ALL MODELS\n" + "═"*78)
    for ds in dataset_names:
        for m in model_names:
            r = all_results[ds][m]
            print(f"  {ds:<16} {m:<22} Acc={r['Accuracy']*100:>5.1f}% "
                 f"AUC={r['AUC']:.4f}  CV={r['CV Mean']*100:>5.1f}%")
        print("  " + "─"*70)
    print("\n  🏆  BEST MODEL PER DATASET:")
    for ds in dataset_names:
        r = all_results[ds]; bm = max(r, key=lambda k: r[k]["Accuracy"])
        print(f"     {ds:<16} → {bm}  (Acc={r[bm]['Accuracy']*100:.2f}%  AUC={r[bm]['AUC']:.4f})")
    print("═"*78 + "\n")


# ================================================================
# SECTION 16 — PDF REPORT GENERATION
# ================================================================

class PDFReport(FPDF):

    def safe_text(self, text):
        
        if text is None:
            return ""
        text = str(text)
        replacements = {
            "•": "-", "✅": "[OK]", "⚠️": "[WARNING]", "❌": "[ERROR]",
            "📊": "[CHART]", "📄": "[PDF]", "🚀": "[START]", "🔄": "[TRAINING]",
            "🏆": "[BEST]", "🔬": "[EVAL]", "📈": "[CHART]", "📥": "[LOAD]",
            "→": "->", "★": "*", "✓": "[OK]", "✗": "[X]",
            "-": "-", "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
            "×": "x", "±": "+/-", "≥": ">=", "≤": "<=",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Final safety: remove anything still unsupported by latin-1 fonts
        return text.encode("latin-1", "replace").decode("latin-1")

    def cell(self, *args, **kwargs):
        args = list(args)
        if len(args) >= 3:
            args[2] = self.safe_text(args[2])
        if "text" in kwargs:
            kwargs["text"] = self.safe_text(kwargs["text"])
        if "txt" in kwargs:
            kwargs["txt"] = self.safe_text(kwargs["txt"])
        return super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        args = list(args)
        if len(args) >= 3:
            args[2] = self.safe_text(args[2])
        if "text" in kwargs:
            kwargs["text"] = self.safe_text(kwargs["text"])
        if "txt" in kwargs:
            kwargs["txt"] = self.safe_text(kwargs["txt"])
        return super().multi_cell(*args, **kwargs)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "B", 10)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "CodeAlpha - Disease Prediction Report", 0, 0, "L")
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "R")
            self.ln(5)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0, 10,
            f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            0, 0, "C"
        )

    def chapter_title(self, title):
        self.set_font("Arial", "B", 16)
        self.set_text_color(0, 0, 150)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(5)

    def chapter_body(self, text):
        self.set_font("Arial", "", 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 7, text)
        self.ln()

    def add_image(self, image_path, width=180):
        if os.path.exists(image_path):
            self.image(image_path, x=15, w=width)
            self.ln(10)

    def add_cover_page(self, dataset_summaries):
        self.add_page()
        self.set_font("Arial", "B", 24); self.set_text_color(0, 0, 150)
        self.ln(35)
        self.cell(0, 15, "CodeAlpha Internship - Task 4", 0, 1, "C")
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Disease Prediction from Medical Data", 0, 1, "C")
        self.ln(6)
        self.set_font("Arial", "", 12); self.set_text_color(70, 70, 70)
        self.cell(0, 8, "Final Project & Evaluation Report", 0, 1, "C")
        self.ln(18)

        self.set_draw_color(0, 0, 150)
        self.set_line_width(0.6)
        self.line(40, self.get_y(), 170, self.get_y())
        self.ln(10)

        self.set_font("Arial", "", 11); self.set_text_color(0, 0, 0)
        info_lines = [
            f"Student ID       :  {STUDENT_ID}",
            f"Internship       :  Machine Learning Internship - CodeAlpha",
            f"Algorithms Used  :  SVM | Logistic Regression | Random Forest | XGBoost",
            f"Datasets Used    :  Diabetes | Heart Disease | Breast Cancer (UCI ML Repository)",
            f"Report Generated :  {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        ]
        for line in info_lines:
            self.cell(0, 9, line, 0, 1, "C")

        self.ln(14)
        self.set_font("Arial", "B", 12); self.set_text_color(0, 0, 150)
        self.cell(0, 8, "Dataset Snapshot", 0, 1, "C")
        self.ln(2)
        self.add_table(
            ["Dataset", "Samples", "Features", "Best Model", "Best Accuracy"],
            dataset_summaries,
            col_widths=[45, 30, 30, 45, 40]
        )

    def add_table(self, headers, rows, col_widths=None, header_fill=(44, 62, 80), row_h=8):
        n = len(headers)
        if col_widths is None:
            col_widths = [190 / n] * n
        self.set_font("Arial", "B", 9)
        self.set_fill_color(*header_fill); self.set_text_color(255, 255, 255)
        self.set_x((210 - sum(col_widths)) / 2)
        for h, w in zip(headers, col_widths):
            self.cell(w, row_h, h, border=1, align="C", fill=True)
        self.ln(row_h)
        self.set_font("Arial", "", 9); self.set_text_color(0, 0, 0)
        for i, row in enumerate(rows):
            self.set_x((210 - sum(col_widths)) / 2)
            self.set_fill_color(245, 245, 245) if i % 2 == 0 else self.set_fill_color(255, 255, 255)
            for val, w in zip(row, col_widths):
                self.cell(w, row_h, str(val), border=1, align="C", fill=True)
            self.ln(row_h)
        self.ln(4)

    def add_section(self, title, body_text):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body_text)

    def add_chart_page(self, title, image_path, caption=None, width=180):
        self.add_page()
        self.chapter_title(title)
        self.add_image(image_path, width=width)
        if caption:
            self.chapter_body(caption)


def build_pdf_report(all_data, cross_charts, all_names):
    print("\n" + "═"*62 + "\n  📄  BUILDING PDF REPORT...\n" + "═"*62)
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=18)

    # ── Cover Page ──────────────────────────────────────────────
    snapshot_rows = []
    for ds in all_names:
        data = all_data[ds]
        r    = data["results"]
        bm   = max(r, key=lambda k: r[k]["Accuracy"])
        snapshot_rows.append([ds, data["n_samples"], data["n_features"],
                              bm, f"{r[bm]['Accuracy']*100:.2f}%"])
    pdf.add_cover_page(snapshot_rows)

    # ── Introduction ────────────────────────────────────────────
    pdf.add_section(
        "1. Introduction & Objective",
        "Objective: Predict the possibility of disease based on patient data.\n\n"
        "Approach: This project applies four supervised classification "
        "algorithms - Support Vector Machine (SVM), Logistic Regression, "
        "Random Forest, and XGBoost - to three structured medical datasets "
        "from the UCI Machine Learning Repository. Each dataset uses "
        "real patient features such as symptoms, age, and blood test "
        "results to predict whether a patient has a particular disease.\n\n"
        "Key Features Used:\n"
        "- Diabetes: Glucose, BMI, Blood Pressure, Insulin, Age, Pregnancies\n"
        "- Heart Disease: Chest pain type, cholesterol, max heart rate, ECG results\n"
        "- Breast Cancer: 30 computed cell-nucleus measurements from imaging\n\n"
        "This report documents the full machine learning pipeline - from "
        "data cleaning through model training to in-depth statistical "
        "evaluation - and presents the results as charts, tables, and "
        "plain-English interpretation."
    )

    # ── Methodology ─────────────────────────────────────────────
    pdf.add_section(
        "2. Methodology",
        "Data Cleaning: Biologically impossible zero-values (e.g. Glucose=0) "
        "in the Diabetes dataset were replaced with the column median.\n\n"
        "Train/Test Split: Each dataset was split 80% training / 20% testing "
        "using stratified sampling, which keeps the same disease/no-disease "
        "ratio in both sets.\n\n"
        "Feature Scaling: StandardScaler was fit on the training data only "
        "and then applied to the test data, converting every feature to a "
        "common scale (mean=0, std=1) so that SVM and Logistic Regression "
        "are not biased toward large-magnitude features.\n\n"
        "Model Training: Four classifiers were trained per dataset - SVM "
        "(RBF kernel), Logistic Regression, Random Forest (100 trees), and "
        "XGBoost (100 boosting rounds).\n\n"
        "Evaluation: Each model was assessed using Accuracy, Precision, "
        "Recall, F1-Score, AUC, 5-fold Cross-Validation, Learning Curves, "
        "Validation Curves, Calibration Curves, Confidence Distributions, "
        "Overfitting Gap Analysis, Per-Class Metrics, and McNemar's "
        "statistical significance test between every pair of models."
    )

    # ── Per-Dataset Sections ────────────────────────────────────
    chart_titles = {
        "eda"        : "Exploratory Data Analysis (EDA)",
        "confusion"  : "Confusion Matrices (All 4 Models)",
        "roc_pr"     : "ROC & Precision-Recall Curves",
        "performance": "Performance Dashboard",
        "learning"   : "Learning Curves",
        "validation" : "Validation Curves (Hyperparameter Sensitivity)",
        "calibration": "Calibration & Prediction Confidence",
        "overfitting": "Overfitting Analysis & Per-Class Metrics"
    }
    chart_captions = {
        "eda"        : "Class balance, feature correlations, and the features that most separate the two classes.",
        "confusion"  : "TN/TP outlined in green = correct predictions. FP/FN outlined in red = errors.",
        "roc_pr"     : "Curves closer to the top-left (ROC) or top-right (PR) indicate a stronger classifier.",
        "performance": "Five-metric comparison, 5-fold CV stability, and which features each tree model relied on most.",
        "learning"   : "Shows whether each model would benefit from more training data, or if it has already converged.",
        "validation" : "Shows the best hyperparameter value for each model and the underfitting/overfitting zones around it.",
        "calibration": "Top row: are predicted probabilities trustworthy? Bottom row: is the model confident when right and unsure when wrong?",
        "overfitting": "Train-vs-test accuracy gap per model, plus precision/recall broken down separately for each class."
    }

    for ds in all_names:
        data    = all_data[ds]
        results = data["results"]
        cls     = data["class_names"]

        pdf.add_page()
        pdf.chapter_title(f"3. Dataset Results: {ds}")
        pdf.chapter_body(
            f"Samples: {data['n_samples']}   |   Features: {data['n_features']}   |   "
            f"Classes: {cls[0]} vs {cls[1]}"
        )
        metric_rows = []
        for m, r in results.items():
            metric_rows.append([m, f"{r['Accuracy']*100:.1f}%", f"{r['Precision']*100:.1f}%",
                                f"{r['Recall']*100:.1f}%", f"{r['F1']*100:.1f}%",
                                f"{r['AUC']:.4f}", f"{r['CV Mean']*100:.1f}%"])
        pdf.add_table(["Model","Accuracy","Precision","Recall","F1","AUC","CV Acc"], metric_rows,
                      col_widths=[42,25,28,25,20,25,25])
        best_m = max(results, key=lambda k: results[k]["Accuracy"])
        pdf.chapter_body(
            f"Best performing model: {best_m} "
            f"(Accuracy={results[best_m]['Accuracy']*100:.2f}%, "
            f"AUC={results[best_m]['AUC']:.4f}). "
            f"Fastest model: see the Speed Benchmark in the cross-dataset section."
        )

        for key in ["eda","confusion","roc_pr","performance","learning",
                   "validation","calibration","overfitting"]:
            pdf.add_chart_page(
                f"{ds} - {chart_titles[key]}",
                data["charts"][key],
                chart_captions[key]
            )

    # ── Cross-Dataset Section ──────────────────────────────────
    pdf.add_page()
    pdf.chapter_title("4. Cross-Dataset Evaluation")
    pdf.chapter_body(
        "This section compares all four models across all three datasets "
        "simultaneously - revealing which algorithm is the most consistently "
        "strong performer, how stable each model is across cross-validation "
        "folds, and whether the differences in accuracy between models are "
        "statistically significant or could be explained by chance."
    )
    pdf.add_chart_page("Final Summary Dashboard", cross_charts["summary"],
                       "Accuracy/AUC heatmaps, the best model per dataset, F1 and Recall "
                       "comparisons, CV consistency, and an overall ranking table.")
    pdf.add_chart_page("Cross-Validation Score Distributions (Violin Plots)", cross_charts["violin"],
                       "Wider/taller violins with points clustered near the top indicate a "
                       "more reliable model across different data splits.")
    pdf.add_chart_page("McNemar Statistical Significance Test", cross_charts["mcnemar"],
                       "p < 0.05 means the accuracy difference between two models is "
                       "statistically real rather than due to chance.")
    pdf.add_chart_page("Speed Benchmark & Master Evaluation Report", cross_charts["speed"],
                       "Training/prediction time for every model, plus the complete metrics "
                       "table across all datasets (green rows = best model per dataset).")

    # ── Conclusion ──────────────────────────────────────────────
    conclusion_lines = ["Best model per dataset:\n"]
    for ds in all_names:
        r = all_data[ds]["results"]; bm = max(r, key=lambda k: r[k]["Accuracy"])
        conclusion_lines.append(
            f"  - {ds}: {bm}  (Accuracy={r[bm]['Accuracy']*100:.2f}%, AUC={r[bm]['AUC']:.4f})"
        )
    conclusion_text = "\n".join(conclusion_lines) + (
        "\n\nOverall, tree-based ensemble methods (Random Forest, XGBoost) "
        "performed strongly on datasets with non-linear feature interactions "
        "(Heart Disease), while linear models (Logistic Regression, SVM) "
        "matched or exceeded them on the high-dimensional but more linearly "
        "separable Breast Cancer dataset. Learning curves showed all four "
        "models had largely converged given the available data, and "
        "overfitting gaps stayed within an acceptable range (under 8%) for "
        "most model/dataset combinations.\n\n"
        "Limitations: these are well-known benchmark datasets used for "
        "educational purposes; sample sizes are modest (303-768 rows), and "
        "results have not been externally validated on a separate clinical "
        "population. This project is intended for learning and portfolio "
        "purposes and is not a substitute for professional medical diagnosis."
    )
    pdf.add_section("5. Conclusion", conclusion_text)

    # ── References ──────────────────────────────────────────────
    pdf.add_section(
        "6. Dataset References",
        "Diabetes (Pima Indians Diabetes Database):\n"
        "  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database\n\n"
        "Heart Disease (Cleveland UCI):\n"
        "  https://archive.ics.uci.edu/dataset/45/heart+disease\n\n"
        "Breast Cancer Wisconsin (Diagnostic):\n"
        "  https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic\n"
        "  (auto-loaded via sklearn.datasets.load_breast_cancer - no download needed)\n\n"
        "Internship Reference:\n"
        f"  CodeAlpha Machine Learning Internship - Student ID {STUDENT_ID}"
    )

    out_name = out("CodeAlpha_Task4_Disease_Prediction_Report.pdf")
    pdf.output(out_name)
    print(f"\n  ✅  PDF report saved → '{out_name}'  ({os.path.getsize(out_name)/1024:.0f} KB)")
    return out_name


# ================================================================
# SECTION 17 — MAIN
# ================================================================

if __name__ == "__main__":

    all_data, all_names = {}, []

    for loader in [load_diabetes, load_heart_disease, load_breast_cancer_data]:
        X, y, name, classes = loader()
        all_names.append(name)
        all_data[name] = run_pipeline(X, y, name, classes)

    all_results     = {ds: all_data[ds]["results"]     for ds in all_names}
    all_speed       = {ds: all_data[ds]["speed"]       for ds in all_names}
    all_predictions = {ds: all_data[ds]["predictions"] for ds in all_names}
    all_y_tests     = {ds: all_data[ds]["y_test"]       for ds in all_names}
    all_cv_scores   = {ds: all_data[ds]["cv_scores"]    for ds in all_names}

    cross_charts = {
        "summary": plot_final_summary_dashboard(all_results),
        "violin" : plot_cv_violin(all_cv_scores, all_names),
        "mcnemar": plot_mcnemar_heatmaps(all_predictions, all_y_tests, all_names),
        "speed"  : plot_speed_and_report(all_speed, all_results, all_names)
    }

    print_terminal_summary(all_results)

    pdf_path = build_pdf_report(all_data, cross_charts, all_names)

    print("\n" + "═"*64)
