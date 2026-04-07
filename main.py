import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score


# Set seed so results stay the same each time
SEED = 42
np.random.seed(SEED)


# Load the dataset

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 1 = benign, 0 = malignant

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)


# Train the model
# Random Forest works well for this dataset and can also be explained with SHAP
clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)

print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")


# Compute predictive entropy

def entropy(probabilities):
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    return -np.sum(probabilities * np.log(probabilities), axis=1)


H = entropy(proba)

# Split samples into most confident and least confident groups
high_conf_mask = H <= np.percentile(H, 25)
low_conf_mask = H >= np.percentile(H, 75)

print(f"High-confidence samples: {high_conf_mask.sum()}")
print(f"Low-confidence samples: {low_conf_mask.sum()}")

# Compute calibration (ECE)
# ECE checks whether predicted probabilities match actual correctness

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])

        if np.sum(mask) == 0:
            continue

        accuracy_in_bin = (y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean()
        confidence_in_bin = y_prob[mask].mean()

        ece += (np.sum(mask) / len(y_true)) * abs(accuracy_in_bin - confidence_in_bin)

    return ece


# Use probability of class 1 (benign), since y=1 means benign
ece = compute_ece(y_test, proba[:, 1])
print(f"ECE: {ece:.4f}")


# SHAP explanations

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Depending on SHAP version, output can be a list or a 3D array
if isinstance(shap_values, list):
    sv_malignant = shap_values[0]
else:
    sv_malignant = shap_values[:, :, 0]

# Compare SHAP importance for confident vs uncertain predictions
sv_high = sv_malignant[high_conf_mask]
sv_low = sv_malignant[low_conf_mask]

mean_abs_high = np.abs(sv_high).mean(axis=0)
mean_abs_low = np.abs(sv_low).mean(axis=0)

# Plot settings

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150
})

# Entropy distribution

fig1, ax = plt.subplots(figsize=(6, 3.5))

bins = np.linspace(0, H.max() + 0.05, 20)

ax.hist(H[high_conf_mask], bins=bins, alpha=0.7, color="#2196F3", label="High confidence")
ax.hist(H[low_conf_mask], bins=bins, alpha=0.7, color="#F44336", label="Low confidence")

ax.set_xlabel("Predictive Entropy")
ax.set_ylabel("Count")
ax.set_title("Distribution of Predictive Entropy")
ax.set_xlim(left=0)
ax.legend()

fig1.tight_layout()
fig1.savefig("entropy_distribution.png")
plt.close(fig1)


# Reliability diagram

fig2, ax = plt.subplots(figsize=(5, 4))

frac_pos, mean_pred = calibration_curve(y_test, proba[:, 1], n_bins=10)

ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.plot(mean_pred, frac_pos, "s-", color="#4CAF50", label=f"Random Forest (ECE={ece:.3f})")

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability Diagram")
ax.legend()

fig2.tight_layout()
fig2.savefig("reliability_diagram.png")
plt.close(fig2)

# Global SHAP summary

plt.figure(figsize=(7, 5))

shap.summary_plot(
    sv_malignant,
    X_test,
    plot_type="bar",
    max_display=10,
    show=False
)

plt.title("Global SHAP Importance for Malignant Class")
plt.tight_layout()
plt.savefig("shap_summary_global.png")
plt.close()


# SHAP comparison
top_n = 10
top_idx = np.argsort(mean_abs_high + mean_abs_low)[::-1][:top_n]
feature_names = X_test.columns[top_idx]

x = np.arange(top_n)
width = 0.35

fig4, ax = plt.subplots(figsize=(8, 4))

ax.bar(x - width / 2, mean_abs_high[top_idx], width, color="#2196F3", label="High confidence")
ax.bar(x + width / 2, mean_abs_low[top_idx], width, color="#F44336", label="Low confidence")

ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean |SHAP value|")
ax.set_title("SHAP Importance for High vs Low Confidence Predictions")
ax.legend()

fig4.tight_layout()
fig4.savefig("shap_stability.png")
plt.close(fig4)


print("\nSaved figures:")
print("- entropy_distribution.png")
print("- reliability_diagram.png")
print("- shap_summary_global.png")
print("- shap_stability.png")
