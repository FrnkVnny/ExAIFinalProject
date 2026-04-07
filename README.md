# Explaining Prediction Uncertainty in Breast Cancer Diagnosis Models

**Course:** Explainable AI - Radboud University  
**Author:** Francisco Vanney (s1138154)

## Overview

This project looks at how prediction uncertainty can be explained in a breast cancer classification task.

The main idea is not only to see what the model predicts, but also how certain it is about those predictions. For that, two different explanation methods are used.

The first one is **SHAP**, which helps show which input features matter most for the model’s decision. The second one is **predictive entropy**, which is used as a measure of uncertainty. Entropy was not covered in the lectures, so it is included here as the more novel part of the project.

To evaluate the results, the project also uses **Expected Calibration Error (ECE)** to check whether the model’s predicted probabilities are reliable. In addition, **SHAP stability** is compared between more confident and less confident predictions, to see whether the explanations change depending on the certainty of the model.

## Dataset

The model is tested on the Wisconsin Breast Cancer Dataset, which is loaded directly from `sklearn.datasets`.

## Output

The script produces four figures:

| File | Description |
|---|---|
| `entropy_distribution.png` | Distribution of uncertainty values for more confident and less confident predictions |
| `reliability_diagram.png` | Calibration plot used together with ECE |
| `shap_summary_global.png` | Global SHAP feature importance |
| `shap_stability.png` | Comparison of SHAP importance for high- and low-confidence predictions |

## Notes

- The random seed is fixed at 42 so that the results stay the same across runs.
- High-confidence and low-confidence groups are defined using the lowest 25% and highest 25% of entropy values.
- The classifier used in the project is a Random Forest model.
- The focus of the project is on explaining both predictions and uncertainty, not on building the most advanced predictive model possible.
