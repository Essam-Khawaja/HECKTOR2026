# HECKTOR 2026 — Task Overview

The HECKTOR 2026 challenge consists of a **single unified task**: an end-to-end pipeline for head and neck cancer patient assessment using multimodal FDG-PET/CT and clinical data. The pipeline comprises three sequential, clinically-linked subtasks that all participants must complete.

---

## Pipeline Structure

```
FDG-PET/CT + Clinical Data
        │
        ▼
┌───────────────────────────────────┐
│  Subtask 1: Segmentation          │
│  Segment GTVp (primary tumor)     │
│  and GTVn (lymph nodes)           │
│  Metric: Mean Dice (GTVp + GTVn)  │
└────────────────┬──────────────────┘
                 │ segmentation masks + imaging features
                 ▼
┌───────────────────────────────────┐
│  Subtask 2: TN Staging            │
│  Classify T stage (T1–T4) and     │
│  N stage (N0–N3) per AJCC 7th Ed  │
│  Metric: Balanced Accuracy +      │
│          Recall (T and N)         │
└────────────────┬──────────────────┘
                 │ predicted TN stage + imaging features
                 ▼
┌───────────────────────────────────┐
│  Subtask 3: Prognosis             │
│  Predict Recurrence-Free Survival │
│  (RFS) risk score                 │
│  Metric: C-index                  │
└───────────────────────────────────┘
```

---

## Subtask Details

### Subtask 1 — Segmentation (`Segmentation/`)

- **Goal:** Automatically detect and delineate the primary tumor (GTVp) and all metastatic lymph nodes (GTVn) in paired FDG-PET/CT volumes.
- **Output:** A 3D segmentation mask (label 0 = background, 1 = GTVp, 2 = GTVn)
- **Metric:** Mean Dice score averaged over GTVp and GTVn
- **Baseline models:** UNet3D, SegResNet, UNETR, SwinUNETR (via MONAI)
- **Weight in final ranking:** 0.25

### Subtask 2 — TN Staging (`TNStaging/`)

- **Goal:** Classify the radiological T stage (T1–T4) and N stage (N0–N3) for each patient using PET/CT images and clinical information. N subcategories N2b and N2c are collapsed to N2 per AJCC/UICC 7th Edition.
- **Output:** Two categorical predictions per patient: `T_stage` and `N_stage`
- **Metric:** Balanced accuracy and recall for T and N classification
- **Baseline model:** Multimodal ResNet18 with dual classification heads
- **Weight in final ranking:** 0.35

### Subtask 3 — Prognosis (`Prognosis/`)

- **Goal:** Predict each patient's recurrence-free survival (RFS) risk score using PET/CT images, clinical variables, and (optionally) the predicted TN stage from Subtask 2.
- **Output:** A continuous risk score per patient (higher = higher risk)
- **Metric:** Concordance index (C-index)
- **Baseline model:** ResNet18 feature extractor + BaggedIcareSurvival ensemble
- **Weight in final ranking:** 0.40

---

## Participation Options

Participants may implement:
- **Modular approaches**: separately optimized models for each subtask
- **End-to-end models**: a single model that jointly predicts all three outputs

Both approaches are valid. End-to-end solutions are encouraged as they more closely reflect real-world clinical practice.

---

## Data

All subtasks share the same dataset. See the [main README](../README.md#-getting-the-data) for the full dataset description and download instructions.

The training CSV (`HECKTOR_2026_Training.csv`) includes:
- Clinical variables: age, gender, tobacco, alcohol, performance status, HPV status
- Subtask 1 labels: segmentation masks (`.nii.gz` files)
- Subtask 2 labels: `T_stage`, `N_stage`
- Subtask 3 labels: `Relapse`, `RFS`
