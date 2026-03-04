# Health Sensing using Visualizations and 1D CNN Classification
### AI for Health – SRIP 2026 

This prerequisite assignment analyses overnight sleep data (of 8 hours) from five participants to detect abnormal breathing events during sleep. Using physiological signals such as Nasal Airflow, Thoracic Movement, and SpO₂, the system performs signal preprocessing, window-based labelling and 1D CNN classification evaluated through Leave-One-Participant-Out (LOPO) cross-validation.

---

## Project Scope

The objective is to:

- Visualize 8-hour physiological recordings with annotated breathing events
- Apply digital filtering to isolate the physiological breathing frequency range (0.17–0.4 Hz)
- Segment signals into 30-second windows with 50% overlap
- Label windows based on >50% overlap with annotated events
- Train a 1D Convolutional Neural Network for classification
- Evaluate using Leave-One-Participant-Out (LOPO) cross-validation
- Report metrics such as Accuracy, Precision, Recall and Confusion Matrix

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sunaina88/Health-Sensing-and-Breathing-Event-Classification.git
cd Health-Sensing-and-Breathing-Event-Classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Visualization

Generate visualization for one participant:

```bash
python scripts/vis.py -name "Data/AP01"
```

This will:

- Load Nasal Airflow, Thoracic Movement, and SpO₂
- Align signals using timestamps
- Overlay annotated breathing events
- Save a PDF in the `Visualizations/` directory

### 2. Dataset Creation

Create labelled 30-second windows:

```bash
python scripts/create_dataset.py "Data" "Dataset"
```

This script:

- Applies bandpass filtering (0.17–0.4 Hz)
- Segments signals into overlapping windows
- Assigns labels based on >50% event overlap
- Saves processed dataset to the `Dataset/` directory

### 3. Model Training and Evaluation

Train and evaluate the 1D CNN model:

```bash
python scripts/train_model.py
```

This performs:

- Leave-One-Participant-Out cross-validation
- Training on four participants
- Testing on the held-out participant
- Reporting the following metrics:
  - Accuracy
  - Precision
  - Recall
  - Confusion Matrix

---

## Methodological Highlights

- Frequency-based digital filtering aligned with physiological breathing rates
- Timestamp-based alignment of multi-rate signals (32 Hz and 4 Hz)
- Subject-independent evaluation strategy (LOPO)
- Window-based labelling using majority overlap rule

---

## Repository Contents

| File / Folder | Description |
|---|---|
| `scripts/vis.py` | Signal visualization and PDF generation |
| `scripts/create_dataset.py` | Preprocessing and labeling pipeline |
| `scripts/train_model.py` | CNN training and LOPO evaluation |
| `models/cnn_model.py` | CNN Model definitions |
| `Visualizations/` | Generated PDFs |
| `Dataset/` | Processed datasets |
| `report.pdf` | Detailed methodology and results |

---

> This repository was developed as a part of the AI for Health – SRIP 2026 prerequisite assignment. 

> Note: Claude tools were minimally used while implementing the CNN model, and for correcting a few coding errors. However, I have completed the assignment with full understanding of the code.
