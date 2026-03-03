# AI for Health – SRIP 2026
## Sleep Breathing Irregularity Detection

This project analyzes overnight sleep data (8 hours × 5 participants) to detect:

- Normal Breathing
- Hypopnea
- Obstructive Apnea

**Signals used:**

- Nasal Airflow (32 Hz)
- Thoracic Movement (32 Hz)
- SpO₂ (4 Hz)
- Event annotations (apnea/hypopnea)
- Sleep stage profile

The pipeline includes visualization, preprocessing, window-based labeling, 1D CNN modeling, and Leave-One-Participant-Out (LOPO) evaluation.

---

## 1️⃣ Visualization (4 Marks)

**Script**

```bash
python scripts/vis.py -name "Data/AP01"
```

**What it does**

- Loads Nasal Airflow, Thoracic Movement, and SpO₂
- Aligns signals using timestamps
- Plots full 8-hour recording
- Overlays annotated breathing events
- Saves PDF to `Visualizations/`

**Output:**

```
Visualizations/AP01_visualization.pdf
```

---

## 2️⃣ Dataset Creation (6 Marks)

**Script**

```bash
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

**Processing Steps**

1. **Bandpass Filtering (0.17–0.4 Hz)** — Retains physiological breathing frequency (10–24 BrPM).
2. **Window Segmentation**
   - 30-second windows
   - 50% overlap
3. **Labeling Rule**
   - 50% overlap with event → assign event label
   - Otherwise → Normal

**Output**

Saved to `Dataset/` (CSV / NumPy format):

- Labeled 30s breathing windows
- Corresponding class labels

---

## 3️⃣ Modeling & Evaluation (5 Marks)

**Script**

```bash
python scripts/train_model.py
```

**Model**

- 1D Convolutional Neural Network (CNN)
- Input: 30-second airflow windows
- Output: 3-class classification

**Evaluation Strategy**

Leave-One-Participant-Out (LOPO)

- Train on 4 participants
- Test on remaining 1
- Repeat for all 5 participants

**Reported Metrics**

- Accuracy
- Precision
- Recall
- Confusion Matrix

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

**Main libraries:**

- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- torch / tensorflow

---

## Deliverables Included

| File / Folder | Description |
|---|---|
| `vis.py` | Visualization script |
| `create_dataset.py` | Preprocessing & labeling |
| `train_model.py` | CNN training + LOPO evaluation |
| `Visualizations/` | Generated PDFs |
| `Dataset/` | Processed labeled windows |
| `report.pdf` | Methodology and results |
