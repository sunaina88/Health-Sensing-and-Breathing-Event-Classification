# Breathing Event Classification using 1D CNN

## Project Overview

This project processes sleep breathing signals to detect:
- Normal breathing
- Hypopnea
- Obstructive Apnea

The pipeline includes:
1. Signal preprocessing using bandpass filtering (0.17–0.4 Hz)
2. Window segmentation (30 seconds, 50% overlap)
3. Event-based labeling using >50% overlap rule
4. 1D CNN model training
5. Leave-One-Participant-Out (LOPO) evaluation

---

## Folder Structure

Data/
- APO1/ 
    - [Flow - 30-05-2024.txt](Data%2FAP01%2FFlow%20-%2030-05-2024.txt)
    - [Flow Events - 30-05-2024.txt](Data%2FAP01%2FFlow%20Events%20-%2030-05-2024.txt)
    - [Sleep profile - 30-05-2024.txt](Data%2FAP01%2FSleep%20profile%20-%2030-05-2024.txt)
    - [SPO2 - 30-05-2024.txt](Data%2FAP01%2FSPO2%20-%2030-05-2024.txt)
    - [Thorac - 30-05-2024.txt](Data%2FAP01%2FThorac%20-%2030-05-2024.txt)
- APO2 / 
    - ....
- APO3 / 
    - ....
- APO4 / 
    - ....
- APO5 / 
    - ....

Dataset/
- AP01_X.npy
- AP01_y.npy
- APO2_X.npy
- AP02_y.npy
- APO3_X.npy
- AP03_y.npy
- APO4_X.npy
- AP04_y.npy
- APO5_X.npy
- AP05_y.npy

models/
- cnn_model.py

scripts/
- create_dataset.py
- train_model.py
- vis.py

Visualizations/
- AP01_visualization.pdf
- AP02_visualization.pdf
- AP03_visualization.pdf
- AP04_visualization.pdf
- AP05_visualization.pdf

report.pdf

requirements.txt

---

## How to Run

### 1. Create Dataset

```bash
python create_dataset.py -name AP01