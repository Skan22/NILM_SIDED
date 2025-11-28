# Running NILM Project on Google Colab

This guide explains how to train your models using Google Colab's free GPU resources.

## Prerequisites
*   A Google Account.
*   The `SIDED` dataset folder (original data).
*   This project's code files (`src/`, `reproduce_paper.py`, `requirements.txt`, etc.).

## Step 1: Upload to Google Drive
1.  Go to [Google Drive](https://drive.google.com).
2.  Create a new folder named `NILM_Project`.
3.  Upload your **project code** (the `src` folder, `requirements.txt`, `reproduce_paper.py`, `data_augmentation.py`) into this folder.
4.  Upload your **dataset folder** (`SIDED`) into this folder as well.

Your Drive structure should look like this:
```
My Drive/
└── NILM_Project/
    ├── SIDED/              <-- Original Data
    ├── src/                <-- Source code
    ├── requirements.txt
    ├── reproduce_paper.py
    └── data_augmentation.py
```

## Step 2: Create a Colab Notebook
1.  Go to [Google Colab](https://colab.research.google.com).
2.  Click **New Notebook**.
3.  **Enable GPU**: Go to `Runtime` > `Change runtime type` > Select `T4 GPU` (or better) > Click `Save`.

## Step 3: Run the Training
Copy and paste the following commands into code cells in your Colab notebook:

### 1. Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Navigate to Project Directory
```python
import os
# Change this path if you named your folder differently
os.chdir('/content/drive/MyDrive/NILM_Project')
print("Current Working Directory:", os.getcwd())
```

### 3. Install Dependencies
```bash
!pip install -r requirements.txt
```

### 4. Run Data Augmentation (First Time Only)
If you haven't generated the augmented data (`AMDA_SIDED`) yet, run this:
```bash
!python data_augmentation.py --input ./SIDED --output ./AMDA_SIDED --alphas 1.5 4.0
```

### 5. Run Training (Reproduction Script)
Now run the main reproduction script. This will train the TCN, ATCN, and BiLSTM models and save them.
```bash
!python reproduce_paper.py
```

## Troubleshooting
*   **Path Errors**: If you see "File not found", double-check your folder name in Drive and the `os.chdir` command.
*   **Memory Issues**: If Colab crashes, try reducing the `batch_size` in `reproduce_paper.py` (e.g., from 64 to 32).
