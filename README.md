# Feature-Engineering

# Audio Feature Extraction with Librosa

## Overview

This project extracts numerical audio features from a directory of sound files and compiles them into a structured dataset (`CSV`). The resulting feature matrix can be used for machine learning tasks such as classification (e.g., genre detection, speech recognition, or sound classification).

The script processes audio files and computes:

* Mel-Frequency Cepstral Coefficients (MFCCs)
* Root Mean Square (RMS) Energy
* Zero-Crossing Rate (ZCR)

---

## Features Extracted

### 1. MFCC (Mel-Frequency Cepstral Coefficients)

* `N_MFCC = 9`
* For each coefficient:

  * Mean
  * Standard deviation
* Total: **18 features**

### 2. RMS Energy

* Mean
* Standard deviation
* Maximum
* Minimum
* Total: **4 features**

### 3. Zero-Crossing Rate (ZCR)

* Mean
* Standard deviation
* Maximum
* Minimum
* Total: **4 features**

### Total Features per Audio File

**26 features + 1 filename column**

---

## Directory Structure

```
project/
│── data/              # Input audio files (.wav, .mp3, etc.)
│── features.csv       # Output dataset (generated)
│── extract.py         # Main script
│── README.md
```

---

## Supported Audio Formats

* `.wav`
* `.mp3`
* `.flac`
* `.ogg`
* `.m4a`

---

## Installation

Install required dependencies:

```bash
pip install numpy pandas librosa
```

---

## Configuration

At the top of the script, modify:

```python
AUDIO_DIR = "./data"     # Folder containing audio files
OUTPUT_CSV = "features.csv"
N_MFCC = 9
SAMPLE_RATE = None       # Use original sampling rate
```

---

## Usage

Run the script:

```bash
python extract.py
```

---

## Output

The script generates a CSV file where:

* Each row = one audio file
* Each column = a feature

Example:

| filename   | mfcc_mean_01 | mfcc_std_01 | ... | rms_mean | zcr_mean |
| ---------- | ------------ | ----------- | --- | -------- | -------- |
| audio1.wav | -120.3       | 15.2        | ... | 0.032    | 0.08     |

---

## Console Output

The script will:

* List all processed files
* Display feature matrix shape
* Show a preview of extracted features

Example:

```
Found 25 audio file(s)

FEATURE MATRIX – Shape Explanation
Rows (samples): 25
Columns (features): 26

CSV saved → /path/to/features.csv
```

---

## Error Handling

* Files that fail to process are skipped
* Errors are printed without stopping execution

---

## Use Cases

This dataset can be used for:

* Sound classification
* Music genre prediction
* Speech emotion recognition
* Machine learning model comparison (k-NN, SVM, Random Forest)

---

## Notes

* Feature scaling is recommended before training models (especially for k-NN and SVM)
* Ensure consistent labeling if using this for supervised learning (labels are not included in this script)

---

## Future Improvements

* Add label extraction from filenames or metadata
* Include additional features (spectral centroid, chroma features)
* Support batch processing with subdirectories
* Integrate directly with machine learning pipelines

---

## Author

Your Name
Course / Project Name
Date

---
