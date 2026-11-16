# Student Academic Performance Prediction System

A beginner-friendly end-to-end project using the UCI Student Performance dataset.

## Features

- Mathematics dataset (395 student records)
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature selection using mutual information
- Deep Neural Network (TensorFlow/Keras)
- Random Forest comparison
- Early prediction (without G1 and G2)
- Updated prediction (with G1 and G2)
- Streamlit web application
- Accuracy, macro precision, macro recall, macro F1 and confusion matrix

## 1. Requirements

Install Python 3.10 or 3.11 (recommended). Use a stable internet connection for the first dataset download and package installation.

## 2. Open the project folder

Open this folder in VS Code:

`Student_Academic_Performance_Prediction`

Open the VS Code terminal.

## 3. Create a virtual environment (Windows)

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

If `py -3.11` is not available, use:

```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow can take a few minutes to install.

## 5. Download the dataset

```bash
python download_dataset.py
```

This downloads the Mathematics CSV from the UCI Student Performance archive into `data/student-mat.csv`.

## 6. Run EDA

```bash
python eda.py
```

Charts and summary outputs are saved in the `outputs` folder.

## 7. Train both models

```bash
python train_model.py
```

This trains both prediction modes. The first training may take a few minutes. Trained models are saved in `models`, and evaluation metrics are saved in `outputs`.

## 8. Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit will show a local address, usually:

`http://localhost:8501`

## Prediction modes

- **Early prediction:** does not use G1 or G2. This is the more realistic early-warning experiment.
- **Updated prediction:** uses G1 and G2. These grades are strongly related to G3, so this mode may achieve higher scores but uses information from earlier exam periods.

## Performance categories

| Final grade (G3) | Category |
|---|---|
| 0–9 | Low |
| 10–14 | Average |
| 15–20 | High |

## Project architecture

```text
Student details
      |
      v
Preprocessing
(imputation, encoding, scaling)
      |
      v
Feature selection
      |
      +-------------------+
      |                   |
      v                   v
Deep Neural Network   Random Forest
      |                   |
      +---------+---------+
                |
                v
       Predicted category
        Low / Average / High
                |
                v
          Streamlit UI
```

## Important academic notes

1. The target is the final grade `G3`, converted into three categories.
2. The project compares a DNN with Random Forest on the same train/test split for each experiment.
3. The early model excludes `G1` and `G2` to avoid using previous exam grades.
4. The updated model includes `G1` and `G2`. Explain this difference in your interview.
5. Do not claim that this model directly predicts employability. It predicts academic grade categories only.
6. Model scores are generated when you run training. Do not copy example accuracy numbers from another project.

## Dataset citation

Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository.  
https://doi.org/10.24432/C5TG7T

Dataset page: https://archive.ics.uci.edu/dataset/320/student-performance
