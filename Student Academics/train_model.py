from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "student-mat.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
GRADE_CATEGORIES = ["Low", "Average", "High"]

def grade_to_category(grade):
    if grade <= 9:
        return 0
    if grade <= 14:
        return 1
    return 2

def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical)
    ])

def build_dnn(input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def evaluate_model(name, y_true, y_pred):
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "classification_report": classification_report(
            y_true, y_pred, target_names=GRADE_CATEGORIES, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def train_mode(df, mode):
    # Early prediction deliberately excludes G1 and G2.
    drop_columns = ["G3"]
    if mode == "early":
        drop_columns += ["G1", "G2"]

    X = df.drop(columns=drop_columns)
    y = df["G3"].apply(grade_to_category).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    # Select useful input columns using mutual information.
    k = min(30, X_train_ready.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_ready, y_train)
    X_test_selected = selector.transform(X_test_ready)

    # TensorFlow expects dense NumPy arrays, while one-hot encoding may be sparse.
    if hasattr(X_train_selected, "toarray"):
        X_train_selected = X_train_selected.toarray()
    if hasattr(X_test_selected, "toarray"):
        X_test_selected = X_test_selected.toarray()

    # Traditional ML baseline.
    rf = RandomForestClassifier(
        n_estimators=250, max_depth=None, min_samples_split=2,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    rf.fit(X_train_selected, y_train)
    rf_pred = rf.predict(X_test_selected)

    # Deep Neural Network.
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    dnn = build_dnn(X_train_selected.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    history = dnn.fit(
        X_train_selected, y_train,
        validation_split=0.20,
        epochs=120,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    dnn_prob = dnn.predict(X_test_selected, verbose=0)
    dnn_pred = np.argmax(dnn_prob, axis=1)

    # Save all objects required by the Streamlit app.
    joblib.dump(preprocessor, MODEL_DIR / f"{mode}_preprocessor.pkl")
    joblib.dump(selector, MODEL_DIR / f"{mode}_selector.pkl")
    joblib.dump(rf, MODEL_DIR / f"{mode}_random_forest.pkl")
    dnn.save(MODEL_DIR / f"{mode}_dnn.keras")

    results = {
        "mode": mode,
        "features": X.columns.tolist(),
        "selected_feature_count": int(k),
        "random_forest": evaluate_model("Random Forest", y_test, rf_pred),
        "dnn": evaluate_model("Deep Neural Network", y_test, dnn_pred),
        "epochs_trained": len(history.history["loss"]),
        "test_records": int(len(y_test))
    }
    with open(OUTPUT_DIR / f"{mode}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{mode.upper()} PREDICTION")
    print("Random Forest accuracy:", results["random_forest"]["accuracy"])
    print("DNN accuracy:", results["dnn"]["accuracy"])
    print("Selected features:", k)
    print("Epochs trained:", results["epochs_trained"])
    return results

def main():
    if not DATA_FILE.exists():
        print("Dataset not found. Downloading it now...")
        import subprocess, sys
        subprocess.check_call([sys.executable, str(ROOT / "download_dataset.py")])

    df = pd.read_csv(DATA_FILE, sep=";")
    if "G3" not in df.columns:
        raise ValueError("Dataset must contain the target column G3.")

    print("Dataset shape:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Grade categories:", df["G3"].apply(grade_to_category).value_counts().to_dict())

    train_mode(df, "early")
    train_mode(df, "updated")
    print("\nTraining completed. Model files are saved in the models folder.")

if __name__ == "__main__":
    main()
