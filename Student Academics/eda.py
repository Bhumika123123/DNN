from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "student-mat.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError("Run: python download_dataset.py")

df = pd.read_csv(DATA, sep=";")
print("\nDataset shape:", df.shape)
print("\nFirst five records:\n", df.head())
print("\nData information:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe())

sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 4))
sns.countplot(x="G3", data=df)
plt.title("Distribution of Final Grades (G3)")
plt.tight_layout()
plt.savefig(OUT / "grade_distribution.png")
plt.close()

plt.figure(figsize=(10, 7))
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png")
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(x="studytime", y="G3", data=df)
plt.title("Study Time vs Final Grade")
plt.tight_layout()
plt.savefig(OUT / "studytime_vs_grade.png")
plt.close()

print("\nEDA charts saved in the outputs folder.")
