import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["prepare"]
Path("data/processed").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/raw/students.csv")
print(f"Dataset charge : {df.shape[0]} lignes, {df.shape[1]} colonnes")

# Renommer les colonnes
df.columns = [
    "hours_studied",
    "previous_scores",
    "extracurricular",
    "sleep_hours",
    "papers_practiced",
    "performance_index"
]

# Creer la cible binaire : 1 = Succes, 0 = Echec
df["pass_fail"] = (df["performance_index"] >= params["pass_threshold"]).astype(int)
df = df.drop(columns=["performance_index"])

print(df["pass_fail"].value_counts())
print(df.head(3))

df = df.dropna().drop_duplicates()

train, test = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"],
    stratify=df["pass_fail"]
)

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv",  index=False)
print(f"Train : {len(train)} | Test : {len(test)}")