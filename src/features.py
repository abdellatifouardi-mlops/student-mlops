import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import yaml

params = yaml.safe_load(open("params.yaml"))["prepare"]
target = params["target_column"]

train = pd.read_csv("data/processed/train.csv")
test  = pd.read_csv("data/processed/test.csv")

X_train = train.drop(columns=[target])
y_train = train[target]
X_test  = test.drop(columns=[target])
y_test  = test[target]

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Features numeriques : {num_cols}")
print(f"Features categorielles : {cat_cols}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), cat_cols),
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

pickle.dump(X_train_proc,  open("data/processed/X_train.pkl", "wb"))
pickle.dump(X_test_proc,   open("data/processed/X_test.pkl",  "wb"))
pickle.dump(y_train.values, open("data/processed/y_train.pkl", "wb"))
pickle.dump(y_test.values,  open("data/processed/y_test.pkl",  "wb"))
pickle.dump(preprocessor,   open("data/processed/preprocessor.pkl", "wb"))
print(f"Features shape : {X_train_proc.shape}")