import mlflow
import mlflow.sklearn
import yaml
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

params  = yaml.safe_load(open("params.yaml"))["train"]
X_train = pickle.load(open("data/processed/X_train.pkl", "rb"))
y_train = pickle.load(open("data/processed/y_train.pkl", "rb"))
X_test  = pickle.load(open("data/processed/X_test.pkl",  "rb"))
y_test  = pickle.load(open("data/processed/y_test.pkl",  "rb"))

Path("models").mkdir(exist_ok=True)
Path("metrics").mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("student_performance")

with mlflow.start_run(run_name="RandomForest_v1"):

    mlflow.log_params(params)

    model = RandomForestClassifier(
        n_estimators      = params["n_estimators"],
        max_depth         = params["max_depth"],
        min_samples_split = params["min_samples_split"],
        min_samples_leaf  = params["min_samples_leaf"],
        class_weight      = params["class_weight"],
        random_state      = params["random_state"],
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
    mlflow.set_tag("dataset", "students_kaggle")

    json.dump(metrics, open("metrics/metrics.json", "w"), indent=2)
    pickle.dump(model,  open("models/model.pkl", "wb"))

    print("Metrics :", metrics)
    print("Run MLflow termine avec succes")