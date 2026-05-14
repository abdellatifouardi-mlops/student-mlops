import mlflow
import mlflow.sklearn
import yaml
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Charger donnees
X_train = pickle.load(open("data/processed/X_train.pkl", "rb"))
y_train = pickle.load(open("data/processed/y_train.pkl", "rb"))
X_test  = pickle.load(open("data/processed/X_test.pkl",  "rb"))
y_test  = pickle.load(open("data/processed/y_test.pkl",  "rb"))

Path("metrics").mkdir(exist_ok=True)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("student_performance_comparison")

# Definir les 4 algorithmes
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=42
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1, random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "SVM": SVC(
        kernel="rbf", class_weight="balanced",
        probability=True, random_state=42
    ),
}

results = {}

for name, model in models.items():
    print(f"\nEntrainement : {name} ...")

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("algorithm", name)
        mlflow.set_tag("dataset_version", "v1")

        # Entrainement
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metriques
        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        }

        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        results[name] = metrics
        print(f"  {name} -> {metrics}")

# Trouver le meilleur modele
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_metrics = results[best_name]

print(f"\nMeilleur modele : {best_name}")
print(f"Metriques      : {best_metrics}")

# Sauvegarder le resume
summary = {"best_model": best_name, "all_results": results}
json.dump(summary, open("metrics/comparison.json", "w"), indent=2)
print("\nResume sauvegarde dans metrics/comparison.json")