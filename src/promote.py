import mlflow
import json

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

# Charger le meilleur modele depuis comparison.json
summary    = json.load(open("metrics/comparison.json"))
best_name  = summary["best_model"]
best_metrics = summary["all_results"][best_name]

print(f"Meilleur modele : {best_name}")
print(f"Metriques       : {best_metrics}")

# Chercher le run correspondant dans MLflow
experiment = client.get_experiment_by_name("student_performance_comparison")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.algorithm = '{best_name}'",
    order_by=["metrics.roc_auc DESC"],
    max_results=1
)

if not runs:
    print("Aucun run trouve pour ce modele.")
    exit(1)

best_run = runs[0]
run_id   = best_run.info.run_id
print(f"Run ID : {run_id}")

# Enregistrer dans le Model Registry
model_uri  = f"runs:/{run_id}/model"
model_name = "student_predictor_champion"

result = mlflow.register_model(model_uri, model_name)
version = result.version
print(f"Modele enregistre : version {version}")

# Promouvoir en Production
client.transition_model_version_stage(
    name    = model_name,
    version = version,
    stage   = "Production",
    archive_existing_versions=True
)

print(f"\nModele {best_name} v{version} promu en PRODUCTION")
print(f"Accuracy : {best_metrics['accuracy']*100:.2f}%")
print(f"AUC-ROC  : {best_metrics['roc_auc']:.4f}")