import mlflow
import json

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = mlflow.MlflowClient()

model_name = "student_predictor_champion"

# Afficher tous les modeles disponibles
print("Modeles disponibles dans le Registry :")
versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"  v{v.version} | {v.tags.get('algorithm', 'unknown')} | Stage : {v.current_stage}")

# Trouver le modele en Production
production = [v for v in versions if v.current_stage == "Production"]
archived   = [v for v in versions if v.current_stage == "Archived"]

if not production:
    print("Aucun modele en Production.")
    exit(1)

current = production[0]
print(f"\nChampion actuel : v{current.version} — Stage : {current.current_stage}")

# Simuler un drift detecte
print("\nDrift detecte ! PSI > 0.2 sur 'previous_scores'")
print("Declenchement du rollback...")

if archived:
    # Rollback vers le dernier archive
    previous = sorted(archived, key=lambda v: int(v.version), reverse=True)[0]

    # Remettre l'ancien en Production
    client.transition_model_version_stage(
        name    = model_name,
        version = previous.version,
        stage   = "Production",
        archive_existing_versions=True
    )
    print(f"\nRollback effectue !")
    print(f"Nouveau champion : v{previous.version}")
    print(f"Ancien champion  : v{current.version} archive")
else:
    print("\nAucun modele archive disponible pour rollback.")
    print("Conseil : entrainez un nouveau modele avec train_all.py")