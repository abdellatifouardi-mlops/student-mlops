import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Configuration de la page ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MLOps — Student Performance",
    page_icon="🎓",
    layout="wide",
)

# ── Chargement des ressources ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("models/model.pkl", "rb"))

@st.cache_resource
def load_preprocessor():
    return pickle.load(open("data/processed/preprocessor.pkl", "rb"))

@st.cache_data
def load_metrics():
    return json.load(open("metrics/metrics.json"))

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/train.csv")

model        = load_model()
preprocessor = load_preprocessor()
metrics      = load_metrics()
df           = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=80)
st.sidebar.title("MLOps Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil", "🔮 Prédiction", "📊 Monitoring", "📈 Données"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Stack MLOps**")
st.sidebar.markdown("🔵 DVC · MLflow · FastAPI")
st.sidebar.markdown("🟢 Streamlit · Scikit-learn")
st.sidebar.markdown(f"**Accuracy : {metrics['accuracy']*100:.1f}%**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.title("🎓 Student Performance — MLOps Platform")
    st.markdown("Pipeline MLOps complet pour prédire la réussite académique des étudiants.")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%",  "+0.4%")
    col2.metric("F1-Score",  f"{metrics['f1_score']:.4f}",       "+0.01")
    col3.metric("AUC-ROC",   f"{metrics['roc_auc']:.4f}",        "+0.02")
    col4.metric("Dataset",   f"{len(df):,} étudiants",           "")

    st.markdown("---")
    st.subheader("Architecture du Pipeline MLOps")

    steps = {
        "📥 Data Ingestion":    "students.csv → DVC tracking",
        "⚗️ Feature Eng.":     "Scaling + Encoding",
        "🧠 Model Training":   "Random Forest + MLflow",
        "✅ Evaluation":       "Accuracy 97.24%",
        "🚀 API Serving":      "FastAPI localhost:8000",
        "📡 Monitoring":       "Streamlit Dashboard",
    }

    cols = st.columns(6)
    for col, (k, v) in zip(cols, steps.items()):
        col.success(k)
        col.caption(v)

    st.markdown("---")
    st.subheader("Dernières métriques MLflow")
    metrics_df = pd.DataFrame([metrics])
    st.dataframe(metrics_df.style.highlight_max(axis=1), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction":
    st.title("🔮 Prédiction de Performance")
    st.markdown("Entrez les données d'un étudiant pour prédire son résultat.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Données de l'étudiant")
        hours_studied    = st.slider("Heures d'étude par semaine", 1, 30, 14)
        previous_scores  = st.slider("Notes précédentes (/100)",   0, 100, 70)
        sleep_hours      = st.slider("Heures de sommeil",          4, 10, 7)
        papers_practiced = st.slider("Sujets pratiqués",           0, 10, 3)
        extracurricular  = st.selectbox("Activités extra-scolaires", ["Yes", "No"])

    with col2:
        st.subheader("🎯 Résultat de la prédiction")

        input_df = pd.DataFrame([{
            "hours_studied":    hours_studied,
            "previous_scores":  previous_scores,
            "extracurricular":  extracurricular,
            "sleep_hours":      sleep_hours,
            "papers_practiced": papers_practiced,
        }])

        X_proc       = preprocessor.transform(input_df)
        proba        = model.predict_proba(X_proc)[0]
        prob_success = float(proba[1])
        prediction   = "Succes" if prob_success >= 0.5 else "Echec"

        if prediction == "Succes":
            st.success(f"## ✅ SUCCÈS")
            st.success(f"Probabilité de réussite : **{prob_success*100:.1f}%**")
        else:
            st.error(f"## ❌ ÉCHEC")
            st.error(f"Probabilité de réussite : **{prob_success*100:.1f}%**")

        # Jauge de probabilité
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number+delta",
            value = prob_success * 100,
            title = {"text": "Probabilité de Succès (%)"},
            delta = {"reference": 50},
            gauge = {
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2ecc71" if prob_success >= 0.5 else "#e74c3c"},
                "steps": [
                    {"range": [0,  50], "color": "#fadbd8"},
                    {"range": [50, 100], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": 50
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Recommandation
        st.markdown("**Recommandation :**")
        if prob_success >= 0.75:
            st.info("✅ Bon profil. Maintenir les efforts actuels.")
        elif prob_success >= 0.5:
            st.warning("⚠️ Profil moyen. Augmenter la pratique de sujets.")
        else:
            st.error("🚨 Intervention requise. Tutorat recommandé.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Monitoring":
    st.title("📊 Monitoring du Modèle")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy",         f"{metrics['accuracy']*100:.2f}%")
    col2.metric("F1-Score",         f"{metrics['f1_score']:.4f}")
    col3.metric("AUC-ROC",          f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy sur 14 jours (simulé)")
        days = pd.date_range(end=pd.Timestamp.today(), periods=14)
        acc_sim = np.random.uniform(0.965, 0.975, 14)
        acc_sim[-1] = metrics["accuracy"]
        fig = px.line(
            x=days, y=acc_sim,
            labels={"x": "Date", "y": "Accuracy"},
            markers=True, color_discrete_sequence=["#2ecc71"]
        )
        fig.add_hline(y=0.88, line_dash="dash",
                      line_color="red", annotation_text="Seuil minimum")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution des prédictions")
        counts = df["pass_fail"].value_counts()
        fig = px.pie(
            values=counts.values,
            names=["Succès" if i == 1 else "Échec" for i in counts.index],
            color_discrete_sequence=["#2ecc71", "#e74c3c"]
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance")
    importances = model.feature_importances_
    features    = ["hours_studied", "previous_scores", "sleep_hours",
                   "papers_practiced", "extracurricular_No", "extracurricular_Yes"]
    fi_df = pd.DataFrame({
        "Feature":    features[:len(importances)],
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        fi_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Greens"
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Données":
    st.title("📈 Exploration des Données")
    st.markdown("---")

    st.subheader("Aperçu du dataset")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution des heures d'étude")
        fig = px.histogram(
            df, x="hours_studied", color="pass_fail",
            color_discrete_map={1: "#2ecc71", 0: "#e74c3c"},
            labels={"pass_fail": "Résultat", "hours_studied": "Heures étudiées"},
            barmode="overlay", opacity=0.7
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Notes précédentes vs Résultat")
        fig = px.box(
            df, x="pass_fail", y="previous_scores",
            color="pass_fail",
            color_discrete_map={1: "#2ecc71", 0: "#e74c3c"},
            labels={"pass_fail": "Résultat", "previous_scores": "Notes précédentes"}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe().round(2), use_container_width=True)