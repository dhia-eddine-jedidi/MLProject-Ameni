# app_streamlit.py

import time
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =============== UTILS ===============
def typewriter(text: str, delay: float = 0.02):
    """Display text letter by letter in Streamlit."""
    placeholder = st.empty()
    current = ""
    for char in text:
        current += char
        placeholder.markdown(current)
        time.sleep(delay)


@st.cache_resource
def load_models_and_data():
    cluster_model = joblib.load("kmeans_patient_segmentation.pkl")
    risk_model = joblib.load("risk_classifier_xgboost.pkl")
    pca_model = joblib.load("pca_model.pkl")
    df_pca_bg = pd.read_csv("pca_background.csv")
    return cluster_model, risk_model, pca_model, df_pca_bg


cluster_model, risk_model, pca_model, df_pca_bg = load_models_and_data()

RISK_CLASSES = ["Faible", "Moyenne", "Élevé"]

cluster_descriptions = {
    0: {
        "title": "Cluster 0 – Senior, Low-Cost, Longer Care",
        "profile": "Patients âgés (~70 ans), facturation modérée (~16k), séjours les plus longs (~16 jours).",
        "interpretation": "Pathologies chroniques mais stables, nécessitant un suivi prolongé plus qu’une prise en charge intensive.",
        "label": "Senior – Chronic & Stable Care",
    },
    1: {
        "title": "Cluster 1 – Middle-Aged, High-Cost, Intensive Care",
        "profile": "Patients d’âge moyen (~52 ans), facturation très élevée (~40k), séjours d’environ 15 jours.",
        "interpretation": "Cas complexes et coûteux (cancer, obésité sévère, hypertension…), nécessitant des traitements intensifs.",
        "label": "High-Complexity / High-Cost Patients",
    },
    2: {
        "title": "Cluster 2 – Young Adults, Moderate Cost",
        "profile": "Jeunes adultes (~33 ans), facturation modérée (~16.8k), séjours plus courts (~15 jours).",
        "interpretation": "Pathologies aiguës mais moins sévères (asthme, infections, blessures mineures) avec récupération rapide.",
        "label": "Young – Acute & Fast Recovery",
    },
}

# pastel girly colors per cluster label
GIRLY_COLORS = {
    "Cluster 0": "#FFB7DD",  # baby pink
    "Cluster 1": "#B5A1FF",  # lavender
    "Cluster 2": "#FF9E8C",  # peach pink
}


# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="Patient Segmentation & Risk Prediction",
    layout="wide",
)

st.title("🏥 Patient Segmentation & Risk Prediction")

st.markdown(
    """
Cette application :

- Segmente le patient dans un **cluster** (KMeans),
- Affiche sa position dans l’espace **3D PCA** (Plotly),
- Prédit son **niveau de risque** (Faible / Moyenne / Élevé) via **XGBoost**.
"""
)

# =============== FORMULAIRE ===============
st.sidebar.header("🧍‍♂️ Informations Patient")

with st.sidebar.form("patient_form"):
    age = st.number_input("Âge", min_value=0, max_value=120, value=45, step=1)
    gender = st.selectbox("Genre", ["Male", "Female"])
    insurance = st.selectbox(
        "Assureur", ["Aetna", "Blue Cross", "Cigna", "Medicare", "UnitedHealthcare"]
    )
    medical_condition = st.selectbox(
        "Condition médicale",
        ["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"],
    )
    admission_type = st.selectbox("Type d’admission", ["Emergency", "Urgent", "Elective"])
    medication = st.selectbox(
        "Médication", ["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin"]
    )
    test_results = st.selectbox(
        "Résultats des tests", ["Normal", "Abnormal", "Inconclusive"]
    )
    billing_amount = st.number_input(
        "Montant facturé ($)",
        min_value=0.0,
        max_value=100000.0,
        value=20000.0,
        step=500.0,
    )
    stay_days = st.number_input(
        "Durée de séjour (jours)", min_value=0, max_value=365, value=10, step=1
    )

    submitted = st.form_submit_button("🔍 Lancer la prédiction")

# =============== TRAITEMENT ===============
if submitted:
    # ---- données d’entrée
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "Insurance Provider": insurance,
                "Medical Condition": medical_condition,
                "Admission Type": admission_type,
                "Medication": medication,
                "Test Results": test_results,
                "Billing Amount": billing_amount,
                "Stay_Days": stay_days,
            }
        ]
    )

    st.subheader("📥 Données saisies")
    st.write(input_data)

    # ---- CLUSTERING
    cluster_pred = int(cluster_model.predict(input_data)[0])
    info = cluster_descriptions.get(cluster_pred, {})

    st.subheader("🧩 Résultat du Clustering")

    typewriter(f"**Cluster prédit :** `Cluster {cluster_pred}`")

    if info:
        typewriter(f"**Nom du segment :** {info['label']}")
        typewriter(f"**Profil :** {info['profile']}")
        typewriter(f"**Interprétation :** {info['interpretation']}")

    # ---- PCA 3D
    st.subheader("🧭 Visualisation 3D PCA – Position du patient")

    # prétraitement + PCA
    X_proc = cluster_model.named_steps["preprocess"].transform(input_data)
    X_pca_new = pca_model.transform(X_proc)  # (1, 3)

    # fond historique
    df_bg = df_pca_bg.copy()
    df_bg["Type"] = "Données historiques"
    df_bg["Cluster_str"] = df_bg["Cluster"].apply(lambda x: f"Cluster {x}")

    # nouveau patient
    df_point = pd.DataFrame(
        {
            "PCA1": [X_pca_new[0, 0]],
            "PCA2": [X_pca_new[0, 1]],
            "PCA3": [X_pca_new[0, 2]],
            "Cluster": [cluster_pred],
            "Cluster_str": [f"Cluster {cluster_pred}"],
            "Type": ["Nouveau patient"],
        }
    )

    df_plot = pd.concat([df_bg, df_point], ignore_index=True)

    fig = px.scatter_3d(
        df_plot,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Cluster_str",
        symbol="Type",
        opacity=0.8,
        title="3D PCA – Segmentation des patients",
        hover_data=["Type"],
        color_discrete_map=GIRLY_COLORS,
    )

    # points historiques
    for trace in fig.data:
        if "Données historiques" in trace.name:
            trace.marker.size = 3

    # nouveau patient : gros losange rouge
    for trace in fig.data:
        if "Nouveau patient" in trace.name:
            trace.marker.size = 14
            trace.marker.symbol = "diamond"
            trace.marker.color = "#FF2D55"  # girly red
            trace.marker.line = dict(width=3, color="white")

    st.plotly_chart(fig, use_container_width=True)

    # ---- RISQUE
    st.subheader("⚠️ Prédiction du niveau de risque (XGBoost)")

    risk_pred_idx = int(risk_model.predict(input_data)[0])
    risk_label = (
        RISK_CLASSES[risk_pred_idx]
        if 0 <= risk_pred_idx < len(RISK_CLASSES)
        else "Inconnu"
    )

    st.markdown(f"**Niveau de risque prédit :** `{risk_label}`")

    st.markdown(
        """
- **Faible** : cas moins sévère, récupération rapide probable.  
- **Moyenne** : suivi nécessaire, risque intermédiaire.  
- **Élevé** : cas complexe / coûteux, nécessite une prise en charge renforcée.
"""
    )
else:
    st.info(
        "➡️ Remplis les informations du patient dans le panneau de gauche puis clique sur **Lancer la prédiction**."
    )
