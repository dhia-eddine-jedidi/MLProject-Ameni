# app_flask.py

from flask import Flask, render_template, request
import joblib
import pandas as pd
import plotly.express as px
from plotly.offline import plot

app = Flask(__name__)

# =============== LOAD MODELS & DATA ===============
cluster_model = joblib.load("kmeans_patient_segmentation.pkl")
risk_model = joblib.load("risk_classifier_xgboost.pkl")
pca_model = joblib.load("pca_model.pkl")
df_pca_bg = pd.read_csv("pca_background.csv")

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

GIRLY_COLORS = {
    "Cluster 0": "#FFB7DD",
    "Cluster 1": "#B5A1FF",
    "Cluster 2": "#FF9E8C",
}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    plot_div = None

    if request.method == "POST":
        try:
            age = int(request.form.get("age"))
            gender = request.form.get("gender")
            insurance = request.form.get("insurance")
            medical_condition = request.form.get("medical_condition")
            admission_type = request.form.get("admission_type")
            medication = request.form.get("medication")
            test_results = request.form.get("test_results")
            billing_amount = float(request.form.get("billing_amount"))
            stay_days = int(request.form.get("stay_days"))

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

            # ---- CLUSTERING
            cluster_pred = int(cluster_model.predict(input_data)[0])
            c_info = cluster_descriptions.get(cluster_pred, {})

            # ---- PCA POSITION
            X_proc = cluster_model.named_steps["preprocess"].transform(input_data)
            X_pca_new = pca_model.transform(X_proc)

            df_bg = df_pca_bg.copy()
            df_bg["Type"] = "Données historiques"
            df_bg["Cluster_str"] = df_bg["Cluster"].apply(lambda x: f"Cluster {x}")

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

            # petits points pour l'historique
            for trace in fig.data:
                if "Données historiques" in trace.name:
                    trace.marker.size = 3

            # point rouge en losange pour le nouveau patient
            for trace in fig.data:
                if "Nouveau patient" in trace.name:
                    trace.marker.size = 14
                    trace.marker.symbol = "diamond"
                    trace.marker.color = "#FF2D55"
                    trace.marker.line = dict(width=3, color="white")

            plot_div = plot(fig, output_type="div", include_plotlyjs="cdn")

            # ---- RISQUE
            risk_idx = int(risk_model.predict(input_data)[0])
            risk_label = (
                RISK_CLASSES[risk_idx]
                if 0 <= risk_idx < len(RISK_CLASSES)
                else "Inconnu"
            )

            result = {
                "cluster": f"Cluster {cluster_pred}",
                "cluster_title": c_info.get("title", ""),
                "cluster_label": c_info.get("label", ""),
                "cluster_profile": c_info.get("profile", ""),
                "cluster_interpretation": c_info.get("interpretation", ""),
                "risk": risk_label,
            }

        except Exception as e:
            result = {"error": str(e)}

    return render_template("index.html", result=result, plot_div=plot_div)


if __name__ == "__main__":
    app.run(debug=True)
