"""API FastAPI + interface Gradio - Prédiction du prix d'une raquette de padel.

  /          -> interface Gradio
  /docs      -> documentation Swagger de l'API
  /predict   -> POST JSON, retourne le prix estimé
  /health    -> vérification de l'état du service
"""
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Chargement du modèle depuis le dossier local model/
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "model",
    "padel_price_model.joblib",
)
model = joblib.load(MODEL_PATH)
FEATURES = list(model.feature_names_in_)  # colonnes exactes de l'entraînement
VENDORS = sorted(
    model.named_steps["prep"].named_transformers_["cat"].categories_[0].tolist()
)


def predict_price(row: dict) -> float:
    df = pd.DataFrame([row])
    for c in FEATURES:          # colonne absente -> 0
        if c not in df.columns:
            df[c] = 0
    df = df[FEATURES]           # même ordre qu'à l'entraînement
    return float(np.expm1(model.predict(df))[0])


# --------------------------------------------------------------------------
# API FastAPI
# --------------------------------------------------------------------------
class Racket(BaseModel):
    vendor: str
    model_year: float = 2025
    created_year: int = 2025
    title_len: int = 45
    has_weight: int = 0
    has_shape: int = 1
    has_foam: int = 1
    has_frame: int = 1
    has_surface: int = 0
    has_professional_player: int = 0
    has_color: int = 0
    has_racket_type: int = 0
    has_collection: int = 0
    tag_ronde: int = 0
    tag_carbone: int = 0
    tag_soft_eva: int = 0
    tag_controle: int = 0
    tag_polyvalente: int = 0
    tag_avance: int = 0
    tag_goutte_deau: int = 0
    tag_femmes: int = 0
    tag_tolerance: int = 0
    tag_diamant: int = 0


app = FastAPI(title="Padel Racket Price Predictor")


@app.get("/health")
def health():
    return {"status": "ok", "features": len(FEATURES)}


@app.get("/vendors")
def vendors():
    return {"vendors": VENDORS}


@app.post("/predict")
def predict(r: Racket):
    return {"predicted_price_eur": round(predict_price(r.model_dump()), 2)}


# --------------------------------------------------------------------------
# Interface Gradio
# --------------------------------------------------------------------------
def predict_ui(vendor, year, title_len, carbone, soft_eva, sablee, pro,
               ronde, goutte, diamant, controle, polyvalente, avance,
               femmes, tolerance):
    row = {
        "vendor": vendor,
        "model_year": float(year),
        "created_year": int(min(year, 2026)),
        "title_len": int(title_len),
        "has_frame": int(carbone),
        "has_surface": int(sablee),
        "has_professional_player": int(pro),
        "has_shape": 1,
        "has_foam": 1,
        "has_weight": 0,
        "has_color": 0,
        "has_racket_type": int(controle or polyvalente),
        "has_collection": 1 if year >= 2025 else 0,
        "tag_ronde": int(ronde),
        "tag_carbone": int(carbone),
        "tag_soft_eva": int(soft_eva),
        "tag_controle": int(controle),
        "tag_polyvalente": int(polyvalente),
        "tag_avance": int(avance),
        "tag_goutte_deau": int(goutte),
        "tag_femmes": int(femmes),
        "tag_tolerance": int(tolerance),
        "tag_diamant": int(diamant),
    }
    try:
        return f"{predict_price(row):,.0f} €".replace(",", " ")
    except Exception as e:
        return f"Erreur : {e}"


with gr.Blocks(title="Padel Price Predictor") as demo:
    gr.Markdown("# 🎾 Prédicteur de prix de raquette de padel\n"
                "Renseigne les caractéristiques : le modèle estime le prix en euros.")
    with gr.Row():
        with gr.Column():
            vendor = gr.Dropdown(VENDORS, value=VENDORS[0], label="Marque")
            year = gr.Slider(2022, 2026, value=2025, step=1, label="Année du modèle")
            title_len = gr.Slider(20, 90, value=45, step=1,
                                  label="Longueur du titre (réglage avancé)")
        with gr.Column():
            gr.Markdown("**Construction**")
            carbone = gr.Checkbox(label="Carbone", value=True)
            soft_eva = gr.Checkbox(label="Mousse Soft EVA")
            sablee = gr.Checkbox(label="Surface sablée")
            pro = gr.Checkbox(label="Joueur professionnel associé")
        with gr.Column():
            gr.Markdown("**Forme et style**")
            ronde = gr.Checkbox(label="Forme ronde")
            goutte = gr.Checkbox(label="Forme goutte d'eau")
            diamant = gr.Checkbox(label="Forme diamant")
            controle = gr.Checkbox(label="Style contrôle")
            polyvalente = gr.Checkbox(label="Style polyvalent")
            avance = gr.Checkbox(label="Niveau avancé")
            femmes = gr.Checkbox(label="Gamme femme")
            tolerance = gr.Checkbox(label="Tolérance")

    btn = gr.Button("Prédire le prix", variant="primary")
    out = gr.Textbox(label="Prix estimé", interactive=False)
    btn.click(
        predict_ui,
        inputs=[vendor, year, title_len, carbone, soft_eva, sablee, pro,
                ronde, goutte, diamant, controle, polyvalente, avance,
                femmes, tolerance],
        outputs=out,
    )
    gr.Markdown("⚠️ Estimation indicative : le prix dépend surtout de la marque "
                "et du modèle exact (dataset de 628 raquettes).")

# L'interface est montée sur "/" ; les routes API ci-dessus restent prioritaires
app = gr.mount_gradio_app(app, demo, path="/")