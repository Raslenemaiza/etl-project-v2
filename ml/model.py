# ══════════════════════════════════════════════════════════════════════
# ml/model.py — Module Machine Learning automatique
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

from config.settings import ML


def get_ml_features(df: pd.DataFrame, target: str) -> list:
    """Retourne toutes les colonnes numériques utilisables comme features."""
    excl_kw = ML["ml_exclude_keywords"]
    return [
        c for c in df.select_dtypes(include=np.number).columns
        if not any(k in c for k in excl_kw) and c != target
    ]


def detect_problem_type(y: pd.Series) -> str:
    """Détecte automatiquement si c'est classification ou régression."""
    threshold = ML["classification_threshold"]
    return "classification" if y.nunique() <= threshold else "regression"


def train_model(df: pd.DataFrame, target: str,
                selected_features: list = None) -> dict:
    """
    Entraîne un modèle ML automatiquement.

    Args:
        df                : DataFrame transformé
        target            : Colonne cible (ce qu'on veut prédire)
        selected_features : Liste de features choisies par l'utilisateur
                            Si None → toutes les features disponibles

    Returns:
        dict avec model, score, importances, prob_type
    """
    # Toutes les features disponibles (hors target)
    all_features = get_ml_features(df, target)

    if len(all_features) == 0:
        raise ValueError("Pas assez de colonnes numeriques pour entrainer un modele.")

    # Utiliser les features sélectionnées ou toutes par défaut
    if selected_features and len(selected_features) > 0:
        # Garder seulement celles qui existent dans all_features
        feature_cols = [f for f in selected_features if f in all_features]
        if len(feature_cols) == 0:
            feature_cols = all_features
    else:
        feature_cols = all_features

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    prob_type = detect_problem_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = ML["test_size"],
        random_state = ML["random_state"]
    )

    # Choix du modèle
    if prob_type == "classification":
        model = RandomForestClassifier(
            n_estimators = ML["n_estimators"],
            random_state = ML["random_state"]
        )
    else:
        model = RandomForestRegressor(
            n_estimators = ML["n_estimators"],
            random_state = ML["random_state"]
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Importance des features
    importances = pd.DataFrame({
        "Feature"   : feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    # Score
    if prob_type == "classification":
        score = {
            "type"    : "classification",
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        }
    else:
        score = {
            "type": "regression",
            "mae" : round(mean_absolute_error(y_test, y_pred), 4),
            "r2"  : round(r2_score(y_test, y_pred) * 100, 2),
        }

    return {
        "model"      : model,
        "score"      : score,
        "importances": importances,
        "prob_type"  : prob_type,
        "features"   : feature_cols,
        "target"     : target,
        "n_train"    : len(X_train),
        "n_test"     : len(X_test),
    }
