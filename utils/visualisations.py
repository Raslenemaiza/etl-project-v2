# ══════════════════════════════════════════════════════════════════════
# utils/visualisations.py — Graphiques automatiques
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from config.settings import UI, VIZ


def get_num_cols(df: pd.DataFrame) -> list:
    """Retourne les colonnes numériques pures (sans colonnes techniques)."""
    excl = ["_outlier", "_encoded", "completeness_score"]
    return [
        c for c in df.select_dtypes(include=np.number).columns
        if not any(k in c for k in excl)
    ]


def get_cat_cols(df: pd.DataFrame) -> list:
    """Retourne les colonnes catégorielles exploitables."""
    max_u = VIZ["max_cat_unique"]
    return [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique() < max_u
    ]


def plot_distributions_num(df: pd.DataFrame):
    """Histogrammes pour les colonnes numériques."""
    num_cols = get_num_cols(df)
    max_cols = VIZ["max_num_cols"]
    n        = min(len(num_cols), max_cols)

    if n == 0:
        st.info("Aucune colonne numérique disponible.")
        return

    cols = st.columns(n)
    for i, col in enumerate(num_cols[:n]):
        with cols[i]:
            fig, ax = plt.subplots(figsize=VIZ["fig_size_small"])
            ax.hist(
                df[col].dropna(),
                bins       = VIZ["hist_bins"],
                color      = UI["secondary_color"],
                edgecolor  = "white"
            )
            ax.axvline(
                df[col].mean(),
                color     = UI["primary_color"],
                linestyle = "--",
                label     = f"Moy: {df[col].mean():.1f}"
            )
            ax.set_title(col, fontsize=10)
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


def plot_distributions_cat(df: pd.DataFrame):
    """Bar charts pour les colonnes catégorielles."""
    cat_cols = get_cat_cols(df)
    max_cols = VIZ["max_cat_cols"]
    max_bars = VIZ["max_bar_values"]
    n        = min(len(cat_cols), max_cols)

    if n == 0:
        st.info("Aucune colonne catégorielle disponible.")
        return

    cols = st.columns(n)
    for i, col in enumerate(cat_cols[:n]):
        with cols[i]:
            fig, ax = plt.subplots(figsize=VIZ["fig_size_small"])
            top = df[col].value_counts().head(max_bars)
            ax.barh(top.index[::-1], top.values[::-1], color=UI["accent_color"])
            ax.set_title(col, fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


def plot_correlation(df: pd.DataFrame):
    """Matrice de corrélation."""
    num_cols = get_num_cols(df)

    if len(num_cols) < 2:
        st.info("Pas assez de colonnes numériques pour la corrélation.")
        return

    fig, ax = plt.subplots(figsize=VIZ["fig_size_large"])
    corr    = df[num_cols].corr()
    sns.heatmap(
        corr,
        annot      = True,
        fmt        = ".2f",
        cmap       = "coolwarm",
        ax         = ax,
        linewidths = 0.5
    )
    ax.set_title("Matrice de Corrélation", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_missing_values(df_raw: pd.DataFrame):
    """Bar chart horizontal des valeurs manquantes."""
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        st.success("✅ Aucune valeur manquante !")
        return

    fig, ax = plt.subplots(figsize=VIZ["fig_size_medium"])
    ax.barh(missing.index, missing.values, color=UI["primary_color"])
    ax.set_xlabel("Nombre de valeurs manquantes")
    ax.set_title("Valeurs manquantes par colonne")
    for bar, val in zip(ax.patches, missing.values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val}", va="center", fontsize=9
        )
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def plot_feature_importance(importances: pd.DataFrame, target: str):
    """Bar chart de l'importance des features ML."""
    fig, ax = plt.subplots(figsize=VIZ["fig_size_medium"])
    ax.barh(
        importances["Feature"][::-1],
        importances["Importance"][::-1],
        color = UI["primary_color"]
    )
    ax.set_title(f"Prédiction : {target}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
