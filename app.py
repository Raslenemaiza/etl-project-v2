# ══════════════════════════════════════════════════════════════════════
# app.py — Interface Streamlit principale (propre et maintenable)
# ══════════════════════════════════════════════════════════════════════

import streamlit as st

from config.settings import UI, ML
from etl.loader import load_file, dataframe_to_csv_bytes, dataframe_to_excel_bytes
from etl.pipeline import ETLPipeline
from ml.model import train_model, get_ml_features
from utils.visualisations import (
    plot_distributions_num, plot_distributions_cat,
    plot_correlation, plot_missing_values, plot_feature_importance
)
from utils.rapport_pdf import generer_pdf

# ── Configuration page ─────────────────────────────────────────────────
st.set_page_config(
    page_title = UI["page_title"],
    page_icon  = UI["page_icon"],
    layout     = UI["layout"]
)

# ── Style CSS ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main-title {{
        font-size: 2.5rem; font-weight: bold;
        color: {UI['primary_color']}; text-align: center; margin-bottom: 0.5rem;
    }}
    .sub-title {{
        text-align: center; color: #666; margin-bottom: 2rem;
    }}
    .metric-card {{
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; text-align: center;
        border-left: 4px solid {UI['primary_color']};
    }}
</style>
""", unsafe_allow_html=True)

# ── Titre ──────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Pipeline ETL Automatique</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload ton dataset — le pipeline fait tout automatiquement</div>',
    unsafe_allow_html=True
)

# ── Session state ──────────────────────────────────────────────────────
if "etl" not in st.session_state:
    st.session_state.etl         = None
if "transformed" not in st.session_state:
    st.session_state.transformed = False

# ── Sidebar navigation ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", [
        "📁 Upload Dataset",
        "🔍 Audit Qualité",
        "⚙️ ETL Transformation",
        "📊 Visualisations",
        "🤖 Modèle ML",
        "📄 Rapport PDF",
    ])
    st.markdown("---")
    st.markdown(f"**ETL Automatique v{UI['version']}**")
    st.markdown("Pipeline IA/ML — Python & Streamlit")


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — UPLOAD
# ══════════════════════════════════════════════════════════════════════
if page == "📁 Upload Dataset":
    st.header("Upload ton Dataset")

    uploaded = st.file_uploader(
        "Glisse ton fichier ici",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded:
        try:
            df = load_file(uploaded)
            st.session_state.etl         = ETLPipeline(df, filename=uploaded.name)
            st.session_state.transformed = False

            st.success("Dataset chargé avec succès !")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes"   , f"{df.shape[0]:,}")
            col2.metric("Colonnes" , f"{df.shape[1]}")
            col3.metric("Manquants", f"{df.isnull().sum().sum():,}")
            col4.metric("Doublons" , f"{df.duplicated().sum():,}")

            st.subheader("Aperçu des données")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")
    else:
        st.info("Uploade un fichier CSV ou Excel pour commencer !")


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — AUDIT
# ══════════════════════════════════════════════════════════════════════
elif page == "🔍 Audit Qualité":
    st.header("Audit Qualité des Données")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    else:
        etl      = st.session_state.etl
        audit_df = etl.audit()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Rapport des valeurs manquantes")
            st.dataframe(audit_df, use_container_width=True)

        with col2:
            st.subheader("Visualisation")
            plot_missing_values(etl.df_raw)

        st.subheader("Types de colonnes détectés automatiquement")
        type_info = etl._detecter_types()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Colonnes ID"         , len(type_info["id_cols"]))
        c2.metric("Colonnes Date"       , len(type_info["date_cols"]))
        c3.metric("Colonnes Numériques" , len(type_info["num_cols"]))
        c4.metric("Colonnes Catégorielles", len(type_info["cat_cols"]))

        st.subheader("Statistiques générales")
        st.dataframe(etl.df_raw.describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — ETL TRANSFORMATION
# ══════════════════════════════════════════════════════════════════════
elif page == "⚙️ ETL Transformation":
    st.header("ETL — Transformations Automatiques")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    else:
        etl = st.session_state.etl

        if not st.session_state.transformed:
            st.info("Clique sur le bouton pour lancer toutes les transformations !")
            if st.button("🚀 Lancer le Pipeline ETL", type="primary", use_container_width=True):
                with st.spinner("Transformation en cours..."):
                    log = etl.transform()
                    st.session_state.transformed = True
                st.success("Pipeline ETL terminé !")
                for item in log:
                    st.write(f"✅ {item}")
                if etl.errors:
                    with st.expander("⚠️ Erreurs rencontrées"):
                        for err in etl.errors:
                            st.error(err)
        else:
            st.success("Pipeline ETL déjà exécuté !")
            for item in etl.rapport:
                st.write(f"✅ {item}")

        if st.session_state.transformed:
            summary = etl.get_summary()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes originales", f"{summary['lignes_originales']:,}")
            col2.metric("Lignes finales"   , f"{summary['lignes_finales']:,}")
            col3.metric("Colonnes finales" , f"{summary['colonnes_finales']}")
            col4.metric("Score complétude" , f"{summary['score_completude']}%")

            st.subheader("Dataset transformé")
            st.dataframe(etl.df.head(10), use_container_width=True)

            col_csv, col_excel = st.columns(2)
            with col_csv:
                st.download_button(
                    "⬇️ Télécharger CSV",
                    data           = dataframe_to_csv_bytes(etl.df),
                    file_name      = f"{etl.filename}_ETL.csv",
                    mime           = "text/csv",
                    use_container_width=True
                )
            with col_excel:
                st.download_button(
                    "⬇️ Télécharger Excel",
                    data           = dataframe_to_excel_bytes(etl.df, etl.rapport),
                    file_name      = f"{etl.filename}_ETL.xlsx",
                    mime           = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 Visualisations":
    st.header("Visualisations Automatiques")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl = st.session_state.etl

        st.subheader("Distribution — Colonnes numériques")
        plot_distributions_num(etl.df)

        st.subheader("Distribution — Colonnes catégorielles")
        plot_distributions_cat(etl.df)

        st.subheader("Matrice de corrélation")
        plot_correlation(etl.df)


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — MODÈLE ML
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 Modèle ML":
    st.header("Modèle ML Automatique")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl      = st.session_state.etl
        num_cols = get_ml_features(etl.df, target="")

        if len(num_cols) < 2:
            st.error("Pas assez de colonnes numériques pour le ML !")
        else:
            # Toutes les colonnes numériques disponibles comme cible possible
            all_num = [c for c in etl.df.select_dtypes(include="object").__class__
                       if True] or num_cols
            target = st.selectbox(
                "Choisis la colonne cible (ce que tu veux prédire)",
                num_cols
            )

            st.info(f"🔎 Seuil classification/régression : ≤ {ML['classification_threshold']} valeurs uniques")

            if st.button("🚀 Entraîner le Modèle ML", type="primary", use_container_width=True):
                try:
                    with st.spinner("Entraînement en cours..."):
                        result = train_model(etl.df, target)

                    prob_type   = result["prob_type"]
                    score       = result["score"]
                    importances = result["importances"]

                    st.success(f"Modèle entraîné ! Type : **{prob_type.upper()}**")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Lignes entraînement", f"{result['n_train']:,}")
                    col2.metric("Lignes test"        , f"{result['n_test']:,}")

                    if prob_type == "classification":
                        col3.metric("Accuracy", f"{score['accuracy']:.2f}%")
                    else:
                        col3.metric("R² Score", f"{score['r2']:.2f}%")
                        st.metric("MAE", f"{score['mae']:.4f}")

                    st.subheader("Importance des features")
                    col_chart, col_table = st.columns(2)
                    with col_chart:
                        plot_feature_importance(importances, target)
                    with col_table:
                        st.dataframe(importances, use_container_width=True)

                except Exception as e:
                    st.error(f"Erreur lors de l'entraînement : {e}")


# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — RAPPORT PDF
# ══════════════════════════════════════════════════════════════════════
elif page == "📄 Rapport PDF":
    st.header("Rapport PDF Automatique")

    if st.session_state.etl is None:
        st.warning("Uploade d'abord un dataset !")
    elif not st.session_state.transformed:
        st.warning("Lance d'abord le Pipeline ETL !")
    else:
        etl = st.session_state.etl
        st.info("Clique pour générer et télécharger ton rapport PDF complet !")

        if st.button("📄 Générer le Rapport PDF", type="primary", use_container_width=True):
            try:
                with st.spinner("Génération du PDF..."):
                    pdf_buf = generer_pdf(
                        df_raw   = etl.df_raw,
                        df       = etl.df,
                        rapport  = etl.rapport,
                        filename = etl.filename
                    )
                st.success("Rapport PDF généré !")
                st.download_button(
                    "⬇️ Télécharger le Rapport PDF",
                    data      = pdf_buf,
                    file_name = f"Rapport_ETL_{etl.filename}.pdf",
                    mime      = "application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Erreur lors de la génération PDF : {e}")
