# ══════════════════════════════════════════════════════════════════════
# config/settings.py — Tous les paramètres centralisés (plus de hard code !)
# ══════════════════════════════════════════════════════════════════════

# ── Apparence ──────────────────────────────────────────────────────────
UI = {
    "primary_color"    : "#E50914",
    "secondary_color"  : "#3498DB",
    "accent_color"     : "#9B59B6",
    "success_color"    : "#2ECC71",
    "warning_color"    : "#F39C12",
    "page_title"       : "ETL Automatique",
    "page_icon"        : "🔧",
    "layout"           : "wide",
    "version"          : "2.0",
}

# ── ETL Pipeline ───────────────────────────────────────────────────────
ETL = {
    # Encodages à tester pour la détection automatique
    "encodings"            : ["utf-8", "latin-1", "cp1252"],

    # Mots-clés pour détecter les colonnes ID
    "id_keywords"          : ["id", "_id", "code", "ref", "num", "no", "number", "key"],

    # Mots-clés pour détecter les colonnes dates
    "date_keywords"        : ["date", "time", "created", "updated", "born",
                              "start", "end", "added", "at", "day", "month", "year"],

    # Mots-clés pour détecter les colonnes prix
    "price_keywords"       : ["price", "prix", "cost", "rate", "tarif", "unit"],

    # Mots-clés pour détecter les colonnes quantité
    "qty_keywords"         : ["qty", "quantity", "units", "sold", "qte", "nb", "count"],

    # Mots-clés pour détecter les colonnes total
    "total_keywords"       : ["total", "revenue", "sales", "amount", "sum", "turnover"],

    # Mots-clés pour validation (valeurs ne peuvent pas être négatives)
    "positive_keywords"    : ["price", "prix", "revenue", "salary", "salaire",
                              "cost", "montant", "amount", "total"],

    # Seuil outlier IQR
    "outlier_iqr_factor"   : 1.5,

    # Seuil minimum % outliers pour les flaguer
    "outlier_min_pct"      : 0.05,

    # Seuil max valeurs uniques pour encodage LabelEncoder
    "encoding_max_unique"  : 50,

    # Seuil % de conversion réussie pour considérer une colonne comme numérique
    "numeric_conversion_threshold": 0.8,

    # Valeur de remplacement pour les valeurs manquantes texte
    "missing_text_fill"    : "UNKNOWN",

    # Colonnes à exclure de l'encodage (par mot-clé)
    "encoding_exclude_keywords": ["name", "dayofweek"],

    # Nombre max de colonnes pour le score de complétude
    "completeness_max_cols": 6,

    # Age maximum autorisé (validation)
    "age_max"              : 150,
}

# ── Machine Learning ───────────────────────────────────────────────────
ML = {
    # Seuil de valeurs uniques pour décider classification vs régression
    "classification_threshold" : 10,

    # Random Forest
    "n_estimators"             : 100,
    "random_state"             : 42,
    "test_size"                : 0.2,

    # Colonnes à exclure des features ML
    "ml_exclude_keywords"      : ["_outlier", "_encoded", "completeness_score"],
}

# ── Visualisations ─────────────────────────────────────────────────────
VIZ = {
    # Nombre max de colonnes à afficher dans les graphiques
    "max_num_cols"     : 3,
    "max_cat_cols"     : 3,
    "max_cat_unique"   : 20,

    # Nombre max de valeurs dans un bar chart catégoriel
    "max_bar_values"   : 8,

    # Taille des figures
    "fig_size_small"   : (5, 3),
    "fig_size_medium"  : (8, 4),
    "fig_size_large"   : (10, 6),

    # Nombre de bins pour les histogrammes
    "hist_bins"        : 30,
}

# ── Export / Rapport ───────────────────────────────────────────────────
EXPORT = {
    "pdf_header_color"  : (229, 9, 20),   # RGB rouge
    "pdf_table_header"  : (52, 73, 94),   # RGB bleu foncé
    "pdf_row_alt"       : (245, 245, 245),
    "sheet_data"        : "Data_Cleaned",
    "sheet_rapport"     : "Rapport_ETL",
}
