# ══════════════════════════════════════════════════════════════════════
# etl/pipeline.py — Classe ETL complète et améliorée
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from config.settings import ETL


class ETLPipeline:
    """
    Pipeline ETL universel et auto-adaptatif.
    Fonctionne avec n'importe quel dataset CSV ou Excel.
    """

    def __init__(self, df: pd.DataFrame, filename: str = "dataset"):
        self.df_raw   = df.copy()
        self.df       = df.copy()
        self.filename = filename
        self.rapport  = []
        self.errors   = []

        # Colonnes détectées automatiquement
        self.id_cols   = []
        self.date_cols = []
        self.num_cols  = []
        self.cat_cols  = []

    # ══════════════════════════════════════════════════════════════════
    # DÉTECTION AUTOMATIQUE DES TYPES
    # ══════════════════════════════════════════════════════════════════
    def _detecter_types(self):
        """Détecte automatiquement les types de colonnes."""

        # Colonnes ID
        kw = ETL["id_keywords"]
        self.id_cols = [
            c for c in self.df.columns
            if any(
                k == c.lower()
                or c.lower().startswith(k + "_")
                or c.lower().endswith("_" + k)
                for k in kw
            )
        ]

        # Colonnes Date — détection par nom ET vérification par contenu
        date_kw = ETL["date_keywords"]
        potential_dates = [
            c for c in self.df.select_dtypes(include="object").columns
            if any(k in c.lower() for k in date_kw)
        ]
        self.date_cols = []
        for col in potential_dates:
            try:
                parsed = pd.to_datetime(
                    self.df[col].dropna().head(20),
                    dayfirst=True, errors="coerce"
                )
                if parsed.notna().sum() >= 5:
                    self.date_cols.append(col)
            except Exception:
                pass

        # Colonnes numériques (hors ID)
        self.num_cols = [
            c for c in self.df.select_dtypes(include=np.number).columns
            if c not in self.id_cols
        ]

        # Colonnes catégorielles (hors ID et dates)
        self.cat_cols = [
            c for c in self.df.select_dtypes(include="object").columns
            if c not in self.id_cols and c not in self.date_cols
        ]

        return {
            "id_cols"   : self.id_cols,
            "date_cols" : self.date_cols,
            "num_cols"  : self.num_cols,
            "cat_cols"  : self.cat_cols,
        }

    # ══════════════════════════════════════════════════════════════════
    # AUDIT QUALITÉ
    # ══════════════════════════════════════════════════════════════════
    def audit(self) -> pd.DataFrame:
        """Retourne un rapport de qualité des données."""
        self._detecter_types()

        df_audit    = self.df.drop(columns=self.id_cols, errors="ignore")
        missing     = df_audit.isnull().sum()
        missing_pct = (missing / len(df_audit) * 100).round(2)

        audit_df = pd.DataFrame({
            "Manquants"    : missing,
            "Pourcentage %": missing_pct,
            "Type"         : df_audit.dtypes,
            "Uniques"      : df_audit.nunique(),
        }).sort_values("Manquants", ascending=False)

        return audit_df

    # ══════════════════════════════════════════════════════════════════
    # PIPELINE TRANSFORM — 9 ÉTAPES
    # ══════════════════════════════════════════════════════════════════
    def transform(self) -> list:
        """Lance toutes les transformations et retourne le log."""
        self._detecter_types()
        log = []

        # ── T1 : Suppression doublons ──────────────────────────────
        try:
            before = len(self.df)
            self.df.drop_duplicates(inplace=True)
            n = before - len(self.df)
            msg = f"T1 - Doublons supprimés : {n}"
            log.append(msg)
        except Exception as e:
            self.errors.append(f"T1 erreur : {e}")

        # ── T2 : Conversion texte → nombre ────────────────────────
        try:
            n_conv = 0
            threshold = ETL["numeric_conversion_threshold"]
            for col in self.df.select_dtypes(include="object").columns:
                if col in self.id_cols or col in self.date_cols:
                    continue
                try:
                    converted = pd.to_numeric(
                        self.df[col].str.replace(",", ".", regex=False),
                        errors="coerce"
                    )
                    if converted.notna().sum() / len(self.df) > threshold:
                        self.df[col] = converted
                        if col not in self.num_cols:
                            self.num_cols.append(col)
                        n_conv += 1
                except Exception:
                    pass
            log.append(f"T2 - Conversion types : {n_conv} colonnes texte → nombre")
        except Exception as e:
            self.errors.append(f"T2 erreur : {e}")

        # ── T3 : Nettoyage espaces + normalisation casse ───────────
        try:
            cols_to_clean = [
                c for c in self.df.select_dtypes(include="object").columns
                if c not in self.id_cols and c not in self.date_cols
            ]
            for col in cols_to_clean:
                self.df[col] = self.df[col].astype(str).str.strip().str.upper()
            log.append(f"T3 - Nettoyage espaces : {len(cols_to_clean)} colonnes normalisées")
        except Exception as e:
            self.errors.append(f"T3 erreur : {e}")

        # ── T4 : Imputation valeurs manquantes ────────────────────
        try:
            fill_val = ETL["missing_text_fill"]
            n_total  = 0
            for col in self.df.columns:
                if col in self.id_cols:
                    continue
                n_miss  = self.df[col].isin(["nan", "NaN", "None", ""]).sum()
                n_miss += self.df[col].isnull().sum()
                if n_miss > 0:
                    if self.df[col].dtype == "object":
                        self.df[col] = self.df[col].replace(
                            {"nan": fill_val, "NaN": fill_val,
                             "None": fill_val, "": fill_val}
                        ).fillna(fill_val)
                    else:
                        self.df[col] = self.df[col].fillna(self.df[col].median())
                    n_total += n_miss
            log.append(f"T4 - Imputation : {n_total} valeurs traitées")
        except Exception as e:
            self.errors.append(f"T4 erreur : {e}")

        # ── T5 : Validation métier (valeurs négatives / ages) ─────
        try:
            n_invalid = 0
            pos_kw    = ETL["positive_keywords"]
            for col in self.num_cols:
                if col not in self.df.columns:
                    continue
                if any(k in col.lower() for k in pos_kw):
                    neg_mask = self.df[col] < 0
                    if neg_mask.sum() > 0:
                        self.df.loc[neg_mask, col] = self.df[col].median()
                        n_invalid += neg_mask.sum()
                if "age" in col.lower():
                    age_max = ETL["age_max"]
                    invalid = (self.df[col] < 0) | (self.df[col] > age_max)
                    if invalid.sum() > 0:
                        self.df.loc[invalid, col] = self.df[col].median()
                        n_invalid += invalid.sum()
            log.append(f"T5 - Validation métier : {n_invalid} valeurs corrigées")
        except Exception as e:
            self.errors.append(f"T5 erreur : {e}")

        # ── T6 : Conversion des dates + feature engineering ───────
        try:
            n_dates = 0
            for col in self.date_cols:
                try:
                    self.df[col] = pd.to_datetime(
                        self.df[col], dayfirst=True, errors="coerce"
                    )
                    prefix = (col.lower()
                                 .replace("date", "")
                                 .replace("time", "")
                                 .strip("_") or col.lower())
                    self.df[f"year_{prefix}"]       = self.df[col].dt.year.astype("Int64")
                    self.df[f"month_{prefix}"]      = self.df[col].dt.month.astype("Int64")
                    self.df[f"day_{prefix}"]        = self.df[col].dt.day.astype("Int64")
                    self.df[f"quarter_{prefix}"]    = self.df[col].dt.quarter.astype("Int64")
                    self.df[f"month_name_{prefix}"] = self.df[col].dt.strftime("%B")
                    self.df[f"dayofweek_{prefix}"]  = self.df[col].dt.strftime("%A")
                    self.df[f"is_weekend_{prefix}"] = self.df[col].dt.dayofweek.isin(
                        [5, 6]).astype(int)
                    n_dates += 1
                except Exception:
                    pass
            log.append(f"T6 - Dates : {n_dates} colonnes → 7 features chacune")
        except Exception as e:
            self.errors.append(f"T6 erreur : {e}")

        # ── T7 : Détection outliers (IQR) ─────────────────────────
        try:
            factor  = ETL["outlier_iqr_factor"]
            min_pct = ETL["outlier_min_pct"]
            n_out   = 0
            for col in self.num_cols:
                if col not in self.df.columns:
                    continue
                Q1   = self.df[col].quantile(0.25)
                Q3   = self.df[col].quantile(0.75)
                IQR  = Q3 - Q1
                mask = (
                    (self.df[col] < Q1 - factor * IQR) |
                    (self.df[col] > Q3 + factor * IQR)
                )
                # Flaguer seulement si > seuil minimum
                if mask.sum() / len(self.df) > min_pct:
                    self.df[f"{col}_outlier"] = mask.astype(int)
                    n_out += mask.sum()
            log.append(f"T7 - Outliers flagués : {n_out}")
        except Exception as e:
            self.errors.append(f"T7 erreur : {e}")

        # ── T8 : Encodage ML adaptatif ────────────────────────────
        try:
            le          = LabelEncoder()
            max_unique  = ETL["encoding_max_unique"]
            excl_kw     = ETL["encoding_exclude_keywords"]
            n_encoded   = 0
            cols_encode = [
                c for c in self.df.select_dtypes(include="object").columns
                if c not in self.id_cols
                and not any(k in c.lower() for k in excl_kw)
                and self.df[c].nunique() < max_unique
            ]
            for col in cols_encode:
                self.df[f"{col}_encoded"] = le.fit_transform(
                    self.df[col].astype(str)
                )
                n_encoded += 1
            log.append(f"T8 - Encodage ML : {n_encoded} colonnes encodées")
        except Exception as e:
            self.errors.append(f"T8 erreur : {e}")

        # ── T9 : Score de complétude ──────────────────────────────
        try:
            max_cols     = ETL["completeness_max_cols"]
            fill_val     = ETL["missing_text_fill"]
            quality_cols = [
                c for c in self.cat_cols if c in self.df.columns
            ][:max_cols]

            if quality_cols:
                self.df["completeness_score"] = (
                    self.df[quality_cols].apply(lambda row: sum(
                        1 for v in row
                        if pd.notna(v) and str(v) not in [fill_val, "nan", ""]
                    ), axis=1) / len(quality_cols) * 100
                ).round(1)
            else:
                self.df["completeness_score"] = 100.0

            avg = self.df["completeness_score"].mean()
            log.append(f"T9 - Score complétude : {avg:.1f}% en moyenne")
        except Exception as e:
            self.errors.append(f"T9 erreur : {e}")

        self.rapport = log
        return log

    # ══════════════════════════════════════════════════════════════════
    # RÉSUMÉ
    # ══════════════════════════════════════════════════════════════════
    def get_summary(self) -> dict:
        """Retourne un dictionnaire résumé du pipeline."""
        return {
            "fichier"            : self.filename,
            "lignes_originales"  : len(self.df_raw),
            "lignes_finales"     : len(self.df),
            "colonnes_originales": self.df_raw.shape[1],
            "colonnes_finales"   : self.df.shape[1],
            "score_completude"   : round(self.df.get("completeness_score",
                                   pd.Series([100])).mean(), 1),
            "erreurs"            : self.errors,
            "rapport"            : self.rapport,
        }
