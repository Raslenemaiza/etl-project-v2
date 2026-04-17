# ══════════════════════════════════════════════════════════════════════
# utils/rapport_pdf.py — Génération du rapport PDF
# ══════════════════════════════════════════════════════════════════════

import io
import datetime
import pandas as pd
from fpdf import FPDF

from config.settings import UI, EXPORT


def _clean(text: str) -> str:
    """Nettoie le texte pour Helvetica — remplace les caractères non supportés."""
    replacements = {
        "\u2014": "-",   # — tiret long
        "\u2013": "-",   # – tiret moyen
        "\u2019": "'",   # ' apostrophe courbe
        "\u2018": "'",   # ' apostrophe gauche
        "\u201c": '"',   # " guillemet gauche
        "\u201d": '"',   # " guillemet droit
        "\u00e0": "a",   # à
        "\u00e9": "e",   # é
        "\u00e8": "e",   # è
        "\u00ea": "e",   # ê
        "\u00eb": "e",   # ë
        "\u00e2": "a",   # â
        "\u00f4": "o",   # ô
        "\u00fb": "u",   # û
        "\u00ee": "i",   # î
        "\u00ef": "i",   # ï
        "\u00e7": "c",   # ç
        "\u00f9": "u",   # ù
        "\u00fc": "u",   # ü
        "\u00e4": "a",   # ä
        "\u00f6": "o",   # ö
        "\u00e1": "a",   # á
        "\u00ed": "i",   # í
        "\u00f3": "o",   # ó
        "\u00fa": "u",   # ú
        "\u00f1": "n",   # ñ
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def generer_pdf(df_raw: pd.DataFrame, df: pd.DataFrame,
                rapport: list, filename: str) -> io.BytesIO:
    """
    Génère un rapport PDF complet du pipeline ETL.

    Args:
        df_raw   : DataFrame original (avant ETL)
        df       : DataFrame transformé (après ETL)
        rapport  : Liste des logs du pipeline
        filename : Nom du fichier source

    Returns:
        BytesIO contenant le PDF
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    r, g, b = EXPORT["pdf_header_color"]

    # ── En-tête ──────────────────────────────────────────────────────
    pdf.set_fill_color(r, g, b)
    pdf.rect(0, 0, 210, 35, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(10, 8)
    pdf.cell(0, 10, "RAPPORT ETL AUTOMATIQUE", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(10, 22)
    date_str = datetime.datetime.now().strftime("%d/%m/%Y a %H:%M")
    pdf.cell(0, 8, f"Genere le : {date_str}", ln=True, align="C")

    pdf.set_text_color(0, 0, 0)

    # ── Section 1 : Informations dataset ─────────────────────────────
    _section_title(pdf, "1. INFORMATIONS DATASET", y=45)

    score = df.get("completeness_score", pd.Series([100])).mean()
    infos = [
        ("Fichier"            , _clean(filename)),
        ("Lignes originales"  , f"{len(df_raw):,}"),
        ("Lignes finales"     , f"{len(df):,}"),
        ("Lignes supprimees"  , f"{len(df_raw) - len(df):,}"),
        ("Colonnes originales", f"{df_raw.shape[1]}"),
        ("Colonnes finales"   , f"{df.shape[1]}"),
        ("Score completude"   , f"{score:.1f}%"),
    ]
    for label, val in infos:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(80, 7, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(100, 7, str(val), ln=True)

    # ── Section 2 : Transformations ───────────────────────────────────
    pdf.ln(4)
    _section_title(pdf, "2. TRANSFORMATIONS APPLIQUEES")

    pdf.set_font("Helvetica", "", 10)
    for i, item in enumerate(rapport):
        item_c = _clean(item)
        pdf.set_x(15)
        pdf.cell(190, 7, f"{i+1}. {item_c}", ln=True)

    # ── Section 3 : Audit qualité ─────────────────────────────────────
    pdf.ln(4)
    _section_title(pdf, "3. AUDIT QUALITE PAR COLONNE")

    # En-tête tableau
    hr, hg, hb = EXPORT["pdf_table_header"]
    pdf.set_fill_color(hr, hg, hb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_x(15)
    pdf.cell(70, 7, "Colonne"   , fill=True, border=1)
    pdf.cell(35, 7, "Type"      , fill=True, border=1)
    pdf.cell(40, 7, "Manquants" , fill=True, border=1)
    pdf.cell(40, 7, "Pct %"     , fill=True, border=1, ln=True)

    # Lignes tableau
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 8)
    missing     = df_raw.isnull().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    ar, ag, ab  = EXPORT["pdf_row_alt"]

    for i, col in enumerate(df_raw.columns):
        fill = i % 2 == 0
        if fill:
            pdf.set_fill_color(ar, ag, ab)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_x(15)
        col_c = _clean(col)
        pdf.cell(70, 6, col_c[:28]                  , fill=fill, border=1)
        pdf.cell(35, 6, str(df_raw[col].dtype)      , fill=fill, border=1)
        pdf.cell(40, 6, str(missing[col])            , fill=fill, border=1)
        pdf.cell(40, 6, f"{missing_pct[col]:.2f}%"  , fill=fill, border=1, ln=True)

    # ── Pied de page ─────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(
        190, 9,
        f"  Pipeline ETL Automatique v{UI['version']} - Python & Streamlit",
        fill=True, ln=True, align="C"
    )

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


def _section_title(pdf: FPDF, title: str, y: float = None):
    """Helper : affiche un titre de section."""
    if y:
        pdf.set_xy(10, y)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(190, 10, f"  {title}", ln=True, fill=True)
    pdf.ln(3)