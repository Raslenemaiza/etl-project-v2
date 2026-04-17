# ══════════════════════════════════════════════════════════════════════
# etl/loader.py — Lecture universelle de fichiers (CSV, Excel)
# ══════════════════════════════════════════════════════════════════════

import pandas as pd
import io
from config.settings import ETL


def load_file(uploaded_file) -> pd.DataFrame:
    """
    Charge un fichier uploadé via Streamlit.
    Détecte automatiquement : format, encodage, séparateur.

    Args:
        uploaded_file : objet retourné par st.file_uploader()

    Returns:
        pd.DataFrame
    """
    ext = uploaded_file.name.split(".")[-1].lower()

    # ── Excel ──────────────────────────────────────────────────────
    if ext in ["xlsx", "xls"]:
        engine = "openpyxl" if ext == "xlsx" else "xlrd"
        return pd.read_excel(uploaded_file, engine=engine)

    # ── CSV — détection auto encodage + séparateur ─────────────────
    encodings = ETL["encodings"]
    sample    = uploaded_file.read(4000).decode("utf-8", errors="ignore")
    uploaded_file.seek(0)

    # Détection séparateur
    sep = ";" if sample.count(";") > sample.count(",") else ","
    # Vérifier tabulation
    if sample.count("\t") > max(sample.count(";"), sample.count(",")):
        sep = "\t"

    # Essai des encodages
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(
                uploaded_file,
                sep=sep,
                encoding=enc,
                engine="python",
                on_bad_lines="skip"
            )
            uploaded_file.seek(0)
            return df
        except Exception:
            uploaded_file.seek(0)
            continue

    raise ValueError(
        f"Impossible de lire le fichier '{uploaded_file.name}'. "
        f"Encodages essayés : {encodings}"
    )


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en bytes CSV (UTF-8 BOM pour Excel)."""
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def dataframe_to_excel_bytes(df: pd.DataFrame, rapport: list) -> bytes:
    """Convertit un DataFrame en bytes Excel avec onglet rapport."""
    buf = io.BytesIO()
    from config.settings import EXPORT
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=EXPORT["sheet_data"])
        pd.DataFrame({"Rapport ETL": rapport}).to_excel(
            writer, index=False, sheet_name=EXPORT["sheet_rapport"]
        )
    buf.seek(0)
    return buf
