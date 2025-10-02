#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# -------------------------
# Config & Scopes
# -------------------------
SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/presentations.readonly",
]

DEFAULT_OUTPUT = "projects.json"
DEFAULT_EXCERPT_CHARS = 4000
SHEET_MAX_ROWS = 50   # límite de filas por hoja al leer Sheets
SHEET_MAX_COLS = 30   # límite de columnas

# -------------------------
# Auth Helpers
# -------------------------
def get_creds(client_secret_path: str = "client_secret.json", token_path: str = "token.json"):
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(client_secret_path):
                raise FileNotFoundError(f"No se encontró {client_secret_path}.")
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return creds

def get_services(creds):
    drive = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)
    slides = build("slides", "v1", credentials=creds)
    docs = build("docs", "v1", credentials=creds)
    return drive, sheets, slides, docs

# -------------------------
# URL Parsing
# -------------------------
GDRIVE_FILE_RE = re.compile(r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)")
GDOCS_DOC_RE   = re.compile(r"https://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)")
GSLIDES_RE     = re.compile(r"https://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)")
GSHEETS_RE     = re.compile(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)")
GFOLDER_RE     = re.compile(r"https://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)")
GID_RE         = re.compile(r"[?#&]gid=(\d+)")

def parse_spreadsheet_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    m = GSHEETS_RE.search(url)
    if not m:
        return None, None
    spreadsheet_id = m.group(1)
    gid = None
    mg = GID_RE.search(url)
    if mg:
        gid = mg.group(1)
    return spreadsheet_id, gid

def is_sheet_url(url: str) -> bool:
    return bool(GSHEETS_RE.search(url))

def is_slides_url(url: str) -> bool:
    return bool(GSLIDES_RE.search(url))

def normalize_url(val: Optional[str]) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s in {"-", "—", "N/A", "#N/A", ""}:
        return ""
    return s

# -------------------------
# Utils
# -------------------------
def col_idx_to_letters(n: int) -> str:
    # 1 -> A, 26 -> Z, 27 -> AA, ...
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def parse_number(s: str) -> Optional[float]:
    """
    Extrae el primer número de una cadena, tolerando miles y decimales con coma o punto.
    """
    if not s:
        return None
    m = re.search(r"[-+]?\d{1,3}(\.\d{3})*(,\d+)?|[-+]?\d+(\.\d+)?", s.replace(" ", " "))  # incluye thin space
    if not m:
        return None
    num = m.group(0)
    # normaliza: quita puntos de miles y cambia coma por punto
    num = num.replace(".", "").replace(",", ".")
    try:
        return float(num)
    except Exception:
        return None

# -------------------------
# Sheets Reading & CSV Fallback
# -------------------------
def fetch_sheet_grids(sheets_svc, spreadsheet_id: str, verbose: bool=False) -> List[Tuple[str, List[List[str]]]]:
    """
    Retorna lista de (title, values[][]) para cada hoja, limitado por SHEET_MAX_ROWS x SHEET_MAX_COLS.
    """
    grids = []
    try:
        meta = sheets_svc.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=False).execute()
        end_col = col_idx_to_letters(SHEET_MAX_COLS)
        for s in meta.get("sheets", []):
            title = s["properties"]["title"]
            rng = f"{title}!A1:{end_col}{SHEET_MAX_ROWS}"
            vals = sheets_svc.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng).execute().get("values", [])
            vals = [row[:SHEET_MAX_COLS] for row in vals[:SHEET_MAX_ROWS]] if vals else []
            if vals:
                grids.append((title, vals))
    except Exception as e:
        if verbose:
            print(f"[DEBUG] Sheets API error for {spreadsheet_id}: {repr(e)}")
    return grids

def export_sheet_to_csv_via_drive(drive_svc, spreadsheet_id: str) -> str:
    """Exporta la PRIMERA hoja de un Google Sheet a CSV usando Drive API (fallback)."""
    try:
        request = drive_svc.files().export(fileId=spreadsheet_id, mimeType="text/csv")
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        data = fh.read().decode("utf-8", errors="ignore")
        return data.strip()
    except Exception:
        return ""

def parse_kpis_from_grids(grids: List[Tuple[str, List[List[str]]]]) -> Dict[str, Optional[float]]:
    """
    Busca KPIs en los grids:
    - PVP (UF) / PVP / Precio
    - Meses / Duración
    - Fee CIS (%)
    - Costos Directos (%)
    - Margen Bruto (%)
    Devuelve dict con floats (None si no se halló).
    """
    kpis = {
        "pvp": None,
        "meses": None,
        "fee_cis_pct": None,
        "costos_directos_pct": None,
        "margen_bruto_pct": None,
    }
    # patrones por KPI (case-insensitive)
    patt = {
        "pvp": re.compile(r"\b(pvp|precio|valor)(?!\s*cis)\b", re.I),
        "meses": re.compile(r"\b(meses?|duraci[oó]n)\b", re.I),
        "fee_cis_pct": re.compile(r"\b(fee\s*cis|fee)\b", re.I),
        "costos_directos_pct": re.compile(r"\b(costos?\s*directos?)\b", re.I),
        "margen_bruto_pct": re.compile(r"\b(margen(\s*bruto)?)\b", re.I),
    }

    for _, vals in grids:
        for row in vals:
            joined = " ".join([str(c) for c in row if c is not None])
            for key, rx in patt.items():
                if rx.search(joined):
                    num = parse_number(joined)
                    if num is not None and kpis[key] is None:
                        kpis[key] = num
    return kpis

def kpis_to_summary(k: Dict[str, Optional[float]]) -> str:
    parts = []
    if k.get("pvp") is not None:
        parts.append(f"PVP≈{int(k['pvp']) if k['pvp'].is_integer() else k['pvp']}")
    if k.get("meses") is not None:
        parts.append(f"Meses≈{int(k['meses']) if k['meses'].is_integer() else k['meses']}")
    if k.get("fee_cis_pct") is not None:
        parts.append(f"Fee CIS≈{k['fee_cis_pct']}%")
    if k.get("costos_directos_pct") is not None:
        parts.append(f"Costos Directos≈{k['costos_directos_pct']}%")
    if k.get("margen_bruto_pct") is not None:
        parts.append(f"Margen Bruto≈{k['margen_bruto_pct']}%")
    return " | ".join(parts)

# -------------------------
# Drive helpers & text fetchers
# -------------------------
def fetch_text_from_docs(docs_svc, doc_id: str) -> str:
    doc = docs_svc.documents().get(documentId=doc_id).execute()
    content = doc.get("body", {}).get("content", [])
    chunks = []
    for elem in content:
        p = elem.get("paragraph")
        if not p:
            continue
        for r in p.get("elements", []):
            t = r.get("textRun", {}).get("content", "")
            if t:
                chunks.append(t)
    return "".join(chunks).strip()

def fetch_text_from_slides(slides_svc, pres_id: str) -> str:
    pres = slides_svc.presentations().get(presentationId=pres_id).execute()
    chunks = []
    for slide in pres.get("slides", []):
        for elem in slide.get("pageElements", []):
            shape = elem.get("shape")
            if not shape:
                continue
            for te in shape.get("text", {}).get("textElements", []):
                t = te.get("textRun", {}).get("content", "")
                if t:
                    chunks.append(t)
    return "\n".join(chunks).strip()

def drive_get_mime(drive_svc, file_id: str) -> Optional[str]:
    meta = drive_svc.files().get(fileId=file_id, fields="id,name,mimeType").execute()
    return meta.get("mimeType")

def drive_list_files_in_folder(drive_svc, folder_id: str, page_size: int = 50) -> List[Dict[str, Any]]:
    q = f"'{folder_id}' in parents and trashed = false"
    fields = "nextPageToken, files(id, name, mimeType)"
    resp = drive_svc.files().list(q=q, fields=fields, pageSize=page_size).execute()
    return resp.get("files", [])

# -------------------------
# High-level fetchers (econ/tec)
# -------------------------
def fetch_tecnica_text(drive_svc, sheets_svc, slides_svc, docs_svc, url: str, verbose: bool=False) -> Tuple[str, str]:
    """
    Devuelve (status, text) para Propuesta Técnica: Docs/Slides/Sheets → texto completo (Sheets como CSV string).
    """
    if not url:
        return "no_link", ""
    # Detecta por URL rápida
    if GDOCS_DOC_RE.search(url):
        try:
            txt = fetch_text_from_docs(docs_svc, GDOCS_DOC_RE.search(url).group(1))
            return ("ok" if txt else "parsed_empty", txt)
        except Exception as e:
            if verbose: print(f"[DEBUG] Docs read failed (TEC): {repr(e)}")
            return "unreachable", ""
    if GSLIDES_RE.search(url):
        try:
            txt = fetch_text_from_slides(slides_svc, GSLIDES_RE.search(url).group(1))
            return ("ok" if txt else "parsed_empty", txt)
        except Exception as e:
            if verbose: print(f"[DEBUG] Slides read failed (TEC): {repr(e)}")
            return "unreachable", ""
    if GSHEETS_RE.search(url):
        # Para técnica en Sheet: texto tipo CSV (por si acaso alguien sube la técnica como Sheet)
        try:
            grids = fetch_sheet_grids(sheets_svc, GSHEETS_RE.search(url).group(1), verbose=verbose)
            if grids:
                # render a CSV-ish string
                out = []
                for title, vals in grids:
                    out.append(f"[Hoja: {title}]")
                    for row in vals:
                        out.append(",".join([str(c) for c in row]))
                txt = "\n".join(out).strip()
                if not txt:
                    # fallback export
                    txt = export_sheet_to_csv_via_drive(drive_svc, GSHEETS_RE.search(url).group(1))
                return ("ok" if txt else "parsed_empty", txt)
            else:
                txt = export_sheet_to_csv_via_drive(drive_svc, GSHEETS_RE.search(url).group(1))
                return ("ok" if txt else "parsed_empty", txt)
        except Exception as e:
            if verbose: print(f"[DEBUG] Sheets read failed (TEC): {repr(e)}")
            try:
                txt = export_sheet_to_csv_via_drive(drive_svc, GSHEETS_RE.search(url).group(1))
                return ("ok" if txt else "parsed_empty", txt)
            except Exception:
                return "unreachable", ""
    # Folders o fileId → mira MIME
    m = GFOLDER_RE.search(url)
    if m:
        try:
            for f in drive_list_files_in_folder(drive_svc, m.group(1)):
                mime = f.get("mimeType","")
                fid  = f["id"]
                if mime == "application/vnd.google-apps.document":
                    txt = fetch_text_from_docs(docs_svc, fid); return ("ok" if txt else "parsed_empty", txt)
                if mime == "application/vnd.google-apps.presentation":
                    txt = fetch_text_from_slides(slides_svc, fid); return ("ok" if txt else "parsed_empty", txt)
                if mime == "application/vnd.google-apps.spreadsheet":
                    grids = fetch_sheet_grids(sheets_svc, fid, verbose=verbose)
                    out = []
                    for title, vals in grids:
                        out.append(f"[Hoja: {title}]")
                        for row in vals: out.append(",".join([str(c) for c in row]))
                    txt = "\n".join(out).strip()
                    if not txt: txt = export_sheet_to_csv_via_drive(drive_svc, fid)
                    return ("ok" if txt else "parsed_empty", txt)
            return "unsupported", ""
        except Exception as e:
            if verbose: print(f"[DEBUG] Folder read failed (TEC): {repr(e)}")
            return "unreachable", ""
    m = GDRIVE_FILE_RE.search(url)
    if m:
        fid = m.group(1)
        try:
            mime = drive_get_mime(drive_svc, fid)
            if mime == "application/vnd.google-apps.document":
                txt = fetch_text_from_docs(docs_svc, fid); return ("ok" if txt else "parsed_empty", txt)
            if mime == "application/vnd.google-apps.presentation":
                txt = fetch_text_from_slides(slides_svc, fid); return ("ok" if txt else "parsed_empty", txt)
            if mime == "application/vnd.google-apps.spreadsheet":
                grids = fetch_sheet_grids(sheets_svc, fid, verbose=verbose)
                out = []
                for title, vals in grids:
                    out.append(f"[Hoja: {title}]")
                    for row in vals: out.append(",".join([str(c) for c in row]))
                txt = "\n".join(out).strip()
                if not txt: txt = export_sheet_to_csv_via_drive(drive_svc, fid)
                return ("ok" if txt else "parsed_empty", txt)
            return "unsupported", ""
        except Exception as e:
            if verbose: print(f"[DEBUG] File read failed (TEC): {repr(e)}")
            return "unreachable", ""
    return "unsupported", ""

def fetch_economica(drive_svc, sheets_svc, slides_svc, docs_svc, url: str, verbose: bool=False) -> Tuple[str, str, Dict[str, Optional[float]]]:
    """
    Para Propuesta Económica:
      - Sheets: extrae KPIs (pvp, meses, fee_cis_pct, costos_directos_pct, margen_bruto_pct); NO devuelve texto masivo.
      - Slides/Docs: texto completo.
    Retorna (status, text_excerpt, kpis_dict).
    """
    if not url:
        return "no_link", "", {}

    if GSHEETS_RE.search(url):
        sid = GSHEETS_RE.search(url).group(1)
        try:
            grids = fetch_sheet_grids(sheets_svc, sid, verbose=verbose)
            if not grids:
                # fallback export + parse rudimentario
                csvdata = export_sheet_to_csv_via_drive(drive_svc, sid)
                if csvdata:
                    # convierte a grid simple
                    rows = [r.split(",") for r in csvdata.splitlines()]
                    grids = [("Export", rows[:SHEET_MAX_ROWS])]
            kpis = parse_kpis_from_grids(grids) if grids else {}
            summary = kpis_to_summary(kpis) if kpis else ""
            status = "ok" if (kpis or grids) else "parsed_empty"
            return status, summary, kpis
        except Exception as e:
            if verbose: print(f"[DEBUG] Sheets read failed (ECO): {repr(e)}")
            return "unreachable", "", {}

    if GSLIDES_RE.search(url):
        try:
            txt = fetch_text_from_slides(slides_svc, GSLIDES_RE.search(url).group(1))
            return ("ok" if txt else "parsed_empty", txt, {})
        except Exception as e:
            if verbose: print(f"[DEBUG] Slides read failed (ECO): {repr(e)}")
            return "unreachable", "", {}

    if GDOCS_DOC_RE.search(url):
        try:
            txt = fetch_text_from_docs(docs_svc, GDOCS_DOC_RE.search(url).group(1))
            return ("ok" if txt else "parsed_empty", txt, {})
        except Exception as e:
            if verbose: print(f"[DEBUG] Docs read failed (ECO): {repr(e)}")
            return "unreachable", "", {}

    # Carpetas o fileId → detecta MIME
    m = GFOLDER_RE.search(url)
    if m:
        try:
            files = drive_list_files_in_folder(drive_svc, m.group(1))
            for f in files:
                fid = f["id"]; mime = f.get("mimeType","")
                if mime == "application/vnd.google-apps.spreadsheet":
                    grids = fetch_sheet_grids(sheets_svc, fid, verbose=verbose)
                    kpis = parse_kpis_from_grids(grids) if grids else {}
                    summary = kpis_to_summary(kpis) if kpis else ""
                    return ("ok" if kpis else "parsed_empty", summary, kpis)
                if mime == "application/vnd.google-apps.presentation":
                    txt = fetch_text_from_slides(slides_svc, fid); return ("ok" if txt else "parsed_empty", txt, {})
                if mime == "application/vnd.google-apps.document":
                    txt = fetch_text_from_docs(docs_svc, fid); return ("ok" if txt else "parsed_empty", txt, {})
            return "unsupported", "", {}
        except Exception as e:
            if verbose: print(f"[DEBUG] Folder read failed (ECO): {repr(e)}")
            return "unreachable", "", {}

    m = GDRIVE_FILE_RE.search(url)
    if m:
        fid = m.group(1)
        try:
            mime = drive_get_mime(drive_svc, fid)
            if mime == "application/vnd.google-apps.spreadsheet":
                grids = fetch_sheet_grids(sheets_svc, fid, verbose=verbose)
                kpis = parse_kpis_from_grids(grids) if grids else {}
                summary = kpis_to_summary(kpis) if kpis else ""
                return ("ok" if kpis else "parsed_empty", summary, kpis)
            if mime == "application/vnd.google-apps.presentation":
                txt = fetch_text_from_slides(slides_svc, fid); return ("ok" if txt else "parsed_empty", txt, {})
            if mime == "application/vnd.google-apps.document":
                txt = fetch_text_from_docs(docs_svc, fid); return ("ok" if txt else "parsed_empty", txt, {})
            return "unsupported", "", {}
        except Exception as e:
            if verbose: print(f"[DEBUG] File read failed (ECO): {repr(e)}")
            return "unreachable", "", {}

    return "unsupported", "", {}

# -------------------------
# Projects build
# -------------------------
REQUIRED_COLUMNS = [
    "ID Registro",
    "Nombre Cliente",
    "Nombre del Negocio",
    "Año",
    "Catalogación",
    "Director de Negocio",
    "Etapa del Negocio",
    "PVP (precio venta público)",
    "Tipo de Moneda",
    "Industria asociada al Cliente",
    "Propuesta Técnica",
    "Propuesta Económica",
]

OPTIONAL_COLUMNS = [
    "Descripción", "Descripcion", "Descripción / Objetivo", "Comentarios",
    "Carpeta del Proyecto", "Folder", "Carpeta"
]

def build_projects(df: pd.DataFrame, services, excerpt_chars: int, verbose: bool=False) -> List[Dict[str, Any]]:
    drive_svc, sheets_svc, slides_svc, docs_svc = services
    df = df.dropna(how="all").dropna(axis=1, how="all")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print("ERROR: Faltan columnas requeridas en la hoja:")
        for c in missing: print(" -", c)
        sys.exit(1)

    projects: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        base = {
            "id_registro": row.get("ID Registro", ""),
            "nombre_cliente": row.get("Nombre Cliente", ""),
            "nombre_negocio": row.get("Nombre del Negocio", ""),
            "anio": row.get("Año", ""),
            "catalogacion": row.get("Catalogación", ""),
            "director_negocio": row.get("Director de Negocio", ""),
            "etapa_negocio": row.get("Etapa del Negocio", ""),
            "pvp": row.get("PVP (precio venta público)", ""),
            "moneda": row.get("Tipo de Moneda", ""),
            "industria": row.get("Industria asociada al Cliente", ""),
        }

        url_tecnica   = normalize_url(row.get("Propuesta Técnica", ""))
        url_economica = normalize_url(row.get("Propuesta Económica", ""))

        st_tec, txt_tec = fetch_tecnica_text(drive_svc, sheets_svc, slides_svc, docs_svc, url_tecnica, verbose=verbose)
        st_eco, eco_text, eco_kpis = fetch_economica(drive_svc, sheets_svc, slides_svc, docs_svc, url_economica, verbose=verbose)

        if verbose:
            mode = "Sheets→KPIs" if is_sheet_url(url_economica) else ("Slides/Docs" if url_economica else "No link")
            print(f"[{base['id_registro']}] TEC: {st_tec} | {url_tecnica[:100]}")
            print(f"[{base['id_registro']}] ECO: {st_eco} ({mode}) | {url_economica[:100]}")

        proj = {
            **base,
            "url_tecnica": url_tecnica,
            "url_economica": url_economica,
            "tecnica_fetch_status": st_tec,
            "economica_fetch_status": st_eco,
            "tecnica_text_excerpt": (txt_tec or "")[:excerpt_chars],
            # Para económica: si es Sheet, eco_text = resumen KPIs; si es Slides/Docs, eco_text = texto completo
            "economica_text_excerpt": (eco_text or "")[:excerpt_chars],
            "economica_kpis": eco_kpis,  # dict con pvp, meses, fee_cis_pct, costos_directos_pct, margen_bruto_pct
        }

        for oc in OPTIONAL_COLUMNS:
            if oc in df.columns:
                key = re.sub(r"\s+|/+", "_", oc.strip().lower())
                proj[key] = row.get(oc, "")

        projects.append(proj)

    return projects

# -------------------------
# Sheets → DataFrame
# -------------------------
def read_sheet_as_dataframe(sheets_svc, spreadsheet_id: str, gid: Optional[str]) -> pd.DataFrame:
    meta = sheets_svc.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=False).execute()
    sheets_meta = meta.get("sheets", [])
    sheet_id_to_title = {str(s["properties"]["sheetId"]): s["properties"]["title"] for s in sheets_meta}
    if gid and gid in sheet_id_to_title:
        title = sheet_id_to_title[gid]
    else:
        title = sheets_meta[0]["properties"]["title"]
    rng = f"{title}!A:ZZ"
    values = sheets_svc.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=rng).execute().get("values", [])
    if not values:
        return pd.DataFrame()
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Extrae proyectos desde un Google Sheet. Técnica: texto completo. Económica: KPIs si es Sheet; texto completo si es Slides/Docs.")
    parser.add_argument("--sheet-url", dest="sheet_url", required=False, help="URL del Google Sheet (ej: https://docs.google.com/spreadsheets/d/ID/edit#gid=0)")
    parser.add_argument("--sheet_url",  dest="sheet_url", required=False, help="URL del Google Sheet (alias)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Ruta para guardar projects.json")
    parser.add_argument("--client-secret", default="client_secret.json", help="Ruta a client_secret.json")
    parser.add_argument("--token", default="token.json", help="Ruta a token.json (se crea/actualiza)")
    parser.add_argument("--excerpt-chars", type=int, default=DEFAULT_EXCERPT_CHARS, help="Máx chars por *text_excerpt")
    parser.add_argument("--verbose", action="store_true", help="Log de depuración por proyecto")
    args = parser.parse_args()

    if not args.sheet_url:
        print("Falta --sheet-url. Ejemplo: --sheet-url \"https://docs.google.com/spreadsheets/d/.../edit#gid=0\"")
        sys.exit(1)

    spreadsheet_id, gid = parse_spreadsheet_url(args.sheet_url)
    if not spreadsheet_id:
        print("No pude extraer spreadsheetId de la URL. Revisa el link.")
        sys.exit(1)

    creds = get_creds(args.client_secret, args.token)
    services = get_services(creds)

    sheets_svc = services[1]
    df = read_sheet_as_dataframe(sheets_svc, spreadsheet_id, gid)
    if df.empty:
        print("La hoja está vacía o no fue posible leerla.")
        sys.exit(1)

    projects = build_projects(df, services, args.excerpt_chars, verbose=args.verbose)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK → {out_path.resolve()}  (proyectos: {len(projects)})")

if __name__ == "__main__":
    main()






