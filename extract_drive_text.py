#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# -------------------------
# Config & Scopes
# -------------------------
# Solo necesitamos leer Google Sheets
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

DEFAULT_OUTPUT = "projects.json"

# -------------------------
# URL Parsing
# -------------------------
GSHEETS_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)")
GID_RE     = re.compile(r"[?#&]gid=(\d+)")

def parse_spreadsheet_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    m = GSHEETS_RE.search(url or "")
    if not m:
        return None, None
    spreadsheet_id = m.group(1)
    gid = None
    mg = GID_RE.search(url)
    if mg:
        gid = mg.group(1)
    return spreadsheet_id, gid

def normalize_url(val: Optional[str]) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if s in {"-", "—", "N/A", "#N/A", ""}:
        return ""
    return s

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

def get_sheets_service(creds):
    return build("sheets", "v4", credentials=creds)

# -------------------------
# Sheets → DataFrame
# -------------------------
def read_sheet_as_dataframe(sheets_svc, spreadsheet_id: str, gid: Optional[str]) -> pd.DataFrame:
    meta = sheets_svc.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=False).execute()
    sheets_meta = meta.get("sheets", [])
    if not sheets_meta:
        return pd.DataFrame()

    # Map sheetId → title
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
# Column aliasing & validation
# -------------------------
# Grupos de columnas requeridas (al menos UNA de cada grupo debe existir)
REQUIRED_GROUPS = {
    "id_registro": ["ID Registro"],
    "nombre_cliente": ["Nombre Cliente"],
    "nombre_proyecto": ["Nombre del Proyecto", "Nombre del Negocio"],  # back-compat
    "anio": ["Año"],
    "catalogacion": ["Catalogación del servicio", "Catalogación"],      # back-compat
    "director_negocio": ["Director de Negocio"],
    "etapa_negocio": ["Etapa del Negocio"],
    "pvp": ["PVP (precio venta público)"],
    "moneda": ["Tipo de Moneda"],
    "industria": ["Industria asociada al Cliente"],
    "url_tecnica": ["Propuesta Técnica"],
    "url_economica": ["Propuesta Económica"],
}

# Columnas opcionales
OPTIONAL_ALIASES = {
    "area_negocio": ["Área de Negocio"],
    "objetivo_general": ["Objetivo General del Proyecto"],
    "objetivos_especificos": ["Objetivos Específicos del Proyecto"],
    "categorizacion_carpeta_drive": ["Categorización Carpeta Drive", "Carpeta del Proyecto", "Folder", "Carpeta"],
    "descripcion": ["Descripción", "Descripcion", "Descripción / Objetivo", "Comentarios"],
}

def first_present(series_like: pd.Series, names: List[str], default="") -> Any:
    # Retorna el primer valor no nulo de los encabezados de 'names' presentes en la fila
    for n in names:
        if n in series_like and pd.notna(series_like.get(n)):
            return series_like.get(n)
    return default

def validate_required_columns(df: pd.DataFrame):
    missing_groups = []
    for logical_key, headers in REQUIRED_GROUPS.items():
        if not any(h in df.columns for h in headers):
            missing_groups.append((logical_key, headers))
    if missing_groups:
        print("ERROR: faltan columnas requeridas (por grupos de alias aceptados):")
        for logical_key, headers in missing_groups:
            print(f" - {logical_key}: alguno de {headers}")
        sys.exit(1)

# -------------------------
# Projects build (solo campos + links)
# -------------------------
def build_projects(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    validate_required_columns(df)

    projects: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        proj = {
            "id_registro": first_present(row, REQUIRED_GROUPS["id_registro"]),
            "nombre_cliente": first_present(row, REQUIRED_GROUPS["nombre_cliente"]),
            "nombre_proyecto": first_present(row, REQUIRED_GROUPS["nombre_proyecto"]),
            "anio": first_present(row, REQUIRED_GROUPS["anio"]),
            "catalogacion": first_present(row, REQUIRED_GROUPS["catalogacion"]),
            "director_negocio": first_present(row, REQUIRED_GROUPS["director_negocio"]),
            "etapa_negocio": first_present(row, REQUIRED_GROUPS["etapa_negocio"]),
            "pvp": first_present(row, REQUIRED_GROUPS["pvp"]),
            "moneda": first_present(row, REQUIRED_GROUPS["moneda"]),
            "industria": first_present(row, REQUIRED_GROUPS["industria"]),
            # Links tal cual (sin leer contenido)
            "url_tecnica": normalize_url(first_present(row, REQUIRED_GROUPS["url_tecnica"])),
            "url_economica": normalize_url(first_present(row, REQUIRED_GROUPS["url_economica"])),
        }

        # Agregar opcionales si existen
        for out_key, header_aliases in OPTIONAL_ALIASES.items():
            val = first_present(row, header_aliases, "")
            if val != "":
                proj[out_key] = val

        projects.append(proj)

    return projects

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Extrae proyectos desde un Google Sheet. Solo campos y links (no lee contenidos).")
    parser.add_argument("--sheet-url", dest="sheet_url", required=False, help="URL del Google Sheet (ej: https://docs.google.com/spreadsheets/d/ID/edit#gid=0)")
    parser.add_argument("--sheet_url",  dest="sheet_url_alias", required=False, help="URL del Google Sheet (alias)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Ruta para guardar projects.json")
    parser.add_argument("--client-secret", default="client_secret.json", help="Ruta a client_secret.json")
    parser.add_argument("--token", default="token.json", help="Ruta a token.json (se crea/actualiza)")
    args = parser.parse_args()

    # Soportar ambos nombres del argumento
    sheet_url = args.sheet_url or args.sheet_url_alias
    if not sheet_url:
        print('Falta --sheet-url. Ejemplo: --sheet-url "https://docs.google.com/spreadsheets/d/.../edit#gid=0"')
        sys.exit(1)

    spreadsheet_id, gid = parse_spreadsheet_url(sheet_url)
    if not spreadsheet_id:
        print("No pude extraer spreadsheetId de la URL. Revisa el link.")
        sys.exit(1)

    creds = get_creds(args.client_secret, args.token)
    sheets_svc = get_sheets_service(creds)

    df = read_sheet_as_dataframe(sheets_svc, spreadsheet_id, gid)
    if df.empty:
        print("La hoja está vacía o no fue posible leerla.")
        sys.exit(1)

    projects = build_projects(df)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK → {out_path.resolve()}  (proyectos: {len(projects)})")

if __name__ == "__main__":
    main()







