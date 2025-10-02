import os
import json
import time
import math
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import streamlit as st

# --- OpenAI SDK (v1) ---
try:
    from openai import OpenAI
except ImportError:
    st.error("Falta el paquete 'openai'. Instala requirements.txt")
    st.stop()

# -------- Config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-5") 
MODEL_EMB = os.getenv("OPENAI_MODEL_EMB", "text-embedding-3-large")

PROJECTS_PATH = os.getenv("PROJECTS_JSON", "projects.json")
TOP_K_DEFAULT = 3

CIS = {
    "rojo": "#F20034",
    "negro": "#000000",
    "blanco": "#FFFFFF",
    "gris_osc": "#21282F",
    "gris_intermedio": "#949596",
    "gris_claro": "#EEEEEE",
    "turquesa": "#005670",
    "celeste": "#9BB3BC",
    "rojo_claro": "#DB5D71",
    "gris_osc_2": "#424142",
}

# -------- Utils ----------
def css_cis():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Hind:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"]  {{
            font-family: 'Hind', sans-serif !important;
        }}
        .cis-title {{
            color: {CIS["negro"]};
            font-weight: 600;
            letter-spacing: 0.2px;
        }}
        .cis-chip {{
            display:inline-block; margin-right:.4rem; margin-top:.35rem;
            padding:.15rem .5rem; border-radius:14px;
            background:{CIS["gris_claro"]}; color:{CIS["gris_osc"]}; font-size:0.85rem;
        }}
        .cis-card {{
            border: 1px solid {CIS["gris_intermedio"]}22;
            border-left: 6px solid {CIS["turquesa"]};
            border-radius: 14px; padding: 14px 16px; margin-bottom: 14px;
            background: #fff;
        }}
        .cis-links a {{
            text-decoration: none; font-weight: 600;
        }}
        .cis-badge {{
            background:{CIS["turquesa"]}; color:white; padding:.15rem .5rem; border-radius:8px;
            font-weight:600; font-size:.85rem;
        }}
        .cis-why {{
            color:{CIS["gris_osc_2"]}; font-size:.95rem;
        }}
        .stButton>button {{
            background:{CIS["rojo"]}; color:white; border:none; border-radius:10px;
            padding:.5rem 1rem; font-weight:600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_projects(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        st.error(f"No encontré {path}. Pon tu projects.json en la carpeta del app.")
        st.stop()
    return json.loads(p.read_text(encoding="utf-8"))

def normalize_text(x: str) -> str:
    return " ".join((x or "").split())

def project_text_for_embedding(p: Dict[str, Any]) -> str:
    parts = [
        str(p.get("nombre_negocio") or ""),
        str(p.get("industria") or ""),
        str(p.get("catalogacion") or ""),
        normalize_text(p.get("tecnica_text_excerpt") or ""),
    ]
    return "\n".join([t for t in parts if t]).strip()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0

def emb_cache_key(texts: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update((t + "\n").encode("utf-8"))
    h.update(MODEL_EMB.encode())
    return h.hexdigest()[:16]

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL_EMB, input=texts)
    vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vecs)

def ensure_embeddings(client: OpenAI, projects: List[Dict[str, Any]]):
    corpus = [project_text_for_embedding(p) for p in projects]
    key = emb_cache_key(corpus)
    cache_file = Path(f".emb_cache_{key}.pkl")
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    # computa embeddings (batching simple)
    vecs = []
    B = 64
    for i in range(0, len(corpus), B):
        chunk = corpus[i:i+B]
        if any(len(c.strip()) == 0 for c in chunk):
            # asegura al menos " " para inputs vacíos
            chunk = [c if c.strip() else " " for c in chunk]
        v = embed_texts(client, chunk)
        vecs.append(v)
        time.sleep(0.1)
    mat = np.vstack(vecs) if vecs else np.zeros((0, 3072))
    cache_file.write_bytes(pickle.dumps(mat))
    return mat

def llm_brief_why(client: OpenAI, user_prompt: str, hits: List[Dict[str, Any]]) -> str:
    bullets = []
    for p in hits:
        ctx = normalize_text(p.get("tecnica_text_excerpt") or "")
        bullets.append(
            f"- ID {p.get('id_registro')} • {p.get('nombre_negocio','')} ({p.get('industria','')}). "
            f"Extracto técnico: {ctx[:500]}"
        )
    sys_prompt = (
        "Eres un consultor senior de CIS. Te doy un requerimiento y una lista de proyectos similares.\n"
        "Devuélveme 2–4 bullets, concisos, explicando por qué esos proyectos calzan con lo pedido.\n"
        "Evita repetir obviedades. Usa industria, objetivos, enfoque y resultados. Máximo 80 palabras."
    )
    content = f"Requerimiento del usuario:\n{user_prompt}\n\nProyectos candidatos:\n" + "\n".join(bullets)
    try:
        r = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": content},
            ],
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"(No pude generar explicación automática: {e})"

def format_kpi_badges(p: Dict[str, Any]) -> str:
    k = p.get("economica_kpis") or {}
    chips = []
    if "pvp" in k and k["pvp"] is not None:
        chips.append(f'<span class="cis-chip">PVP ≈ {int(k["pvp"]) if float(k["pvp"]).is_integer() else k["pvp"]} UF</span>')
    if "meses" in k and k["meses"] is not None:
        chips.append(f'<span class="cis-chip">Duración ≈ {int(k["meses"]) if float(k["meses"]).is_integer() else k["meses"]} meses</span>')
    if "margen_bruto_pct" in k and k["margen_bruto_pct"] is not None:
        chips.append(f'<span class="cis-chip">Margen ≈ {k["margen_bruto_pct"]}%</span>')
    if "fee_cis_pct" in k and k["fee_cis_pct"] is not None:
        chips.append(f'<span class="cis-chip">Fee CIS ≈ {k["fee_cis_pct"]}%</span>')
    if "costos_directos_pct" in k and k["costos_directos_pct"] is not None:
        chips.append(f'<span class="cis-chip">Costos directos ≈ {k["costos_directos_pct"]}%</span>')
    return " ".join(chips)

# -------- App ----------
st.set_page_config(page_title="CIS • Asistente de Proyectos", page_icon="🧭", layout="wide")
css_cis()

st.markdown(f"<h2 class='cis-title'>Asistente CIS · Recomendador de Proyectos</h2>", unsafe_allow_html=True)
st.caption("Escribe tu proyecto/consulta. Te recomendamos 2–3 antecedentes similares.")

if not OPENAI_API_KEY:
    st.warning("Define la variable de entorno OPENAI_API_KEY antes de ejecutar.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
projects = load_projects(PROJECTS_PATH)
emb_matrix = ensure_embeddings(client, projects)

# Sidebar filtros
with st.sidebar:
    st.subheader("Configuración")
    top_k = st.slider("Resultados (Top-K)", 1, 5, TOP_K_DEFAULT, 1)
    st.divider()
    st.subheader("Filtros opcionales (económicos)")
    pvp_max = st.text_input("PVP máximo (UF)", "")
    dur_max = st.text_input("Duración máxima (meses)", "")
    st.caption("Los filtros se aplican después del match semántico.")

prompt = st.text_area("Describe tu proyecto/objetivo", height=140, placeholder="Ej: PMO para implementar un modelo de operación en empresa de logística, foco en mejora de procesos y gobierno...")

colA, colB = st.columns([1,1])
with colA:
    if st.button("Buscar proyectos similares"):
        if not prompt.strip():
            st.error("Escribe tu requerimiento para buscar.")
            st.stop()

        # Embedding del prompt
        q_vec = embed_texts(client, [prompt])[0]

        # Scoring por similitud
        sims = np.array([cosine_sim(q_vec, emb_matrix[i]) for i in range(len(projects))])
        order = np.argsort(-sims).tolist()  # desc

        # TopN preliminar
        prelim = [projects[i] | {"_score": float(sims[i])} for i in order[:max(top_k*4, top_k)]]

        # Filtros opcionales
        def pass_filters(p):
            ok = True
            k = p.get("economica_kpis") or {}
            if pvp_max.strip():
                try:
                    ok = ok and (k.get("pvp") is not None and float(k["pvp"]) <= float(pvp_max))
                except:
                    ok = False
            if dur_max.strip():
                try:
                    ok = ok and (k.get("meses") is not None and float(k["meses"]) <= float(dur_max))
                except:
                    ok = False
            return ok

        filtered = [p for p in prelim if pass_filters(p)]
        hits = filtered[:top_k] if filtered else prelim[:top_k]

        # Explicación breve con LLM
        why = llm_brief_why(client, prompt, hits)
        st.markdown(f"<div class='cis-card'><div class='cis-why'><b>¿Por qué estos?</b><br>{why}</div></div>", unsafe_allow_html=True)

        # Render de tarjetas
        for p in hits:
            header = f"{p.get('id_registro','')} · {p.get('nombre_negocio','')}"
            industry = p.get("industria", "")
            score = f"{p.get('_score',0):.3f}"

            st.markdown("<div class='cis-card'>", unsafe_allow_html=True)
            st.markdown(f"### {header}", unsafe_allow_html=True)
            st.caption(f"{industry}  ·  similitud {score}")

            # chips económicos
            chips = format_kpi_badges(p)
            if chips:
                st.markdown(chips, unsafe_allow_html=True)

            # extracto técnico
            excerpt = (p.get("tecnica_text_excerpt") or "").strip()
            if excerpt:
                st.markdown(f"<div style='margin-top:.5rem; color:{CIS['gris_osc_2']}'>{excerpt[:700]}{'…' if len(excerpt)>700 else ''}</div>", unsafe_allow_html=True)

            # links
            links = []
            if p.get("url_tecnica"):
                links.append(f"<a href='{p['url_tecnica']}' target='_blank' style='color:{CIS['turquesa']}'>Ver Propuesta Técnica</a>")
            if p.get("url_economica"):
                links.append(f"<a href='{p['url_economica']}' target='_blank' style='color:{CIS['rojo']}'>Ver Propuesta Económica</a>")
            if links:
                st.markdown(f"<div class='cis-links' style='margin-top:.6rem'>{' · '.join(links)}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.info("Tip: puedes ajustar Top-K y filtros de PVP/Duración en la barra lateral.\n\n• Embeddings: texto técnico + título + industria.\n• Económico: mostramos KPIs si existen y link al sheet.")
