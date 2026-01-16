import os
import json
import time
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- OpenAI SDK (v1) ---
try:
    from openai import OpenAI
except ImportError:
    st.error("Falta el paquete 'openai'. Instala requirements.txt")
    st.stop()

# ========================================
# CONFIGURACIÓN
# ========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_CHAT = os.getenv("OPENAI_MODEL_CHAT", "gpt-5-mini-2025-08-07")
MODEL_EMB = os.getenv("OPENAI_MODEL_EMB", "text-embedding-3-large")

PROJECTS_PATH = os.getenv("PROJECTS_JSON", "projects.json")
TOP_K_DEFAULT = 2
EMBEDDING_BATCH_SIZE = 64
CANDIDATE_POOL_SIZE = 4  # Cuántos candidatos enviamos a GPT para re-ranking

# ========================================
# ESTADO DE CAPTURA CONVERSACIONAL
# ========================================
REQUIRED_SLOTS = ["area", "categoria", "industrias", "objetivo"]

# Umbrales de confianza para agente semántico
CONF_ACCEPT = 0.75      # Auto-acepta si confianza >= 0.75
CONF_CONFIRM = 0.45     # Entre 0.45-0.75 pide confirmación

# ========================================
# CONSTANTES DE FORMULARIO
# ========================================
AREAS = ["Estrategia", "Personas", "Operaciones"]

CATEGORIAS = {
    "Estrategia": [
        "Modelo de Negocio",
        "Estrategia de Negocio",
        "Gobierno Corporativo",
        "Gestión de Proyectos",
        "Estrategia Corporativa",
        "Otro"
    ],
    "Personas": [
        "Gestión del Talento",
        "Desarrollo Organizacional",
        "Cultura y Clima",
        "Compensaciones",
        "Otro"
    ],
    "Operaciones": [
        "Procesos",
        "Supply Chain",
        "Eficiencia Operacional",
        "Transformación Digital",
        "Otro"
    ]
}

INDUSTRIAS = [
    "Inmobiliario",
    "Entretenimiento",
    "Construcción",
    "Alimentación",
    "Industria Química",
    "Reciclaje",
    "Movilidad",
    "Desarrollo Tecnológico",
    "Manufactura",
    "Servicios Jurídicos",
    "Minería",
    "Industria",
    "Ferretería",
    "Tecnología de la información y Comunicaciones (TIC's)",
    "Servicios Profesionales",
    "Otro"
]

DURACIONES = [
    "Menos de 3 meses",
    "Entre 3-6 meses",
    "Entre 6-12 meses",
    "Más de 12 meses"
]

ENTREGABLES = [
    "Roadmap",
    "Diagnóstico",
    "Plan de implementación",
    "Carta Gantt",
    "Modelo de negocio",
    "Análisis financiero"
]

ENFOQUES = [
    "Diagnóstico",
    "Diseño",
    "Implementación"
]

# ========================================
# COLORES CIS
# ========================================
CIS = {
    "rojo": "#F20034",
    "negro": "#111111",
    "blanco": "#FFFFFF",
    "gris_osc": "#1F2937",
    "gris_intermedio": "#6B7280",
    "gris_claro": "#F3F4F6",
    "turquesa": "#005670",
    "celeste": "#9BB3BC",
    "rojo_claro": "#DB5D71",
    "gris_osc_2": "#374151",
    "link": "#0F766E"
}

# ========================================
# FUNCIONES DE UTILIDAD
# ========================================

def css_cis():
    """Estilos CSS mejorados con mejor contraste y fuentes más grandes"""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* BASE - Fuentes más grandes */
        html, body, [class*="css"] {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
          color: #FFFFFF !important;
          background: #0E1117;
          font-size: 18px !important;
          line-height: 1.7;
          -webkit-font-smoothing: antialiased;
        }

        .main .block-container {
          padding: 2.5rem 3rem 3rem 3rem !important;
          max-width: 1400px !important;
        }
        
        .stApp {
          background: #0E1117 !important;
        }
        
        /* Títulos con mejor contraste */
        .main h1, .main h2, .main h3 {
          color: #FFFFFF !important;
          font-weight: 700 !important;
        }
        .main h1 {
          font-size: 2.2rem !important;
        }
        .main h2 {
          font-size: 1.7rem !important;
        }
        .main h3 {
          font-size: 1.4rem !important;
        }
        
        /* Labels de formulario - BLANCOS y más grandes */
        label[data-testid="stWidgetLabel"] {
          color: #FFFFFF !important;
          font-weight: 700 !important;
          font-size: 1.15rem !important;
          background: rgba(0, 86, 112, 0.3) !important;
          padding: 0.5rem 0.9rem !important;
          border-radius: 8px !important;
          display: inline-block !important;
          margin-bottom: 0.6rem !important;
          border-left: 4px solid #005670 !important;
        }

        /* Sidebar SIMPLIFICADO - Fondo oscuro con texto blanco */
        section[data-testid="stSidebar"] {
          background: #1a1d24 !important;
          border-right: 2px solid #005670 !important;
          padding-top: 2rem !important;
        }
        section[data-testid="stSidebar"] > div {
          padding: 0 1.5rem !important;
        }
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
          color: #FFFFFF !important;
          font-weight: 700 !important;
          font-size: 1.3rem !important;
          margin-bottom: 1.2rem !important;
          letter-spacing: -0.02em;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
          color: #E5E7EB !important;
          font-size: 1.05rem !important;
        }
        section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] {
          color: #FFFFFF !important;
          font-weight: 600 !important;
          font-size: 1.1rem !important;
        }
        section[data-testid="stSidebar"] .stCaption {
          color: #9CA3AF !important;
          font-size: 0.95rem !important;
        }

        /* Header principal */
        .cis-header {
          background: linear-gradient(135deg, #005670 0%, #007A94 100%);
          color: white;
          padding: 2.2rem 2.8rem;
          border-radius: 16px;
          margin-bottom: 2rem;
          box-shadow: 0 4px 6px rgba(0, 86, 112, 0.1), 0 2px 4px rgba(0, 86, 112, 0.06);
        }
        .cis-header h1 {
          color: white !important;
          font-size: 2.3rem !important;
          font-weight: 800 !important;
          margin: 0 0 0.5rem 0 !important;
          letter-spacing: -0.03em;
        }
        .cis-header p {
          color: rgba(255, 255, 255, 0.95) !important;
          font-size: 1.15rem !important;
          margin: 0;
          font-weight: 400;
        }

        /* Info box con mejor contraste */
        .cis-info-box {
          background: rgba(0, 86, 112, 0.15);
          border: 2px solid #005670;
          border-left: 5px solid #005670;
          border-radius: 12px;
          padding: 1.5rem 1.75rem;
          margin: 1.5rem 0;
          box-shadow: 0 1px 3px rgba(0, 86, 112, 0.2);
        }
        .cis-info-box strong {
          color: #FFFFFF !important;
          font-weight: 700;
          font-size: 1.2rem !important;
        }
        .cis-info-box p {
          color: #E5E7EB !important;
          font-size: 1.05rem !important;
        }

        /* Dividers más visibles */
        .main hr {
          background: linear-gradient(90deg, transparent, #6B7280 50%, transparent) !important;
          margin: 2rem 0 !important;
          opacity: 0.7;
          height: 2px !important;
        }

        /* Tarjetas de resultado con mejor contraste */
        .cis-result-card {
          background: #1F2937 !important;
          border: 2px solid #374151 !important;
          border-radius: 14px;
          padding: 2rem 2.3rem;
          margin-bottom: 1.8rem;
          box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
          transition: all 0.2s ease;
          position: relative;
          overflow: hidden;
        }
        .cis-result-card::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          bottom: 0;
          width: 6px;
          background: linear-gradient(180deg, #005670 0%, #00A0D2 100%);
        }
        .cis-result-card:hover {
          box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3);
          transform: translateY(-2px);
          border-color: #005670 !important;
        }

        /* Título cliente - MÁS VISIBLE */
        .cis-client-name {
          color: #00D4FF !important;
          font-size: 1.6rem !important;
          font-weight: 800 !important;
          margin: 0 0 0.7rem 0;
          letter-spacing: -0.02em;
        }

        /* Metadata - TEXTO BLANCO */
        .cis-meta-item {
          color: #FFFFFF !important;
          font-size: 1.1rem !important;
          margin: 0.5rem 0;
          display: flex;
          align-items: baseline;
          line-height: 1.6;
        }
        .cis-meta-item strong {
          color: #9BB3BC !important;
          font-weight: 700 !important;
          min-width: 160px;
          font-size: 1.1rem !important;
        }

        /* Badge de GPT Score */
        .gpt-badge {
          display: inline-block;
          background: linear-gradient(135deg, #F20034 0%, #C70028 100%);
          color: white;
          padding: 0.4rem 0.8rem;
          border-radius: 6px;
          font-weight: 800;
          font-size: 1rem;
          margin-left: 0.5rem;
          box-shadow: 0 2px 4px rgba(242, 0, 52, 0.3);
        }

        /* Reasoning box */
        .gpt-reasoning {
          background: rgba(242, 0, 52, 0.08);
          border: 2px solid rgba(242, 0, 52, 0.3);
          border-left: 4px solid #F20034;
          border-radius: 10px;
          padding: 1rem 1.3rem;
          margin: 1rem 0;
        }
        .gpt-reasoning strong {
          color: #FF4D6D !important;
          font-size: 1.05rem !important;
        }
        .gpt-reasoning p {
          color: #E5E7EB !important;
          margin: 0.3rem 0 0 0;
          font-size: 1rem !important;
        }
        .gpt-reasoning ul {
          margin: 0.5rem 0;
          padding-left: 1.5rem;
        }
        .gpt-reasoning li {
          color: #E5E7EB !important;
          margin: 0.3rem 0;
        }

        /* Sección económica con mejor contraste */
        .cis-economic-box {
          background: rgba(0, 86, 112, 0.15);
          border: 2px solid #005670;
          border-radius: 10px;
          padding: 1.2rem 1.4rem;
          margin: 1.2rem 0;
        }
        .cis-economic-box strong {
          color: #00D4FF !important;
          font-size: 1.15rem !important;
        }
        .cis-economic-box div {
          font-size: 1.05rem !important;
          color: #FFFFFF !important;
        }

        /* Extracto técnico - TEXTO BLANCO */
        .cis-excerpt {
          color: #FFFFFF !important;
          font-size: 1.05rem !important;
          line-height: 1.7;
          padding: 1.2rem;
          background: rgba(0, 86, 112, 0.1);
          border-radius: 10px;
          margin: 1.2rem 0;
          border-left: 4px solid #9BB3BC;
        }
        .cis-excerpt strong {
          color: #00D4FF !important;
          display: block;
          margin-bottom: 0.5rem;
        }

        /* Enlaces con mejor contraste */
        .cis-links {
          margin-top: 1.4rem;
          padding-top: 1.2rem;
          border-top: 2px solid #374151;
          display: flex;
          gap: 1.2rem;
          flex-wrap: wrap;
        }
        .cis-links a {
          color: white !important;
          background: #005670 !important;
          text-decoration: none !important;
          font-weight: 700 !important;
          font-size: 1.05rem !important;
          padding: 0.75rem 1.5rem !important;
          border-radius: 8px !important;
          transition: all 0.2s ease !important;
          display: inline-flex !important;
          align-items: center !important;
          border: 2px solid #005670 !important;
        }
        .cis-links a:hover {
          background: #003D4D !important;
          border-color: #00D4FF !important;
          transform: translateY(-2px) !important;
          box-shadow: 0 4px 10px rgba(0, 212, 255, 0.3) !important;
        }
                .cis-links a.disabled {
                    background: #1f2937 !important;
                    color: #9ca3af !important;
                    border-color: #374151 !important;
                    opacity: 0.6 !important;
                    pointer-events: none !important;
                    cursor: not-allowed !important;
                    box-shadow: none !important;
                    transform: none !important;
                }

        /* Box "Por qué" con mejor contraste */
        .cis-why-box {
          background: rgba(242, 0, 52, 0.1);
          border: 2px solid #F20034;
          border-left: 6px solid #F20034;
          border-radius: 12px;
          padding: 1.7rem 2rem;
          margin: 2rem 0;
          box-shadow: 0 3px 6px rgba(242, 0, 52, 0.15);
        }
        .cis-why-box strong {
          color: #FF4D6D !important;
          font-size: 1.25rem !important;
          display: block;
          margin-bottom: 0.9rem;
          font-weight: 800 !important;
        }
        .cis-why-box p {
          color: #FFFFFF !important;
          margin: 0;
          line-height: 1.7;
          font-size: 1.05rem !important;
        }

        /* Botón principal más grande */
        .stButton > button {
          background: linear-gradient(135deg, #F20034 0%, #C70028 100%) !important;
          color: white !important;
          border: none !important;
          border-radius: 10px !important;
          padding: 1rem 3rem !important;
          font-weight: 700 !important;
          font-size: 1.15rem !important;
          box-shadow: 0 4px 8px rgba(242, 0, 52, 0.3) !important;
          transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
          transform: translateY(-2px) !important;
          box-shadow: 0 6px 16px rgba(242, 0, 52, 0.4) !important;
        }

        /* Inputs con mejor contraste */
        .stTextArea textarea, .stTextInput input {
          border: 2px solid #374151 !important;
          border-radius: 8px !important;
          padding: 1rem !important;
          font-size: 1.05rem !important;
          background: #1F2937 !important;
          color: #FFFFFF !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
          border-color: #005670 !important;
          box-shadow: 0 0 0 3px rgba(0, 86, 112, 0.2) !important;
        }
        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
          color: #9CA3AF !important;
          opacity: 1 !important;
        }

        /* Multiselect y Selectbox con mejor contraste */
        .stMultiSelect, .stSelectbox {
          margin: 0.6rem 0;
        }
        .stMultiSelect div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] > div {
          color: #FFFFFF !important;
          font-size: 1.05rem !important;
          background: #1F2937 !important;
          border: 2px solid #374151 !important;
        }
        div[role="listbox"] li {
          color: #0B0B0B !important;
          font-size: 1.05rem !important;
        }

        /* Checkbox más visible */
        .stCheckbox {
          background: rgba(242, 0, 52, 0.1) !important;
          padding: 0.7rem 1rem !important;
          border-radius: 8px !important;
          margin: 1rem 0 !important;
          border-left: 4px solid #F20034 !important;
        }
        .stCheckbox label {
          color: #FFFFFF !important;
          font-weight: 700 !important;
          font-size: 1.15rem !important;
        }
        
        /* Markdown text visible */
        .main .stMarkdown {
          color: #FFFFFF !important;
        }
        .main .stMarkdown p {
          color: #E5E7EB !important;
        }

        /* Radio buttons más grandes */
        .stRadio > label {
          font-size: 1.1rem !important;
          color: #FFFFFF !important;
        }
        .stRadio div[role="radiogroup"] label {
          font-size: 1.05rem !important;
          color: #E5E7EB !important;
        }

        /* Sliders más visibles */
        .stSlider {
          padding: 0.5rem 0 !important;
        }

        /* Chat scroll area - estilo ChatGPT */
        .chat-box {
          height: 520px;
          overflow-y: auto;
          overflow-x: hidden;
          padding: 14px;
          border: 1px solid #374151;
          border-radius: 12px;
          background: rgba(0,0,0,0.25);
        }

        .msg {
          display: flex;
          margin: 10px 0;
        }

        .msg.user {
          justify-content: flex-end;
        }

        .bubble {
          max-width: 78%;
          padding: 10px 12px;
          border-radius: 12px;
          border: 1px solid rgba(255,255,255,0.08);
          line-height: 1.45;
          font-size: 1.02rem;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        .msg.assistant .bubble {
          background: #111827;
          color: #E5E7EB;
        }

        .msg.user .bubble {
          background: rgba(242, 0, 52, 0.15);
          border-color: rgba(242, 0, 52, 0.35);
          color: #FFFFFF;
        }

        .msg .meta {
          font-size: 0.85rem;
          opacity: 0.8;
          margin-bottom: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_projects(path: str) -> List[Dict[str, Any]]:
    """Carga el archivo JSON de proyectos"""
    p = Path(path)
    if not p.exists():
        st.error(f"❌ No encontré el archivo {path}. Asegúrate de tener projects.json en la carpeta del proyecto.")
        st.stop()
    
    try:
        content = p.read_text(encoding="utf-8")
        projects = json.loads(content)
        
        if not isinstance(projects, list):
            st.error("❌ El archivo JSON debe contener una lista de proyectos")
            st.stop()
            
        return projects
    except json.JSONDecodeError as e:
        st.error(f"❌ Error al parsear JSON: {e}")
        st.stop()


def normalize_text(x: str) -> str:
    """Normaliza espacios en blanco en texto"""
    return " ".join((x or "").split())


def project_text_for_embedding(p: Dict[str, Any]) -> str:
    """Construye el texto representativo de un proyecto para generar embeddings"""
    parts = [
        str(p.get("nombre_proyecto") or ""),
        str(p.get("industria") or ""),
        str(p.get("catalogacion") or ""),
        str(p.get("area_negocio") or ""),
        normalize_text(p.get("objetivo_general") or ""),
        normalize_text(p.get("objetivos_especificos") or ""),
    ]
    return "\n".join([t for t in parts if t]).strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula similitud de coseno entre dos vectores"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def emb_cache_key(texts: List[str]) -> str:
    """Genera clave de caché basada en el corpus de texto"""
    h = hashlib.sha256()
    for t in texts:
        h.update((t + "\n").encode("utf-8"))
    h.update(MODEL_EMB.encode())
    return h.hexdigest()[:16]


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Genera embeddings usando la API de OpenAI"""
    try:
        resp = client.embeddings.create(model=MODEL_EMB, input=texts)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vecs)
    except Exception as e:
        st.error(f"❌ Error al generar embeddings: {e}")
        raise


def ensure_embeddings(client: OpenAI, projects: List[Dict[str, Any]]) -> np.ndarray:
    """Genera o carga embeddings desde caché"""
    corpus = [project_text_for_embedding(p) for p in projects]
    key = emb_cache_key(corpus)
    cache_file = Path(f".emb_cache_{key}.pkl")
    
    if cache_file.exists():
        try:
            return pickle.loads(cache_file.read_bytes())
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar caché de embeddings: {e}. Regenerando...")
    
    vecs = []
    for i in range(0, len(corpus), EMBEDDING_BATCH_SIZE):
        chunk = corpus[i:i + EMBEDDING_BATCH_SIZE]
        chunk = [c if c.strip() else " " for c in chunk]
        v = embed_texts(client, chunk)
        vecs.append(v)
        time.sleep(0.05)
    
    mat = np.vstack(vecs) if vecs else np.zeros((0, 3072))
    
    try:
        cache_file.write_bytes(pickle.dumps(mat))
    except Exception as e:
        st.warning(f"⚠️ No se pudo guardar caché de embeddings: {e}")
    
    return mat


def llm_rerank_projects(
    client: OpenAI, 
    user_prompt: str, 
    candidates: List[tuple], 
    top_k: int
) -> List[Dict[str, Any]]:
    """
    Usa GPT-5 para analizar y re-rankear proyectos con razonamiento estratégico
    """
    
    # Preparar candidatos para GPT
    candidatos_texto = []
    for idx, (proj_idx, semantic_score, p) in enumerate(candidates, 1):
        objetivo_general = (p.get("objetivo_general") or "")[:300]
        objetivos_especificos = (p.get("objetivos_especificos") or "")[:200]
        
        candidatos_texto.append(f"""
CANDIDATO {idx} | Similitud Semántica: {semantic_score:.1%}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Cliente: {p.get('nombre_cliente', 'N/A')}
• Proyecto: {p.get('nombre_proyecto', 'N/A')}
• Industria: {p.get('industria', 'N/A')}
• Área: {p.get('area_negocio', 'N/A')}
• Categorización: {p.get('catalogacion', 'N/A')}
• Año: {p.get('anio', 'N/A')}
• Objetivo General: {objetivo_general}
• Objetivos Específicos: {objetivos_especificos}
""")
    
    system_prompt = f"""Eres un consultor estratégico senior de CIS con 20+ años de experiencia en matching de proyectos de consultoría.

Tu tarea es ANALIZAR y RE-RANKEAR los {len(candidates)} proyectos candidatos según su RELEVANCIA ESTRATÉGICA REAL para el requerimiento del cliente.

CRITERIOS DE EVALUACIÓN (ponderados):
1. **Transferibilidad de Objetivos** (40%): ¿Los desafíos de negocio y metas son genuinamente comparables y aplicables?
2. **Contexto de Industria** (25%): ¿El sector, mercado y dinámicas de negocio son transferibles?
3. **Aplicabilidad Metodológica** (20%): ¿Las metodologías, enfoques y herramientas usadas son replicables?
4. **Alineación de Categoría** (15%): ¿El tipo de proyecto y área de consultoría son consistentes?

PRINCIPIOS DE RAZONAMIENTO:
✓ PRIORIZA proyectos con aprendizajes genuinamente transferibles, no solo similitud superficial
✓ VALORA industrias relacionadas (ej: Retail ≈ Alimentación) sobre matches exactos sin aplicabilidad
✓ CONSIDERA trade-offs estratégicos (mejor industria vs mejor metodología)
✓ DESCARTA proyectos con similitud semántica alta pero contexto de negocio incompatible
✓ IDENTIFICA fortalezas específicas (qué hace valioso este match) y debilidades (qué limitaciones tiene)

RESPONDE EN JSON ESTRICTO con este formato EXACTO:
{{
  "top_projects": [
    {{
      "candidato_numero": 1,
      "score_final": 95,
      "razon_principal": "Match excepcional: industria idéntica, objetivos de transformación digital altamente transferibles y metodología probada en contexto similar",
      "fortalezas": [
        "Industria y segmento de mercado idénticos",
        "Objetivos de negocio altamente alineados con enfoque en digitalización",
        "Metodología de design thinking aplicable directamente"
      ],
      "debilidades": [
        "Alcance del proyecto original menor al requerimiento actual",
        "No aborda específicamente el componente de analytics mencionado"
      ]
    }}
  ],
  "analisis_general": "Breve análisis de patrones encontrados en el matching (2-3 líneas)"
}}

CRÍTICO: 
- Responde SOLO con JSON válido, sin markdown ni texto adicional
- Incluye EXACTAMENTE {top_k} proyectos en top_projects
- Cada score_final debe ser un número entero entre 0 y 100
- Las razones deben ser específicas y accionables, no genéricas"""

    user_message = f"""REQUERIMIENTO DEL CLIENTE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{user_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CANDIDATOS PRE-SELECCIONADOS (por similitud semántica):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chr(10).join(candidatos_texto)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analiza estratégicamente y selecciona los {top_k} proyectos más valiosos con razonamiento explícito."""

    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        
        # Limpiar posibles markdown
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        
        # Mapear de vuelta a los proyectos originales
        final_ranking = []
        for item in result["top_projects"]:
            candidato_idx = item["candidato_numero"] - 1
            
            if candidato_idx >= len(candidates):
                continue
                
            proj_idx, semantic_score, project = candidates[candidato_idx]
            
            final_ranking.append({
                "project": project,
                "gpt_score": item["score_final"],
                "semantic_score": semantic_score,
                "reasoning": item["razon_principal"],
                "strengths": item.get("fortalezas", []),
                "weaknesses": item.get("debilidades", [])
            })
        
        return final_ranking, result.get("analisis_general", "")
        
    except Exception as e:
        st.warning(f"⚠️ GPT-5 re-ranking falló: {e}. Usando orden semántico como fallback.")
        # Fallback al orden semántico
        fallback = []
        for _, semantic_score, project in candidates[:top_k]:
            fallback.append({
                "project": project,
                "semantic_score": semantic_score,
                "gpt_score": None,
                "reasoning": "Re-ranking no disponible (usando similitud semántica)",
                "strengths": [],
                "weaknesses": []
            })
        return fallback, ""


def llm_explain_selection(client: OpenAI, user_prompt: str, ranked_projects: List[Dict[str, Any]]) -> str:
    """Genera explicación contextual de la selección final"""
    
    bullets = []
    for item in ranked_projects:
        p = item["project"]
        gpt_score = item.get("gpt_score")
        semantic_score = item.get("semantic_score", 0)
        
        score_text = f"GPT-5: {gpt_score}/100" if gpt_score else f"Semántica: {semantic_score:.1%}"
        
        bullets.append(
            f"- **{p.get('nombre_cliente', 'Sin nombre')}** ({p.get('industria', 'N/A')})\n"
            f"  {p.get('nombre_proyecto', 'N/A')} | {score_text}\n"
            f"  {item.get('reasoning', '')[:150]}...\n"
        )
    
    system_prompt = """Eres un consultor senior de CIS. Genera una explicación ejecutiva clara y persuasiva de POR QUÉ estos proyectos son los más relevantes.

INSTRUCCIONES:
- Escribe 2-3 párrafos concisos (máximo 150 palabras total)
- Enfócate en conexiones estratégicas y aprendizajes transferibles
- Menciona proyectos por nombre de cliente cuando sea relevante
- Lenguaje profesional pero conversacional
- Sin bullet points ni listas"""

    content = f"""REQUERIMIENTO:
{user_prompt}

PROYECTOS SELECCIONADOS:
{chr(10).join(bullets)}

Explica de manera ejecutiva por qué estos son los mejores matches."""

    try:
        r = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ No se pudo generar explicación: {e}"


def build_user_prompt(form_data: Dict[str, Any]) -> str:
    """Construye el prompt del usuario desde el formulario"""
    parts = []
    
    parts.append("=== CONTEXTO DEL PROYECTO ===")
    
    if form_data.get("area"):
        parts.append(f"Área de enfoque: {form_data['area']}")
    
    if form_data.get("categoria"):
        parts.append(f"Categoría específica: {form_data['categoria']}")
    
    if form_data.get("industrias"):
        parts.append(f"Industria(s) del cliente: {', '.join(form_data['industrias'])}")
    
    if form_data.get("objetivo"):
        parts.append("\n=== OBJETIVO PRINCIPAL ===")
        parts.append(form_data['objetivo'])
    
    detalles = []
    
    if form_data.get("duracion"):
        detalles.append(f"Duración estimada: {form_data['duracion']}")
    
    if form_data.get("entregables"):
        detalles.append(f"Entregables esperados: {', '.join(form_data['entregables'])}")
    
    if form_data.get("enfoque"):
        detalles.append(f"Enfoque del proyecto: {', '.join(form_data['enfoque'])}")
    
    if detalles:
        parts.append("\n=== DETALLES ADICIONALES ===")
        parts.extend(detalles)
    
    if form_data.get("comentarios"):
        parts.append("\n=== INFORMACIÓN COMPLEMENTARIA ===")
        parts.append(form_data['comentarios'])
    
    return "\n".join(parts)


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """Convierte un valor a float de manera segura"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def render_chat_html(messages: List[Dict[str, str]]) -> str:
    """Renderiza el historial del chat como HTML con burbujas - incluye CSS inline"""
    import html
    
    # CSS inline para el iframe del componente
    css = """
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0E1117;
        }
        .chat-box {
            height: 100%;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 14px;
            background: rgba(0,0,0,0.25);
        }
        .msg {
            display: flex;
            margin: 10px 0;
        }
        .msg.user {
            justify-content: flex-end;
        }
        .bubble {
            max-width: 78%;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            line-height: 1.45;
            font-size: 1.02rem;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .msg.assistant .bubble {
            background: #111827;
            color: #E5E7EB;
        }
        .msg.user .bubble {
            background: rgba(242, 0, 52, 0.15);
            border-color: rgba(242, 0, 52, 0.35);
            color: #FFFFFF;
        }
    </style>
    """
    
    out = [css]
    out.append("<div class='chat-box' id='chat-box'>")
    
    for m in messages:
        role = m.get("role", "assistant")
        safe_content = html.escape(m.get("content", "") or "").replace("\n", "<br>")
        
        out.append(
            "<div class='msg {role}'>"
            "<div class='bubble'>{safe}</div>"
            "</div>"
            .format(role=role, safe=safe_content)
        )
    
    out.append("<div id='chat-bottom'></div></div>")
    return "".join(out)


# ========================================
# CAPTURA CONVERSACIONAL (SLOTS)
# ========================================
def init_chat_state():
    """Inicializa el estado de conversación y slots en session_state"""
    if "slot_values" not in st.session_state:
        st.session_state.slot_values = {
            "area": "",
            "categoria": "",
            "industrias": [],
            "objetivo": "",
            "duracion": None,
            "entregables": [],
            "enfoque": [],
            "comentarios": ""
        }
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hola, soy tu asistente CIS 👋\n\n"
                    "Conversemos sobre tu proyecto para encontrar casos similares relevantes. "
                    "Cuéntame: ¿en qué área general trabaja el proyecto y qué industria o sector cubre?"
                ),
            }
        ]


def reset_chat_state():
    """Reinicia el diálogo de captura"""
    st.session_state.pop("slot_values", None)
    st.session_state.pop("chat_messages", None)
    init_chat_state()


def slot_is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set)):
        return len(value) == 0
    return False


def missing_slots(slots: Dict[str, Any]) -> List[str]:
    return [k for k in REQUIRED_SLOTS if slot_is_empty(slots.get(k))]


def user_is_correcting(text: str) -> bool:
    """Detecta si el usuario está corrigiendo algo"""
    t = (text or "").lower()
    return any(p in t for p in [
        "no es", "en realidad", "más bien", "me corrijo", "corrección",
        "no, es", "no. es", "cambia", "corregir"
    ])


def build_state_summary(slots: Dict[str, Any]) -> str:
    """Genera resumen visual de lo entendido hasta ahora"""
    area = slots.get("area") or "—"
    cat = slots.get("categoria") or "—"
    inds = ", ".join(slots.get("industrias") or []) or "—"
    obj = (slots.get("objetivo") or "").strip()
    obj_short = obj[:160] + ("..." if len(obj) > 160 else "")
    obj_show = obj_short if obj_short else "—"

    return (
        "🧧 Lo que entendí hasta ahora\n"
        f"- Área: {area}\n"
        f"- Categoría: {cat}\n"
        f"- Industria(s): {inds}\n"
        f"- Objetivo: {obj_show}\n\n"
        "Si algo está mal, corrígeme en una frase (ej: \"no es Operaciones, es Estrategia\")."
    )


def objetivo_demasiado_generico(obj: str) -> bool:
    """Detecta si el objetivo es muy genérico y necesita refinamiento"""
    t = (obj or "").strip().lower()
    if len(t) < 25:
        return True
    # señales de genericidad
    generic = any(k in t for k in ["mejorar", "optimizar", "fortalecer", "aumentar", "potenciar"])
    has_kpi = any(k in t for k in ["%", "kpi", "ventas", "margen", "nps", "clientes", "cost", "costos", "participación", "tiempo", "productividad", "crecimiento"])
    return generic and not has_kpi


def normalize_area(raw: str) -> str:
    """Normaliza área a valores canónicos"""
    if not raw:
        return ""
    t = raw.lower()
    if any(k in t for k in ["estrateg", "negocio", "business", "corporativ"]):
        return "Estrategia"
    if any(k in t for k in ["person", "rrhh", "recursos humanos", "talento", "cultura", "clima"]):
        return "Personas"
    if any(k in t for k in ["operacion", "proces", "eficien", "supply", "lean", "transform"]):
        return "Operaciones"
    # si ya viene exactamente:
    for a in AREAS:
        if a.lower() == t.strip():
            return a
    return raw.strip().title()


def normalize_industries(raw_list) -> List[str]:
    """Normaliza industrias a valores de la lista o mantiene custom"""
    out = []
    for x in (raw_list or []):
        if not x:
            continue
        xx = x.strip()
        # match por contains (tolerante)
        hit = None
        for ind in INDUSTRIAS:
            if ind.lower() in xx.lower() or xx.lower() in ind.lower():
                hit = ind
                break
        out.append(hit or xx)
    # unique preservando orden
    seen = set()
    final = []
    for i in out:
        if i not in seen:
            seen.add(i)
            final.append(i)
    return final


def normalize_categoria(raw: str, area: str) -> str:
    """Normaliza categoría según área (tolerante con acentos y case)"""
    if not raw:
        return ""
    
    import unicodedata
    
    def remove_accents(s):
        """Quita acentos para comparación tolerante"""
        return ''.join(c for c in unicodedata.normalize('NFD', s) 
                      if unicodedata.category(c) != 'Mn')
    
    # si ya coincide exacto con lista del área, ok
    if area in CATEGORIAS and raw in CATEGORIAS[area]:
        return raw
    
    # match tolerante (sin acentos, sin case)
    raw_clean = remove_accents(raw.lower().strip())
    for cat in CATEGORIAS.get(area, []):
        cat_clean = remove_accents(cat.lower())
        if cat_clean in raw_clean or raw_clean in cat_clean:
            return cat
    
    return raw.strip().title()


def merge_slots(current: Dict[str, Any], new_values: Dict[str, Any], user_text: str = "") -> Dict[str, Any]:
    """Merge con thresholds dinámicos: bajo cuando vacío, alto cuando sobre-escribe"""
    merged = current.copy()
    conf = new_values.get("_confidence", {}) or {}
    correcting = user_is_correcting(user_text)

    def accept(slot: str, low_when_empty: float, high_when_overwrite: float) -> bool:
        proposed = new_values.get(slot)
        if slot_is_empty(proposed):
            return False

        c = float(conf.get(slot, 0) or 0)
        cur_empty = slot_is_empty(merged.get(slot))

        # Si el usuario está corrigiendo, prioriza el cambio con umbral muy bajo
        if correcting:
            return c >= 0.30

        # Si el campo estaba vacío, acepta con umbral medio (para no estancarte)
        if cur_empty:
            return c >= low_when_empty

        # Si ya tenía valor, exige alto para no sobre-escribir por ruido
        return c >= high_when_overwrite

    if accept("area", low_when_empty=CONF_CONFIRM, high_when_overwrite=CONF_ACCEPT):
        merged["area"] = new_values["area"]

    if accept("categoria", low_when_empty=CONF_CONFIRM, high_when_overwrite=CONF_ACCEPT):
        merged["categoria"] = new_values["categoria"]

    if accept("industrias", low_when_empty=CONF_CONFIRM, high_when_overwrite=CONF_ACCEPT):
        merged["industrias"] = normalize_industries(
            list(dict.fromkeys((merged.get("industrias", []) + (new_values.get("industrias") or []))))
        )

    # Objetivo: más permisivo al llenar por primera vez
    if accept("objetivo", low_when_empty=0.35, high_when_overwrite=CONF_ACCEPT):
        merged["objetivo"] = normalize_text(new_values.get("objetivo") or "")

    # metadata
    merged["_confidence"] = conf
    merged["_evidence"] = new_values.get("_evidence", {})
    merged["_need_clarification"] = new_values.get("_need_clarification", False)
    merged["_clarifying_question"] = new_values.get("_clarifying_question", "")

    return merged


def merge_slots(current: Dict[str, Any], new_values: Dict[str, Any], user_text: str = "") -> Dict[str, Any]:
    """Merge más agresivo: si llega info nueva, se guarda. Prioriza fluidez."""
    merged = current.copy()
    
    # Si detectamos un área y es diferente a la vacía o actual, actualizamos
    if new_values.get("area"):
        merged["area"] = new_values["area"]
        
    if new_values.get("categoria"):
        merged["categoria"] = new_values["categoria"]
        
    if new_values.get("industrias"):
        # Unir listas sin duplicados
        current_inds = set(merged.get("industrias", []))
        new_inds = set(new_values["industrias"])
        merged["industrias"] = normalize_industries(list(current_inds.union(new_inds)))

    # Lógica para Objetivo: Si el nuevo es más descriptivo o diferente, lo tomamos
    new_obj = (new_values.get("objetivo") or "").strip()
    curr_obj = (merged.get("objetivo") or "").strip()
    
    if new_obj:
        # Si teníamos vacío, llenamos. Si es nuevo, reemplazamos o concatenamos.
        if not curr_obj:
            merged["objetivo"] = new_obj
        elif new_obj not in curr_obj: 
            # Si el usuario añade info, reemplazamos para que la IA refine el texto
            merged["objetivo"] = new_obj

    return merged


def infer_slots_from_text(client: OpenAI, text: str, current_slots: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracción robusta compatible con modelos nuevos (o1/o3) que no soportan temperature=0.
    """
    # Prompt del Sistema
    system_prompt = """
    Eres un analista experto (CIS). Tu trabajo es mapear el input del usuario a categorías estructuradas.
    
    TUS LISTAS MAESTRAS:
    1. AREAS PERMITIDAS: ["Estrategia", "Personas", "Operaciones", "Marketing", "Finanzas"]
    2. INDUSTRIAS PERMITIDAS: Usa el listado provisto. Si el usuario dice "helados", mapea a "Alimentación".
    
    INSTRUCCIONES:
    - Analiza el texto del usuario y deduce Área, Industria y Objetivo.
    - Si hay conflicto, prioriza el contexto más reciente.
    - Objetivo: Resume la intención del usuario en una frase corta.

    Output JSON estricto:
    {
      "area": "String exacto de la lista o null",
      "categoria": "String o null",
      "industrias": ["String exacto de la lista"],
      "objetivo": "String resumen"
    }
    """

    user_prompt = f"""
    ESTADO ACTUAL: {json.dumps(current_slots, ensure_ascii=False)}
    INPUT USUARIO: "{text}"
    LISTA INDUSTRIAS: {json.dumps(INDUSTRIAS, ensure_ascii=False)}
    """

    try:
        # NOTA: Eliminamos 'temperature' porque los modelos o1/reasoning no lo soportan.
        # Si usas o1-preview o o1-mini, a veces response_format={"type": "json_object"} también da error.
        # Si te vuelve a fallar, elimina también la línea de response_format.
        r = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0,  <-- ELIMINADO: CAUSABA EL ERROR
            response_format={"type": "json_object"} 
        )
        extracted = json.loads((r.choices[0].message.content or "{}").strip())
    except Exception as e:
        print(f"⚠️ Error CRÍTICO en inferencia: {e}")
        # En caso de error, intentamos salvar lo que el usuario escribió como objetivo
        # para no ignorarlo completamente.
        extracted = {"objetivo": text}

    # Normalización básica
    area = normalize_area(extracted.get("area") or "")
    
    inds_raw = extracted.get("industrias") or []
    inds = normalize_industries(inds_raw)
    if not inds and inds_raw: 
         inds = inds_raw 

    return {
        "area": area,
        "categoria": extracted.get("categoria") or "",
        "industrias": inds,
        "objetivo": extracted.get("objetivo") or ""
    }



def build_assistant_reply(slots: Dict[str, Any], prev_slots: Dict[str, Any]) -> str:
    # 1. Detectar cambios (Feedback positivo)
    changes = []
    if slots.get("area") and slots["area"] != prev_slots.get("area"):
        changes.append(f"entendido que es un tema de **{slots['area']}**")
    
    # Comprobamos si hay industrias nuevas
    current_inds = set(slots.get("industrias", []))
    prev_inds = set(prev_slots.get("industrias", []))
    if current_inds and current_inds != prev_inds:
        changes.append(f"enfocado en la industria **{list(current_inds)[0]}**")

    # 2. SALUDO INICIAL (Lógica corregida)
    # Solo saludamos si NO hay área, NI industria, NI objetivo.
    has_content = slots.get("area") or slots.get("industrias") or slots.get("objetivo")
    
    if not has_content:
        return "¡Hola! Para buscar proyectos similares, cuéntame: ¿cuál es el desafío principal o qué problema buscan resolver?"

    greeting = ""
    if changes:
        greeting = f"Perfecto, {', y '.join(changes)}. "

    # 3. Resumen Visual
    summary_lines = []
    if slots.get('area'): summary_lines.append(f"🔹 **Área:** {slots['area']}")
    if slots.get('categoria'): summary_lines.append(f"🔸 **Enfoque:** {slots['categoria']}")
    if slots.get('industrias'): summary_lines.append(f"🏭 **Industria:** {', '.join(slots['industrias'])}")
    
    if slots.get('objetivo'):
        obj = slots['objetivo']
        obj_short = obj if len(obj) < 85 else obj[:82] + "..."
        summary_lines.append(f"🎯 **Meta:** {obj_short}")

    summary_box = "\n".join(summary_lines)

    # 4. Siguiente Pregunta (Prioridad Lógica)
    # Definimos qué falta
    miss_area = not slots.get("area")
    miss_ind = not slots.get("industrias")
    miss_obj = not slots.get("objetivo")
    
    question = ""

    if not miss_area and not miss_ind and not miss_obj:
        question = "¡Tengo todo lo necesario! ¿Quieres agregar algún detalle más o le doy a **Buscar Proyectos**?"
    
    elif miss_area:
        question = "Basado en lo que me cuentas, ¿esto se ajusta más a **Estrategia**, **Operaciones** o **Personas**?"
        
    elif miss_ind:
        question = "¿Para qué **industria** o sector es este proyecto?"
        
    elif miss_obj:
        question = "¿Cuál es el resultado concreto que esperan lograr?"
    
    else:
        # Fallback por si falta la categoría pero tenemos área
        question = "¿Podrías darme un poco más de detalle para afinar la categoría?"

    # 5. Ensamblaje final
    parts = []
    if greeting: parts.append(greeting)
    if summary_lines: parts.append(summary_box + "\n\n---")
    parts.append(question)
    
    return "\n".join(parts)


# ========================================
# APLICACIÓN PRINCIPAL
# ========================================

st.set_page_config(
    page_title="CIS • Asistente de Proyectos", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

css_cis()

# Header
st.markdown(
    """
    <div class='cis-header'>
        <h1>🎯 Asistente CIS · Recomendador Inteligente con GPT-5</h1>
        <p>Sistema híbrido de búsqueda semántica + razonamiento estratégico con IA</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Validación de API Key
if not OPENAI_API_KEY:
    st.error("⚠️ Define la variable de entorno OPENAI_API_KEY antes de ejecutar la aplicación.")
    st.info("💡 En PowerShell ejecuta: `$env:OPENAI_API_KEY=\"tu-api-key\"`")
    st.stop()

# Inicialización
client = OpenAI(api_key=OPENAI_API_KEY)
projects = load_projects(PROJECTS_PATH)

if not projects:
    st.warning("⚠️ No se encontraron proyectos en el archivo JSON")
    st.stop()

st.success(f"✅ Cargados **{len(projects)} proyectos** de la base de datos")
init_chat_state()

# ========================================
# SIDEBAR - SIMPLIFICADO
# ========================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.markdown("### 🎯 Cantidad de Resultados")
    top_k = st.slider(
        "Top-K",
        min_value=1, 
        max_value=10, 
        value=TOP_K_DEFAULT, 
        step=1,
        help="Número de proyectos a mostrar en resultados finales"
    )
    
    st.caption("💡 Recomendamos 3-5 proyectos para análisis óptimo")
    
    st.divider()
    
    st.markdown("### 🧠 Modo de Búsqueda")
    
    usar_gpt_rerank = st.checkbox(
        "🚀 Activar Re-Ranking Inteligente (GPT-5)",
        value=True,
        help="GPT-5 analiza y re-ordena los resultados con razonamiento estratégico"
    )
    
    if usar_gpt_rerank:
        st.success("✅ **Modo Avanzado**: GPT-5 razonará sobre los mejores matches")
        st.caption(f"📊 Analizará top-{CANDIDATE_POOL_SIZE} candidatos semánticos y seleccionará los {top_k} más estratégicos")
    else:
        st.info("📐 **Modo Básico**: Solo similitud semántica (embeddings)")
    
    st.divider()
    
    st.markdown("### 📊 Ordenamiento Adicional")
    orden = st.radio(
        "Criterio secundario:",
        options=["Score Principal", "Categorización Alfabética"],
        help="Aplica después del análisis principal"
    )
    
    st.divider()
    
    st.markdown("### ℹ️ Sobre Esta Herramienta")
    st.markdown("""
    **Sistema Híbrido:**
    
    1️⃣ **Embeddings** pre-seleccionan candidatos similares
    
    2️⃣ **GPT-5** analiza estratégicamente y re-rankea
    
    3️⃣ **Explicación** contextual de por qué son relevantes
    
    Resultado: Precisión superior combinando IA semántica + razonamiento lógico.
    """)
    
    st.divider()
    
    st.markdown("### 🔧 Detalles Técnicos")
    st.markdown(f"""
    - **Proyectos:** {len(projects)}
    - **Modelo Principal:** {MODEL_CHAT}
    - **Embeddings:** {MODEL_EMB}
    - **Pool de Análisis:** {CANDIDATE_POOL_SIZE}
    """)
    
    st.divider()
    
    with st.expander("🧪 Debug (última extracción)", expanded=False):
        st.json({
            "slots_actuales": st.session_state.slot_values,
            "ultima_extraccion": st.session_state.get("last_extracted", {})
        })

# ========================================
# INFO BOX
# ========================================
st.markdown(
    """
    <div class="cis-info-box">
        <strong>📝 Cómo usar el asistente</strong>
        <p style="margin-top: 0.5rem;">
        Conversemos para completar el <strong>Bloque 1</strong> (obligatorio). 
        El asistente capturará área, categoría, industria(s) y objetivo; luego usará 
        <strong>embeddings semánticos</strong> y <strong>GPT-5</strong> para seleccionar los proyectos más relevantes.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================================
# BLOQUE 1: INFORMACIÓN BASE
# ========================================
st.markdown(
    '<div style="background: rgba(0, 86, 112, 0.2); padding: 1.8rem; border-radius: 14px; '
    'margin: 1.5rem 0; border-left: 6px solid #005670;">'
    '<h3 style="color: #FFFFFF; margin: 0; font-weight: 700; font-size: 1.4rem;">📋 Bloque 1: Información Base (Conversación)</h3>'
    '</div>', 
    unsafe_allow_html=True
)

# Estado de captura
st.markdown("**Estado de captura**")
slots_view = st.session_state.slot_values
st.markdown(
    f"- Área: {slots_view['area'] or '—'}\n"
    f"- Categoría: {slots_view['categoria'] or '—'}\n"
    f"- Industria(s): {', '.join(slots_view['industrias']) or '—'}\n"
    f"- Objetivo: {slots_view['objetivo'] or '—'}"
)

st.button("🔄 Reiniciar diálogo", on_click=reset_chat_state)

st.markdown("---")
st.markdown("### 💬 Conversación")

# Historial DENTRO del cuadro (scrolleable) - Usando components para mayor robustez
import streamlit.components.v1 as components

html_content = render_chat_html(st.session_state.chat_messages)
html_with_scroll = html_content + """
<script>
  const box = document.querySelector('#chat-box');
  if (box) { box.scrollTop = box.scrollHeight; }
</script>
"""

components.html(html_with_scroll, height=560, scrolling=True)

# Input FUERA del cuadro, justo debajo
user_chat = st.chat_input("Escribe tu respuesta aquí...")
if user_chat:
    st.session_state.last_user_message = user_chat
    st.session_state.chat_messages.append({"role": "user", "content": user_chat})
    
    prev_slots = st.session_state.slot_values.copy()
    extracted = infer_slots_from_text(client, user_chat, prev_slots)
    st.session_state.last_extracted = extracted  # Para debug
    
    merged = merge_slots(prev_slots, extracted, user_chat)
    st.session_state.slot_values = merged
    
    reply = build_assistant_reply(st.session_state.slot_values, prev_slots)
    st.session_state.last_bot_question = reply.strip()
    
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    st.rerun()

st.divider()

# ========================================
# BLOQUE 2: EN DESARROLLO
# ========================================
st.markdown(
    '<div style="background: rgba(242, 0, 52, 0.1); padding: 1.5rem; border-radius: 12px; '
    'margin: 1rem 0; border-left: 5px solid #F20034; border: 2px dashed #F20034;">'
    '<h3 style="color: #FF4D6D; margin: 0 0 0.5rem 0; font-weight: 700;">🚧 Bloque 2: Información Adicional (En Desarrollo)</h3>'
    '<p style="color: #E5E7EB; margin: 0; font-size: 1.05rem;">Esta sección estará disponible próximamente. Por ahora, el análisis se basa únicamente en la información del Bloque 1.</p>'
    '</div>', 
    unsafe_allow_html=True
)

st.divider()

# ========================================
# BOTÓN DE BÚSQUEDA
# ========================================
st.markdown("<br>", unsafe_allow_html=True)
search_btn = st.button("🔍 Buscar Proyectos Similares con IA", type="primary", use_container_width=True)

# ========================================
# PRECARGA DE EMBEDDINGS
# ========================================
@st.cache_resource(show_spinner="🔄 Cargando embeddings de proyectos...")
def _embeddings_once(_client: OpenAI, _projects: List[Dict[str, Any]]):
    return ensure_embeddings(_client, _projects)

emb_matrix = _embeddings_once(client, projects)

# ========================================
# PROCESAMIENTO DE BÚSQUEDA
# ========================================
if search_btn:
    # Tomar datos desde el diálogo
    slots_for_search = st.session_state.slot_values
    area_seleccionada = slots_for_search.get("area", "")
    categoria_seleccionada = slots_for_search.get("categoria", "")
    industrias_seleccionadas = slots_for_search.get("industrias", [])
    objetivo_proyecto = slots_for_search.get("objetivo", "")

    # Validación
    errores = []
    
    if not area_seleccionada:
        errores.append("❌ Selecciona un área (Pregunta 1)")
    
    if not categoria_seleccionada:
        errores.append("❌ Selecciona una categoría (Pregunta 2)")
    
    if not industrias_seleccionadas:
        errores.append("❌ Selecciona al menos una industria (Pregunta 3)")
    
    if not objetivo_proyecto.strip():
        errores.append("❌ Describe el objetivo del proyecto (Pregunta 4)")
    
    if errores:
        for error in errores:
            st.error(error)
        st.warning("⚠️ Por favor completa todos los campos obligatorios del Bloque 1")
    else:
        # Construir datos
        form_data = {
            "area": area_seleccionada,
            "categoria": categoria_seleccionada,
            "industrias": industrias_seleccionadas,
            "objetivo": objetivo_proyecto,
            "duracion": None,
            "entregables": [],
            "enfoque": [],
            "comentarios": ""
        }
        
        user_prompt = build_user_prompt(form_data)
        
        # FASE 1: Pre-selección semántica
        with st.spinner("🔍 Fase 1/2: Analizando similitud semántica con embeddings..."):
            q_vec = embed_texts(client, [user_prompt])[0]
            
            sims = []
            for idx, p in enumerate(projects):
                sim = cosine_sim(q_vec, emb_matrix[idx])
                sims.append((idx, sim, p))
            
            sims.sort(key=lambda x: x[1], reverse=True)
            
            # Determinar pool de candidatos
            pool_size = CANDIDATE_POOL_SIZE if usar_gpt_rerank else top_k
            candidates = sims[:pool_size]
        
        # FASE 2: Re-ranking con GPT (si está activado)
        if usar_gpt_rerank:
            with st.spinner(f"🧠 Fase 2/2: GPT-5 analizando estratégicamente {len(candidates)} candidatos..."):
                ranked_results, analisis_general = llm_rerank_projects(
                    client, 
                    user_prompt, 
                    candidates, 
                    top_k
                )
            
            # Ordenamiento adicional si se seleccionó
            if "Categorización" in orden:
                ranked_results.sort(key=lambda x: (
                    x["project"].get("catalogacion", "ZZZ"),
                    -(x.get("gpt_score") or x.get("semantic_score", 0))
                ))
        else:
            # Modo básico: solo semántico
            if "Categorización" in orden:
                candidates.sort(key=lambda x: (x[2].get("catalogacion", "ZZZ"), -x[1]))
            
            ranked_results = []
            for idx, semantic_score, project in candidates[:top_k]:
                ranked_results.append({
                    "project": project,
                    "semantic_score": semantic_score,
                    "gpt_score": None,
                    "reasoning": None,
                    "strengths": [],
                    "weaknesses": []
                })
            analisis_general = ""
        
        # FASE 3: Explicación contextual
        with st.spinner("💭 Generando explicación contextual..."):
            why = llm_explain_selection(client, user_prompt, ranked_results)
        
        # Mostrar análisis general (si GPT lo generó)
        if analisis_general and usar_gpt_rerank:
            st.markdown(
                f"""
                <div class='cis-why-box'>
                    <strong>🎯 Análisis Estratégico GPT-5</strong>
                    <p>{analisis_general}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Mostrar explicación contextual
        st.markdown(
            f"""
            <div class='cis-why-box'>
                <strong>📊 ¿Por qué estos proyectos son relevantes?</strong>
                <p>{why}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Header de resultados
        modo_texto = "GPT-5 Re-Ranking" if usar_gpt_rerank else "Similitud Semántica"
        st.markdown(
            f'<div style="background: rgba(242, 0, 52, 0.15); padding: 1.8rem; border-radius: 14px; '
            f'margin: 2rem 0 1.5rem 0; border-left: 6px solid #F20034;">'
            f'<h3 style="color: #FFFFFF; margin: 0; font-weight: 700; font-size: 1.5rem;">'
            f'📋 {len(ranked_results)} Proyecto{"s" if len(ranked_results) > 1 else ""} Seleccionado{"s" if len(ranked_results) > 1 else ""}'
            f'</h3>'
            f'<p style="color: #E5E7EB; margin: 0.5rem 0 0 0; font-size: 1.05rem;">Método: <strong>{modo_texto}</strong></p>'
            f'</div>', 
            unsafe_allow_html=True
        )
        
        # Renderizar resultados
        for idx, item in enumerate(ranked_results, 1):
            p = item["project"]
            gpt_score = item.get("gpt_score")
            semantic_score = item.get("semantic_score", 0)
            reasoning = item.get("reasoning")
            strengths = item.get("strengths", [])
            weaknesses = item.get("weaknesses", [])
            
            nombre_cliente = p.get('nombre_cliente', 'Sin nombre')
            industria = p.get("industria", "N/A")
            nombre_proyecto = p.get("nombre_proyecto", "N/A")
            area_negocio = p.get("area_negocio", "N/A")
            catalogacion = p.get("catalogacion", "N/A")
            anio = p.get("anio", "N/A")
            
            # Card
            st.markdown("<div class='cis-result-card'>", unsafe_allow_html=True)
            
            # Título con badge de score
            if gpt_score:
                badge_html = f"<span class='gpt-badge'>🧠 GPT-5: {gpt_score}/100</span>"
            else:
                badge_html = f"<span class='gpt-badge'>📐 {semantic_score:.1%}</span>"
            
            st.markdown(
                f"<div class='cis-client-name'>➔ {idx}. {nombre_cliente} {badge_html}</div>", 
                unsafe_allow_html=True
            )
            
            # Metadata
            st.markdown(
                f"<div class='cis-meta-item'><strong>📁 Proyecto:</strong> {nombre_proyecto}</div>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='cis-meta-item'><strong>🏢 Industria:</strong> {industria}</div>", 
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div class='cis-meta-item'><strong>🎯 Área:</strong> {area_negocio} | "
                f"<strong>Tipo:</strong> {catalogacion}</div>", 
                unsafe_allow_html=True
            )
            
            if anio and anio != "N/A":
                st.markdown(
                    f"<div class='cis-meta-item'><strong>📅 Año:</strong> {anio}</div>", 
                    unsafe_allow_html=True
                )
            
            # Score semántico adicional (si hay GPT score)
            if gpt_score:
                st.markdown(
                    f"<div class='cis-meta-item'><strong>📊 Similitud Semántica Base:</strong> "
                    f"<span style='color: #9BB3BC;'>{semantic_score:.1%}</span></div>", 
                    unsafe_allow_html=True
                )
            
            # Razonamiento GPT-5 (si existe)
            if reasoning and usar_gpt_rerank:
                reasoning_html = f"<div class='gpt-reasoning'><strong>🧠 Razonamiento GPT-5:</strong><p>{reasoning}</p>"
                
                if strengths:
                    reasoning_html += "<strong style='color: #00D4FF; display: block; margin-top: 0.8rem;'>✓ Fortalezas:</strong><ul>"
                    for s in strengths:
                        reasoning_html += f"<li>{s}</li>"
                    reasoning_html += "</ul>"
                
                if weaknesses:
                    reasoning_html += "<strong style='color: #FF4D6D; display: block; margin-top: 0.5rem;'>⚠ Consideraciones:</strong><ul>"
                    for w in weaknesses:
                        reasoning_html += f"<li>{w}</li>"
                    reasoning_html += "</ul>"
                
                reasoning_html += "</div>"
                st.markdown(reasoning_html, unsafe_allow_html=True)
            
            # Propuesta económica
            pvp = safe_float_conversion(p.get("pvp"))
            moneda = p.get("moneda", "UF")
            
            if pvp > 0:
                econ_html = (
                    "<div class='cis-economic-box'>"
                    "<strong>💰 Propuesta Económica:</strong><br>"
                    f"<div style='margin-left: 1rem; margin-top: 0.4rem;'>"
                    f"• Valor del Proyecto: <strong>{pvp:,.0f} {moneda}</strong>"
                    f"</div>"
                    "</div>"
                )
                st.markdown(econ_html, unsafe_allow_html=True)
            
            # Extracto técnico - OBJETIVOS COMPLETOS
            objetivo_general = (p.get("objetivo_general") or "").strip()
            objetivos_especificos = (p.get("objetivos_especificos") or "").strip()
            
            if objetivo_general or objetivos_especificos:
                excerpt_html = "<div class='cis-excerpt'>"
                
                if objetivo_general:
                    excerpt_html += f"<strong>Objetivo General:</strong><br>{objetivo_general}<br><br>"
                
                if objetivos_especificos:
                    # MOSTRAR COMPLETO (no truncar)
                    excerpt_html += f"<strong>Objetivos Específicos:</strong><br>{objetivos_especificos}"
                
                excerpt_html += "</div>"
                st.markdown(excerpt_html, unsafe_allow_html=True)
            
            # Enlaces
            def link_button(label: str, url: Optional[str]) -> str:
                if url:
                    return f"<a href='{url}' target='_blank'>{label}</a>"
                return f"<a class='disabled' aria-disabled='true'>{label}</a>"

            links = [
                link_button("📄 Propuesta Técnica", p.get("url_tecnica")),
                link_button("💰 Propuesta Económica", p.get("url_economica")),
                link_button("💳 Factura", p.get("url_factura")),
            ]
            
            if links:
                st.markdown(
                    f"<div class='cis-links'>{''.join(links)}</div>", 
                    unsafe_allow_html=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

