"""
App RAG de Gerardo Schmedling Torres
Chat que responde con citas textuales de las enseñanzas de Schmedling.

Stack: LangChain + FAISS + BM25 + Cross-Encoder Reranker + NVIDIA AI Endpoints + Gradio

Uso:
    export NVIDIA_API_KEY=nvapi-xxxxx
    python app.py

Luego abre http://localhost:7860 en tu navegador.
"""

import os
import re
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import gradio as gr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = Path("./faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
LLM_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

SYSTEM_PROMPT = """Eres un asistente especializado ÚNICAMENTE en las enseñanzas de Gerardo Schmedling Torres y la Escuela de Magia del Amor.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con base en los fragmentos de contexto proporcionados.
2. SIEMPRE cita textualmente entre comillas « » los fragmentos relevantes del texto original.
3. Después de cada cita, indica la fuente entre paréntesis: (Fuente: [nombre del archivo], página [número]).
4. Si la información no está en el contexto proporcionado, di exactamente: "No tengo información sobre ese tema en los materiales disponibles de Schmedling."
5. NO inventes, NO parafrasees sin citar, NO agregues información que no esté en el contexto.
6. Responde en español.
7. Sé conciso pero completo. Si hay múltiples fragmentos relevantes, cita todos."""

# Module display names
MODULE_DISPLAY_NAMES = {
    "01_Sociologia_de_la_Evolucion": "01 — Sociología de la Evolución",
    "02_Leyes_Universales_Vida_Diaria": "02 — Las Leyes Universales en la Vida Diaria",
    "03_Manejo_Practico_Leyes_Universales": "03 — Manejo Práctico de las Leyes Universales",
    "04_Matematicas_del_Amor": "04 — Las Matemáticas del Amor",
    "05_Amor_y_Sexualidad": "05 — Amor y Sexualidad",
    "06_Relaciones_del_Amor": "06 — Las Relaciones del Amor",
    "07_Aceptologia": "07 — Aceptología",
    "08_Alquimia_del_Pensamiento": "08 — Alquimia del Pensamiento",
    "09_Asumiendo_la_Vida_con_Sabiduria": "09 — Asumiendo la Vida con Sabiduría",
    "10_Trascendiendo_las_Limitaciones": "10 — Trascendiendo las Limitaciones",
    "11_Incondicionalidad_Forma_de_Amar": "11 — Incondicionalidad, Una Forma de Amar",
    "12_Creacion_Divina_Realidad_Humana": "12 — Creación Divina Hecha Realidad Humana",
    "13_Encontrandote_Reino_del_Amor": "13 — Encontrándote con el Reino del Amor",
    "14_Felicidad_Familia_Pareja": "14 — La Felicidad en la Familia y la Pareja",
    "15_Escuela_de_Padres_HOUDHO": "15 — Escuela de Padres (HOUDHO)",
    "16_Tu_Eres_lo_Mejor_de_Ti_Mismo": "16 — Tú Eres lo Mejor de Ti Mismo",
    "17_Liderando_Compromiso_Prosperidad": "17 — Liderando un Compromiso con la Prosperidad",
    "Gerardo_en_PDF": "📘 Gerardo en PDF",
    "Manuales_Originales_Resumenes_Transcripciones": "📖 Manuales Originales y Transcripciones",
}

# Reverse lookup: display name → internal key
DISPLAY_TO_KEY = {v: k for k, v in MODULE_DISPLAY_NAMES.items()}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_vectorstore():
    """Carga el índice FAISS."""
    print("🧠 Cargando modelo de embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    print("📊 Cargando índice FAISS...")
    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ Índice FAISS cargado correctamente")
    return vectorstore


def build_bm25_index(vectorstore):
    """Construye un índice BM25 sobre los documentos del vectorstore."""
    print("📝 Construyendo índice BM25...")
    docs_dict = vectorstore.docstore._dict
    doc_ids = list(docs_dict.keys())
    documents = [docs_dict[did] for did in doc_ids]
    tokenized = [re.split(r"\W+", doc.page_content.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    print(f"✅ BM25 construido ({len(doc_ids)} documentos)")
    return bm25, doc_ids, documents


def load_reranker():
    """Carga el modelo cross-encoder para reranking."""
    print("🔄 Cargando modelo de reranking...")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("✅ Reranker cargado")
    return reranker


# ---------------------------------------------------------------------------
# Hybrid search + reranking
# ---------------------------------------------------------------------------

def hybrid_search(query, vectorstore, bm25, bm25_doc_ids, bm25_documents, module_filter=None, k_faiss=20, k_bm25=20, k_final=5):
    """Búsqueda híbrida: FAISS (semántica) + BM25 (keywords), con reranking."""
    # --- FAISS search ---
    if module_filter:
        faiss_filter = {"module": module_filter}
        faiss_results = vectorstore.similarity_search(query, k=k_faiss, filter=faiss_filter)
    else:
        faiss_results = vectorstore.similarity_search(query, k=k_faiss)

    # --- BM25 search ---
    tokenized_query = re.split(r"\W+", query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    scored_indices = list(enumerate(bm25_scores))
    if module_filter:
        scored_indices = [
            (i, s) for i, s in scored_indices
            if bm25_documents[i].metadata.get("module") == module_filter
        ]
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    bm25_results = [bm25_documents[i] for i, _ in scored_indices[:k_bm25]]

    # --- Merge & deduplicate ---
    seen = set()
    candidates = []
    for doc in faiss_results + bm25_results:
        key = (doc.metadata.get("filename", ""), doc.metadata.get("page", ""), doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            candidates.append(doc)

    if not candidates:
        return []

    # --- Reranking with cross-encoder ---
    pairs = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(scores, candidates))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:k_final]]


# ---------------------------------------------------------------------------
# RAG chain helpers
# ---------------------------------------------------------------------------

def format_docs(docs):
    """Formatea documentos para el prompt del LLM."""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        header = f"[Fuente: {meta.get('filename', '?')}, Módulo: {meta.get('module', '?')}, Página: {meta.get('page', '?')}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def format_sources(source_docs):
    """Formatea las fuentes para mostrar al usuario."""
    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        key = f"{meta.get('filename', 'unknown')}:p{meta.get('page', '?')}"
        if key not in seen:
            seen.add(key)
            module_display = MODULE_DISPLAY_NAMES.get(
                meta.get("module", ""), meta.get("module", "N/A")
            )
            sources.append(
                f"📄 {meta.get('filename', 'Archivo desconocido')} "
                f"({module_display}, "
                f"Página: {meta.get('page', '?')}/{meta.get('total_pages', '?')})"
            )
    return "\n".join(sources)


def _extract_text_from_content(content):
    """Extract plain text from Gradio message content (str or list of dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def build_history_context(history, max_turns=3):
    """Extrae las últimas N interacciones del historial para dar contexto conversacional."""
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines = []
    for turn in recent:
        if isinstance(turn, dict):
            role = turn.get("role", "")
            text = _extract_text_from_content(turn.get("content", ""))
            if not text:
                continue
            if role == "user":
                lines.append(f"Usuario: {text}")
            elif role == "assistant":
                clean = text.split("\n\n---\n")[0]
                if len(clean) > 300:
                    clean = clean[:300] + "..."
                lines.append(f"Asistente: {clean}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat function (streaming)
# ---------------------------------------------------------------------------

def chat_fn(message, history, module_choice):
    """Función principal del chat con streaming."""
    if not message.strip():
        yield "Por favor, escribe una pregunta sobre las enseñanzas de Gerardo Schmedling Torres."
        return

    try:
        # Resolve module filter
        module_filter = None
        if module_choice and module_choice != "📚 Todos los módulos":
            module_filter = DISPLAY_TO_KEY.get(module_choice, None)

        # Build conversational context
        history_context = build_history_context(history)
        if history_context:
            enriched_query = f"Contexto de conversación previa:\n{history_context}\n\nPregunta actual: {message}"
        else:
            enriched_query = message

        # Hybrid search + reranking
        source_docs = hybrid_search(
            enriched_query, vectorstore, bm25, bm25_doc_ids, bm25_documents,
            module_filter=module_filter,
        )

        if not source_docs:
            yield "No encontré información relevante en los materiales de Schmedling para esa pregunta."
            return

        context = format_docs(source_docs)
        sources = format_sources(source_docs)

        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "CONTEXTO DE LOS MATERIALES DE SCHMEDLING:\n{context}\n\nPREGUNTA DEL USUARIO:\n{question}\n\nRESPUESTA (con citas textuales):"),
        ])

        chain = prompt | llm | StrOutputParser()

        # Stream response
        partial_response = ""
        for chunk in chain.stream({"context": context, "question": message}):
            partial_response += chunk
            yield partial_response

        # Append sources at the end
        yield f"{partial_response}\n\n---\n**Fuentes consultadas:**\n{sources}"

    except Exception as e:
        yield f"Error al procesar tu pregunta: {str(e)}\n\nVerifica que tienes configurada la API key (NVIDIA_API_KEY o OPENAI_API_KEY)."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_interface():
    """Crea la interfaz de Gradio con filtrado por módulo y theme E.M.A."""

    module_choices = ["📚 Todos los módulos"] + [
        MODULE_DISPLAY_NAMES[k]
        for k in sorted(MODULE_DISPLAY_NAMES.keys())
    ]

    with gr.Blocks(title="Chat Schmedling — E.M.A") as demo:
        gr.Markdown(
            """
            # 🦋 Chat — Enseñanzas de Gerardo Schmedling Torres
            ### Escuela de Magia del Amor

            Pregunta sobre cualquier tema de los **17 módulos de comprensión**.
            Las respuestas incluyen **citas textuales** con referencia a la fuente exacta.

            ---
            """
        )

        with gr.Row():
            module_dropdown = gr.Dropdown(
                choices=module_choices,
                value="📚 Todos los módulos",
                label="🔍 Filtrar por módulo",
                interactive=True,
                scale=3,
            )

        chatbot = gr.Chatbot(
            height=500,
            buttons=["copy"],
            placeholder="Haz una pregunta sobre las enseñanzas de Schmedling...",
        )

        msg = gr.Textbox(
            placeholder="Escribe tu pregunta aquí...",
            label="Tu pregunta",
            show_label=False,
            scale=4,
        )

        with gr.Row():
            submit_btn = gr.Button("Enviar", variant="primary", scale=1)
            clear_btn = gr.Button("🗑️ Limpiar", scale=1)

        gr.Examples(
            examples=[
                ["¿Qué es la Aceptología?"],
                ["¿Cuáles son las 7 Herramientas del Amor?"],
                ["¿Qué dice Gerardo sobre el sufrimiento?"],
                ["¿Cuáles son las Leyes Universales?"],
                ["¿Qué es la Alquimia del Pensamiento?"],
                ["¿Cómo se aplica la Ley de Correspondencia?"],
            ],
            inputs=msg,
        )

        def user_submit(user_message, chat_history, module_choice):
            chat_history = chat_history or []
            chat_history.append({"role": "user", "content": user_message})
            return "", chat_history, chat_history, module_choice

        def extract_text(content):
            """Extract plain text from Gradio message content (str or list of dicts)."""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return " ".join(parts)
            return str(content)

        def bot_response(chat_history, module_choice):
            if not chat_history:
                return chat_history
            user_message = extract_text(chat_history[-1]["content"])
            history_for_context = chat_history[:-1]
            chat_history.append({"role": "assistant", "content": ""})
            for partial in chat_fn(user_message, history_for_context, module_choice):
                chat_history[-1]["content"] = partial
                yield chat_history

        submit_btn.click(
            user_submit,
            inputs=[msg, chatbot, module_dropdown],
            outputs=[msg, chatbot, chatbot, module_dropdown],
        ).then(
            bot_response,
            inputs=[chatbot, module_dropdown],
            outputs=[chatbot],
        )

        msg.submit(
            user_submit,
            inputs=[msg, chatbot, module_dropdown],
            outputs=[msg, chatbot, chatbot, module_dropdown],
        ).then(
            bot_response,
            inputs=[chatbot, module_dropdown],
            outputs=[chatbot],
        )

        clear_btn.click(lambda: ([], None), outputs=[chatbot, msg])

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not LLM_API_KEY:
        print("⚠️  ADVERTENCIA: No se encontró API key.")
        print("   Configura una de estas variables de entorno:")
        print("   - NVIDIA_API_KEY (para NVIDIA endpoints, gratis en build.nvidia.com)")
        print("   - OPENAI_API_KEY (para OpenAI)")
        print()

    print("🚀 Iniciando Chat de Schmedling...")

    # Load components
    vectorstore = load_vectorstore()
    bm25, bm25_doc_ids, bm25_documents = build_bm25_index(vectorstore)
    reranker = load_reranker()

    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        temperature=0.1,
        max_tokens=1500,
        streaming=True,
    )

    demo = create_interface()
    print("\n🌐 Abriendo en http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
