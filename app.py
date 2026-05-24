"""
App RAG de Gerardo Schmedling Torres
Chat que responde con citas textuales de las enseñanzas de Schmedling.

Stack: LangChain + FAISS + NVIDIA AI Endpoints + Gradio

Uso:
    export NVIDIA_API_KEY=nvapi-xxxxx
    python app.py

Luego abre http://localhost:7860 en tu navegador.
"""

import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import gradio as gr

# Configuración
FAISS_INDEX_DIR = Path("./faiss_index")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM
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
    print("✅ Índice cargado correctamente")
    return vectorstore


def create_rag_chain(vectorstore):
    """Crea la cadena de RAG usando LCEL."""
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        temperature=0.1,
        max_tokens=1500,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "CONTEXTO DE LOS MATERIALES DE SCHMEDLING:\n{context}\n\nPREGUNTA DEL USUARIO:\n{question}\n\nRESPUESTA (con citas textuales):"),
    ])

    def format_docs(docs):
        formatted = []
        for doc in docs:
            meta = doc.metadata
            header = f"[Fuente: {meta.get('filename', '?')}, Módulo: {meta.get('module', '?')}, Página: {meta.get('page', '?')}]"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def format_sources(source_docs):
    """Formatea las fuentes para mostrar al usuario."""
    sources = []
    seen = set()
    for doc in source_docs:
        meta = doc.metadata
        key = f"{meta.get('filename', 'unknown')}:p{meta.get('page', '?')}"
        if key not in seen:
            seen.add(key)
            sources.append(
                f"📄 {meta.get('filename', 'Archivo desconocido')} "
                f"(Módulo: {meta.get('module', 'N/A')}, "
                f"Página: {meta.get('page', '?')}/{meta.get('total_pages', '?')})"
            )
    return "\n".join(sources)


def chat_fn(message, history):
    """Función principal del chat."""
    if not message.strip():
        return "Por favor, escribe una pregunta sobre las enseñanzas de Gerardo Schmedling Torres."

    try:
        # Get source documents
        source_docs = retriever.invoke(message)
        # Get answer
        answer = rag_chain.invoke(message)
        sources = format_sources(source_docs)

        response = f"{answer}\n\n---\n**Fuentes consultadas:**\n{sources}"
        return response
    except Exception as e:
        return f"Error al procesar tu pregunta: {str(e)}\n\nVerifica que tienes configurada la API key (NVIDIA_API_KEY o OPENAI_API_KEY)."


def create_interface():
    """Crea la interfaz de Gradio."""
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="🦋 Chat — Enseñanzas de Gerardo Schmedling Torres",
        description="Escuela de Magia del Amor\n\nPregunta sobre cualquier tema de los 17 módulos de comprensión.\nLas respuestas incluyen **citas textuales** con referencia a la fuente exacta.",
        examples=[
            "¿Qué es la Aceptología?",
            "¿Cuáles son las 7 Herramientas del Amor?",
            "¿Qué dice Gerardo sobre el sufrimiento?",
            "¿Cuáles son las Leyes Universales?",
            "¿Qué es la Alquimia del Pensamiento?",
            "¿Cómo se aplica la Ley de Correspondencia?",
        ],
    )
    return demo


if __name__ == "__main__":
    if not LLM_API_KEY:
        print("⚠️  ADVERTENCIA: No se encontró API key.")
        print("   Configura una de estas variables de entorno:")
        print("   - NVIDIA_API_KEY (para NVIDIA endpoints, gratis en build.nvidia.com)")
        print("   - OPENAI_API_KEY (para OpenAI)")
        print()

    print("🚀 Iniciando Chat de Schmedling...")
    vectorstore = load_vectorstore()
    rag_chain, retriever = create_rag_chain(vectorstore)

    demo = create_interface()
    print("\n🌐 Abriendo en http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
