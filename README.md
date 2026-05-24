# Schmedling RAG — Chat de Enseñanzas de Gerardo Schmedling Torres

Chat inteligente que responde con **citas textuales** de las enseñanzas de Gerardo Schmedling Torres, basado en los materiales de la Escuela de Magia del Amor.

## Stack

- **LangChain** (LCEL) — Orquestación RAG
- **NVIDIA AI Endpoints** (Llama 3.1 70B) — LLM
- **FAISS** — Vector Store (búsqueda semántica)
- **PyMuPDF** — Parseo de PDFs
- **sentence-transformers** — Embeddings multilingües
- **Gradio** — Frontend/Chat UI

## Estructura del Proyecto

```
schmedling-rag/
├── descargar_pdfs_telegram.py   # Descarga PDFs de canales de Telegram
├── ingest.py                    # Parsea PDFs → chunks → embeddings → FAISS
├── app.py                       # App de chat (RAG + Gradio)
├── requirements.txt             # Dependencias
├── pdfs_schmedling/             # PDFs por módulo (se genera, no se sube a git)
├── faiss_index/                 # Índice vectorial (se genera, no se sube a git)
└── README.md
```

## Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Descargar PDFs de Telegram

```bash
export TELEGRAM_API_ID=tu_api_id
export TELEGRAM_API_HASH=tu_api_hash
export TELEGRAM_PHONE=+57XXXXXXXXXX

python descargar_pdfs_telegram.py
```

Obtén `api_id` y `api_hash` en https://my.telegram.org → "API Development Tools".

### 3. Crear el índice vectorial

```bash
python ingest.py
```

Esto procesa los 143 PDFs → 12,601 páginas → 68,783 chunks indexados en FAISS.

### 4. Lanzar el chat

```bash
export NVIDIA_API_KEY=nvapi-xxxxx   # Obtén gratis en https://build.nvidia.com
python app.py
```

Abre http://localhost:7860 en tu navegador.

## Ejemplo de uso

**Pregunta:** ¿Qué es la Aceptología?

**Respuesta:**
> La Aceptología es una Nueva Ciencia que, cuando estamos listos para comprenderla, permite cumplir el propósito general que tienen todos los seres humanos, para ENCONTRAR PLENA SATISFACCIÓN EN SUS VIDAS.
> (Fuente: 07-Aceptologia completo Transcripciones Actualizada 2026.pdf, Módulo: 07_Aceptologia, Página: 5)

## Configuración alternativa (OpenAI)

```bash
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=sk-xxxxx
python app.py
```

## Módulos de E.M.A (Escuela de Magia del Amor)

1. Sociología de la Evolución
2. Las Leyes Universales en la Vida Diaria
3. Manejo Práctico de las Leyes Universales
4. Las Matemáticas del Amor
5. Amor y Sexualidad
6. Las Relaciones del Amor
7. Aceptología
8. Alquimia del Pensamiento
9. Asumiendo la Vida con Sabiduría
10. Trascendiendo las Limitaciones
11. Incondicionalidad, Una Forma de Amar
12. Creación Divina Hecha Realidad Humana
13. Encontrándote con el Reino del Amor
14. La Felicidad en la Familia y la Pareja
15. Escuela de Padres (HOUDHO)
16. Tú Eres lo Mejor de Ti Mismo
17. Liderando un Compromiso con la Prosperidad
