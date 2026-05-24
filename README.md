---
title: Chat Enseñanzas Schmedling
emoji: 🦋
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# Schmedling RAG — Chat de Enseñanzas de Gerardo Schmedling Torres

Chat inteligente que responde con **citas textuales** de las enseñanzas de Gerardo Schmedling Torres, basado en los materiales de la Escuela de Magia del Amor.

## Stack

- **LangChain** (LCEL) — Orquestación RAG
- **NVIDIA AI Endpoints** (Llama 3.1 70B) — LLM
- **FAISS** — Vector Store (búsqueda semántica)
- **sentence-transformers** — Embeddings multilingües
- **Gradio** — Frontend/Chat UI

## Configuración

### Variable de entorno requerida

En Hugging Face Spaces, configura el secreto `NVIDIA_API_KEY` en Settings → Repository secrets.

Obtén tu API key gratis en https://build.nvidia.com

### Desarrollo local

```bash
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-xxxxx
python app.py
```

## Estructura

```
schmedling-rag/
├── app.py                       # App de chat (RAG + Gradio)
├── requirements.txt             # Dependencias
├── faiss_index/                 # Índice vectorial FAISS
│   ├── index.faiss
│   └── index.pkl
├── descargar_pdfs_telegram.py   # Script para descargar PDFs (offline)
├── ingest.py                    # Pipeline de ingesta (offline)
└── README.md
```

## Regenerar el índice FAISS

Si necesitas regenerar el índice con nuevo material:

```bash
pip install pymupdf telethon
python descargar_pdfs_telegram.py   # Descargar PDFs de Telegram
python ingest.py                     # Crear índice FAISS
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
