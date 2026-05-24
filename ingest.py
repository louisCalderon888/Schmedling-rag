"""
Paso 1: Ingestar los PDFs de Schmedling en una base vectorial FAISS.

Este script:
1. Lee todos los PDFs de la carpeta pdfs_schmedling/
2. Extrae el texto con PyMuPDF
3. Divide en chunks con metadata (módulo, archivo, página)
4. Genera embeddings con sentence-transformers (local, gratis)
5. Guarda el índice FAISS en disco

Uso:
    python ingest.py
"""

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import fitz  # PyMuPDF


# Configuración
PDF_DIR = Path("./pdfs_schmedling")
FAISS_INDEX_DIR = Path("./faiss_index")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extract_text_from_pdf(pdf_path: Path) -> list[Document]:
    """Extrae texto de un PDF y retorna Documents con metadata."""
    documents = []
    try:
        doc = fitz.open(str(pdf_path))
        # Determinar módulo desde la carpeta
        module_name = pdf_path.parent.name
        filename = pdf_path.name

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "module": module_name,
                        "filename": filename,
                        "page": page_num + 1,
                        "total_pages": len(doc),
                    }
                ))
        doc.close()
    except Exception as e:
        print(f"  ❌ Error procesando {pdf_path.name}: {e}")

    return documents


def main():
    print("=" * 60)
    print("📚 INGESTA DE PDFs DE GERARDO SCHMEDLING TORRES")
    print("=" * 60)

    # 1. Encontrar todos los PDFs
    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"\n📁 PDFs encontrados: {len(pdf_files)}")

    # 2. Extraer texto de todos los PDFs
    print("\n📖 Extrayendo texto de los PDFs...")
    all_documents = []
    for i, pdf_path in enumerate(pdf_files, 1):
        docs = extract_text_from_pdf(pdf_path)
        all_documents.extend(docs)
        if i % 10 == 0 or i == len(pdf_files):
            print(f"  Procesados: {i}/{len(pdf_files)} PDFs ({len(all_documents)} páginas)")

    print(f"\n✅ Total páginas extraídas: {len(all_documents)}")

    # 3. Dividir en chunks
    print(f"\n✂️  Dividiendo en chunks (tamaño={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ Total chunks creados: {len(chunks)}")

    # 4. Generar embeddings y crear índice FAISS
    print(f"\n🧠 Generando embeddings con: {EMBEDDING_MODEL}")
    print("   (Primera vez descarga el modelo, ~500MB, puede tomar unos minutos)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    print("📊 Creando índice FAISS...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # 5. Guardar índice
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))
    print(f"\n💾 Índice FAISS guardado en: {FAISS_INDEX_DIR.absolute()}")

    # Resumen
    print(f"\n{'=' * 60}")
    print(f"🎉 INGESTA COMPLETADA")
    print(f"   PDFs procesados: {len(pdf_files)}")
    print(f"   Páginas extraídas: {len(all_documents)}")
    print(f"   Chunks indexados: {len(chunks)}")
    print(f"   Índice guardado en: {FAISS_INDEX_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
