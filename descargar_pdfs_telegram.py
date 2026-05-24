"""
Script para descargar PDFs de los canales de Telegram de
"Talleres de Comprensión de Amor" (Escuela de Magia del Amor - Gerardo Schmedling Torres)

Uso:
    python descargar_pdfs_telegram.py

Requisitos:
    pip install telethon

Antes de ejecutar:
    1. Ve a https://my.telegram.org
    2. Inicia sesión con tu número de teléfono
    3. Ve a "API Development Tools"
    4. Crea una app y obtén tu api_id y api_hash
    5. Reemplaza los valores abajo o configúralos como variables de entorno
"""

import os
import asyncio
import re
from pathlib import Path
from telethon import TelegramClient
from telethon.tl.types import DocumentAttributeFilename

# ============================================================
# CONFIGURACIÓN - Reemplaza con tus credenciales
# ============================================================
API_ID = int(os.environ.get("TELEGRAM_API_ID", "0"))
API_HASH = os.environ.get("TELEGRAM_API_HASH", "")
PHONE = os.environ.get("TELEGRAM_PHONE", "")  # Tu número con código de país, ej: +573001234567

# Directorio donde se guardarán los PDFs
OUTPUT_DIR = Path("./pdfs_schmedling")

# ============================================================
# CANALES A DESCARGAR (extraídos del índice del canal principal)
# ============================================================
CHANNELS = {
    # 17 Módulos de E.M.A
    "01_Sociologia_de_la_Evolucion": "https://t.me/+8Lw9M0PJddNlN2Qx",
    "02_Leyes_Universales_Vida_Diaria": "https://t.me/+-iJc7iwtoQ02MWIx",
    "03_Manejo_Practico_Leyes_Universales": "https://t.me/+Pa9okVZyzPdlZjhh",
    "04_Matematicas_del_Amor": "https://t.me/+jwl5SeTzvgJjN2Ex",
    "05_Amor_y_Sexualidad": "https://t.me/+7aMwjTKJx8c2YjQx",
    "06_Relaciones_del_Amor": "https://t.me/+Pu-pzlDZXCwwMWEx",
    "07_Aceptologia": "https://t.me/+Dcx3f73dGAtmOTQ5",
    "08_Alquimia_del_Pensamiento": "https://t.me/+qHsoBput0S5hN2Ex",
    "09_Asumiendo_la_Vida_con_Sabiduria": "https://t.me/+E8w1ZoqO0Fw0NWIx",
    "10_Trascendiendo_las_Limitaciones": "https://t.me/+wYTNtpuxo8I5NTlh",
    "11_Incondicionalidad_Forma_de_Amar": "https://t.me/+WgyzJg9Rm38yMGFh",
    "12_Creacion_Divina_Realidad_Humana": "https://t.me/+gjUKfw_no1MxZWZh",
    "13_Encontrandote_Reino_del_Amor": "https://t.me/+kwYnwehcwZ9kYmUx",
    "14_Felicidad_Familia_Pareja": "https://t.me/+tI2j9PGDpgpiNzJh",
    "15_Escuela_de_Padres_HOUDHO": "https://t.me/+MJZOmFNVwEI1N2Jh",
    "16_Tu_Eres_lo_Mejor_de_Ti_Mismo": "https://t.me/+Xo2ryyyQJedlNjE5",
    "17_Liderando_Compromiso_Prosperidad": "https://t.me/+uhHnhrxZOaFjN2Fh",
    # Canales complementarios con PDFs
    "Manuales_Originales_Resumenes_Transcripciones": "https://t.me/+E-OejYlc2CUwNGMx",
    "Gerardo_en_PDF": "https://t.me/+3_UqO4I2Q_pjMjMx",
}


def sanitize_filename(name: str) -> str:
    """Limpia un nombre de archivo para que sea válido en el sistema de archivos."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = name.strip('. ')
    return name[:200]  # Limitar longitud


async def download_pdfs_from_channel(client, channel_name: str, invite_link: str):
    """Descarga todos los PDFs de un canal dado."""
    channel_dir = OUTPUT_DIR / channel_name
    channel_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Intentar unirse al canal si aún no estamos
        entity = await client.get_entity(invite_link)
        print(f"\n{'='*60}")
        print(f"📂 Canal: {channel_name}")
        print(f"   Entidad: {entity.title if hasattr(entity, 'title') else entity}")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ No se pudo acceder al canal '{channel_name}': {e}")
        print(f"   Link: {invite_link}")
        print(f"   (Asegúrate de estar unido a este canal)")
        return 0

    pdf_count = 0

    async for message in client.iter_messages(entity):
        if message.document:
            # Verificar si es un PDF
            filename = None
            for attr in message.document.attributes:
                if isinstance(attr, DocumentAttributeFilename):
                    filename = attr.file_name
                    break

            if filename and filename.lower().endswith('.pdf'):
                safe_name = sanitize_filename(filename)
                filepath = channel_dir / safe_name

                if filepath.exists():
                    print(f"   ⏭️  Ya existe: {safe_name}")
                    pdf_count += 1
                    continue

                print(f"   ⬇️  Descargando: {safe_name}")
                try:
                    await client.download_media(message, file=str(filepath))
                    pdf_count += 1
                except Exception as e:
                    print(f"   ❌ Error descargando {safe_name}: {e}")

    print(f"   ✅ Total PDFs en este canal: {pdf_count}")
    return pdf_count


async def main():
    """Función principal que orquesta la descarga de todos los canales."""
    if not API_ID or not API_HASH:
        print("❌ ERROR: Configura TELEGRAM_API_ID y TELEGRAM_API_HASH")
        print("   Opciones:")
        print("   1. Edita este archivo y reemplaza los valores directamente")
        print("   2. Configura variables de entorno:")
        print("      export TELEGRAM_API_ID=tu_api_id")
        print("      export TELEGRAM_API_HASH=tu_api_hash")
        print("      export TELEGRAM_PHONE=+57XXXXXXXXXX")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("🚀 Iniciando descarga de PDFs de Talleres de Comprensión")
    print(f"📁 Directorio de salida: {OUTPUT_DIR.absolute()}")
    print(f"📋 Canales a procesar: {len(CHANNELS)}")
    print()

    # Crear cliente de Telegram
    client = TelegramClient('schmedling_session', API_ID, API_HASH)

    await client.start(phone=PHONE)
    print("✅ Conectado a Telegram exitosamente")

    total_pdfs = 0

    for channel_name, invite_link in CHANNELS.items():
        count = await download_pdfs_from_channel(client, channel_name, invite_link)
        total_pdfs += count

    print(f"\n{'='*60}")
    print(f"🎉 DESCARGA COMPLETA")
    print(f"   Total PDFs descargados: {total_pdfs}")
    print(f"   Ubicación: {OUTPUT_DIR.absolute()}")
    print(f"{'='*60}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
