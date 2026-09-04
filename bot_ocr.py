import os
import base64
import asyncio
import logging
from datetime import datetime
from PIL import Image, ImageEnhance
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- NUEVA LIBRERÍA GOOGLE.GENAI ---
from google import genai
from google.genai import types

# --- CONFIGURACIÓN (LEYENDO DESDE VARIABLES DE ENTORNO) ---
# Railway leerá estos datos desde su panel de "Variables"
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CANAL_ORIGEN = os.getenv("CANAL_ORIGEN")
CANAL_DESTINO = os.getenv("CANAL_DESTINO")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validación rápida para asegurarnos de que las variables existen en Railway
if not all([TOKEN, CANAL_ORIGEN, CANAL_DESTINO, GOOGLE_API_KEY]):
    raise ValueError("❌ ERROR CRÍTICO: Faltan variables de entorno. Asegúrate de configurar TELEGRAM_BOT_TOKEN, CANAL_ORIGEN, CANAL_DESTINO y GOOGLE_API_KEY en Railway.")

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DEL CLIENTE GEMINI ---
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- FUNCIONES DE MEJORA DE IMAGEN ---
def mejorar_imagen_para_ocr(ruta_imagen):
    """
    Mejora la calidad de la imagen para OCR
    """
    try:
        with Image.open(ruta_imagen) as img:
            # Convertir a RGB si es necesario
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Redimensionar si es muy pequeña
            width, height = img.size
            if width < 1000 or height < 1000:
                scale = max(1000/width, 1000/height)
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Imagen redimensionada de {width}x{height} a {new_size[0]}x{new_size[1]}")
            
            # Mejorar contraste y nitidez
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # Guardar versión mejorada
            ruta_mejorada = ruta_imagen.replace('.jpg', '_mejorada.jpg')
            img.save(ruta_mejorada, 'JPEG', quality=95)
            return ruta_mejorada
            
    except Exception as e:
        logger.error(f"Error al mejorar imagen: {e}")
        return ruta_imagen

# --- FUNCIÓN PARA LIMPIAR TEXTO (CORREGIDA PARA QUE SE VEA LIMPIO) ---
def limpiar_texto_para_telegram(texto):
    """
    Limpia caracteres especiales que pueden causar errores en Telegram
    (Se eliminaron los escapes de números y puntos para que se vea limpio)
    """
    # Solo escapamos los que Telegram interpreta como formato Markdown estricto
    caracteres_especiales = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}']
    for char in caracteres_especiales:
        texto = texto.replace(char, f'\\{char}')
    return texto

# --- FUNCIÓN PARA LIMPIAR EL TEXTO DE GEMINI (NUEVA) ---
def limpiar_texto_gemini(texto):
    """
    Elimina los paréntesis y textos redundantes que agrega Gemini
    Ejemplo: '0.20 (VTA. TARIFA 15.00: 1.30)' -> '0.20'
    """
    import re
    # Eliminar todo lo que esté entre paréntesis
    texto_sin_parentesis = re.sub(r'\(.*?\)', '', texto)
    # Eliminar espacios dobles o sobrantes
    texto_limpio = ' '.join(texto_sin_parentesis.split())
    return texto_limpio

# --- FUNCIÓN DE OCR CON GEMINI ---
async def extraer_texto_imagen(file_path, mejorar=True):
    """
    Extrae texto de la imagen usando el modelo más reciente de Gemini
    """
    try:
        # Mejorar imagen antes de procesar
        if mejorar:
            file_path = mejorar_imagen_para_ocr(file_path)
        
        # Leer imagen
        with open(file_path, "rb") as image_file:
            image_bytes = image_file.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # PROMPT
        prompt = """Eres un lector OCR profesional especializado en facturas.
        
        INSTRUCCIONES ESTRICTAS:
        1. SIEMPRE comienza extrayendo la información del EMISOR (la empresa/vendedor) que se encuentra en la parte superior de la factura.
        2. Organiza la información EXACTAMENTE en este orden y con estos títulos:
        
        EMISOR:
        (Nombre de la empresa)
        (RUC del emisor)
        
        NUMERO DE FACTURA:
        FECHA:
        
        CLIENTE (nombre y documento):
        
        PRODUCTOS/SERVICIOS (cantidad, descripcion, precio):
        
        SUBTOTAL:
        IMPUESTOS:
        TOTAL:
        FORMA DE PAGO:
        
        3. Presta MUCHA atención a los encabezados. NO OMITAS el nombre de la empresa ni el número de RUC.
        4. IMPORTANTE: Para los valores de SUBTOTAL, IMPUESTOS y TOTAL, devuelve SOLO el número final. NO agregues paréntesis con detalles adicionales.
        5. Mantén el formato original de números y fechas.
        
        DEVUELVE SOLO EL TEXTO ESTRUCTURADO EN FORMATO PLANO.
        NO agregues explicaciones adicionales.
        """
        
        # Configurar la generación
        generate_config = types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
        )
        
        # Crear el contenido con la imagen
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=prompt),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=image_bytes
                        )
                    )
                ]
            )
        ]
        
        # Lista de modelos actualizados para probar en orden
        lista_modelos = [
            "gemini-3.6-flash", 
            "gemini-3.5-flash", 
            "gemini-2.5-flash"
        ]
        
        ultimo_error = None
        for modelo in lista_modelos:
            try:
                # Realizar la solicitud
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model=modelo,
                            contents=contents,
                            config=generate_config
                        )
                    ),
                    timeout=60.0
                )
                
                # Verificar si la respuesta tiene texto
                if response and response.text:
                    texto_limpio = response.text.strip()
                    logger.info(f"✅ OCR exitoso usando {modelo}: {len(texto_limpio)} caracteres extraídos")
                    return texto_limpio
                else:
                    logger.warning(f"⚠️ {modelo} no devolvió texto")
                    return "No se pudo extraer texto de la imagen"
                    
            except Exception as e:
                ultimo_error = e
                if "503" in str(e):
                    logger.warning(f"⚠️ Modelo {modelo} saturado. Probando siguiente...")
                    await asyncio.sleep(2) # Esperar 2 segundos
                elif "404" in str(e):
                    logger.warning(f"⚠️ Modelo {modelo} no disponible. Probando siguiente...")
                else:
                    # Si es otro error, lo lanzamos directo
                    raise e
        
        # Si se acabaron los modelos y todos fallaron
        raise ultimo_error
            
    except asyncio.TimeoutError:
        logger.error("❌ Timeout en OCR - La imagen puede ser muy compleja o grande")
        return "Error: La imagen es muy grande o compleja. Por favor, prueba con una imagen más clara."
    except Exception as e:
        logger.error(f"❌ Error en OCR: {str(e)}")
        return f"Error al procesar la imagen: {str(e)}"

# --- FUNCIÓN PARA FORMATEAR EL TEXTO (MODIFICADA) ---
def formatear_respuesta(texto):
    """
    Da formato al texto extraído para mejor legibilidad
    """
    if not texto or len(texto) < 10:
        return texto
    
    lineas = texto.split('\n')
    lineas_formateadas = []
    
    for linea in lineas:
        linea = linea.strip()
        if not linea:
            continue
        
        # Limpiamos los paréntesis sobrantes de las líneas que tienen números importantes
        linea = limpiar_texto_gemini(linea)
        
        linea_lower = linea.lower()
        
        if 'emisor' in linea_lower or 'empresa' in linea_lower or 'ruc' in linea_lower:
            lineas_formateadas.append(f"\n🏢 {linea}")
        elif 'factura' in linea_lower or 'numero' in linea_lower:
            lineas_formateadas.append(f"\n📄 {linea}")
        elif 'cliente' in linea_lower:
            lineas_formateadas.append(f"\n👤 {linea}")
        elif 'producto' in linea_lower or 'servicio' in linea_lower:
            lineas_formateadas.append(f"\n📦 {linea}")
        elif 'subtotal' in linea_lower:
            lineas_formateadas.append(f"\n💰 {linea}")
        elif 'impuesto' in linea_lower or 'iva' in linea_lower:
            lineas_formateadas.append(f"\n🧾 {linea}")
        elif 'total' in linea_lower:
            lineas_formateadas.append(f"\n💳 {linea}")
        elif 'pago' in linea_lower:
            lineas_formateadas.append(f"\n💵 {linea}")
        else:
            lineas_formateadas.append(f"  {linea}")
    
    return '\n'.join(lineas_formateadas)

# --- MANEJADOR DE MENSAJES ---
async def manejar_mensaje(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Maneja los mensajes del canal de origen
    """
    try:
        # Verificar si es un mensaje del canal de origen
        if not (update.channel_post and update.channel_post.chat.username == CANAL_ORIGEN):
            return
        
        # Verificar si tiene foto
        if not update.channel_post.photo:
            logger.info("Mensaje sin foto, ignorado")
            return
        
        # Verificar si hay caption
        caption = update.channel_post.caption or ""
        
        # Descargar la foto (la de mejor calidad)
        foto = update.channel_post.photo[-1]
        archivo = await foto.get_file()
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_local = f"imagen_temp_{timestamp}.jpg"
        
        logger.info(f"📥 Descargando: {ruta_local}")
        await archivo.download_to_drive(ruta_local)
        
        # Enviar mensaje de proceso
        await context.bot.send_message(
            chat_id=f"@{CANAL_DESTINO}",
            text="🔄 Procesando imagen con Gemini Flash... (esto puede tomar hasta 60 segundos)"
        )
        
        # Extraer texto
        texto_extraido = await extraer_texto_imagen(ruta_local, mejorar=True)
        
        # Si hay caption, agregarlo
        if caption:
            texto_extraido = f"📝 Nota: {caption}\n\n{texto_extraido}"
        
        # Formatear para mejor legibilidad
        texto_formateado = formatear_respuesta(texto_extraido)
        
        # Limpiar el texto para evitar errores en Telegram
        texto_limpio = limpiar_texto_para_telegram(texto_formateado)
        
        # Crear mensaje final
        mensaje_final = f"""📄 EXTRACCION DE FACTURA
━━━━━━━━━━━━━━━━━━━━━━━━━━━

{texto_limpio}

━━━━━━━━━━━━━━━━━━━━━━━━━━━
🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
🧠 Gemini Flash"""
        
        # Enviar el texto al canal destino
        await context.bot.send_message(
            chat_id=f"@{CANAL_DESTINO}",
            text=mensaje_final
        )
        
        logger.info(f"✅ Texto enviado a {CANAL_DESTINO}")
        
        # Limpiar archivos temporales
        try:
            os.remove(ruta_local)
            ruta_mejorada = ruta_local.replace('.jpg', '_mejorada.jpg')
            if os.path.exists(ruta_mejorada):
                os.remove(ruta_mejorada)
        except:
            pass
            
    except Exception as e:
        logger.error(f"❌ Error en manejar_mensaje: {str(e)}")
        try:
            await context.bot.send_message(
                chat_id=f"@{CANAL_DESTINO}",
                text=f"❌ Error: {str(e)}"
            )
        except:
            pass

# --- MANEJADOR DE ERRORES ---
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")
    if update and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="⚠️ Ocurrió un error procesando tu mensaje."
            )
        except:
            pass

# --- INICIALIZACIÓN ---
def main():
    print("=" * 60)
    print("🤖 BOT DE OCR CON GEMINI FLASH")
    print(f"📡 Origen: @{CANAL_ORIGEN}")
    print(f"📡 Destino: @{CANAL_DESTINO}")
    print(f"🔑 Tipo: Auth Key (Variables de Entorno)")
    print("=" * 60)
    print("⚠️  IMPORTANTE: Asegúrate de que NO haya otra instancia del bot ejecutándose")
    print("=" * 60)
    
    try:
        app = Application.builder().token(TOKEN).build()
        app.add_handler(MessageHandler(filters.ALL, manejar_mensaje))
        app.add_error_handler(error_handler)
        
        print("✅ Bot iniciado correctamente")
        print("🔄 Esperando mensajes...")
        
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Error fatal: {e}")

if __name__ == '__main__':
    main()