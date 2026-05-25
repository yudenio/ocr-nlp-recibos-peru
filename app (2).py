import streamlit as st
import easyocr
import cv2
import numpy as np
import re
from PIL import Image

# Configuración de página
st.set_page_config(page_title="OCR Avanzado - Recibos Perú", layout="wide")
st.title("📄 Análisis IA de Recibos Peruanos (EasyOCR + NLP)")
st.markdown("Aplicación con Deep Learning para extracción masiva de entidades.")

# --- INICIALIZACIÓN DEL MODELO IA ---
# Usamos cache para que el modelo pesado se cargue solo 1 vez y la app no sea lenta
@st.cache_resource
def cargar_lector_ocr():
    # Carga el modelo en español ('es') usando solo CPU (gpu=False)
    return easyocr.Reader(['es'], gpu=False)

reader = cargar_lector_ocr()

# --- FUNCIONES DE NLP ---
def extraer_entidades(texto_crudo):
    # Convertir a minúsculas y quitar saltos de línea para facilitar la búsqueda
    texto = " ".join(texto_crudo).lower()
    
    entidades = {
        "N° de Suministro": "No detectado",
        "Mes Facturado": "No detectado",
        "Total a Pagar": "No detectado",
        "Fecha de Vencimiento": "No detectada",
        "Fecha de Emisión": "No detectada",
        "Consumo Registrado": "No detectado"
    }
    
    # 1. N° de Suministro (Busca la palabra suministro seguida de 6 a 8 dígitos)
    match_suministro = re.search(r'suministro\D*(\d{6,8})', texto)
    if match_suministro: entidades["N° de Suministro"] = match_suministro.group(1)
        
    # 2. Mes Facturado (Busca "facturado" seguido de un mes y año)
    match_mes = re.search(r'facturado\s+([a-z]+\s+202[4-6])', texto)
    if match_mes: entidades["Mes Facturado"] = match_mes.group(1).title()

    # 3. Total a Pagar (Busca pagar o total y captura el monto con decimales)
    match_total = re.search(r'(?:pagar|total)[^\d]*?(\d{1,4}[.,]\d{2})', texto)
    if match_total: entidades["Total a Pagar"] = f"S/ {match_total.group(1).replace(',', '.')}"

    # 4 y 5. Fechas (Patrón flexible para '11-mar-2025' o '11/03/2025')
    patron_fecha = r'(\d{2}[/.-](?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|\d{2})[/.-]202[4-6])'
    
    match_vencimiento = re.search(r'vencimiento\D*' + patron_fecha, texto)
    if match_vencimiento: entidades["Fecha de Vencimiento"] = match_vencimiento.group(1).title()
        
    match_emision = re.search(r'emisi[oó]n\D*' + patron_fecha, texto)
    if match_emision: entidades["Fecha de Emisión"] = match_emision.group(1).title()

    # 6. Consumo Total en kWh (Captura los montos altos cerca de la palabra consumo o kwh)
    match_consumo = re.search(r'(?:consumo|diferencia).*?(\d{2,4}[.,]\d{2})\s*(?:kwh|kw|x)', texto)
    if match_consumo: entidades["Consumo Registrado"] = match_consumo.group(1).replace(',', '.') + " kWh"

    return texto, entidades

# --- INTERFAZ WEB ---
uploaded_file = st.file_uploader("Sube la imagen del recibo (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', width=400)
    
    with st.spinner('Procesando imagen con IA (EasyOCR)... Esto puede tomar unos 10-15 segundos.'):
        # Convertir imagen para OpenCV
        img_array = np.array(image)
        # Ya no binarizamos en extremo, EasyOCR es inteligente. Solo lo pasamos a escala de grises.
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Extraer texto usando EasyOCR (devuelve una lista de frases)
        # detail=0 nos da solo el texto puro, sin coordenadas, paragraph=True junta textos cercanos
        resultados_ocr = reader.readtext(gray, detail=0, paragraph=True)
        
        # Aplicar NLP
        texto_limpio, entidades = extraer_entidades(resultados_ocr)
        
        # Mostrar resultados visuales
        st.success("¡Lectura y análisis completados!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🔍 Entidades Extraídas (NLP)")
            # Mostramos la metadata interesante usando un diseño más limpio
            for clave, valor in entidades.items():
                st.write(f"**{clave}:** {valor}")
                
        with col2:
            st.subheader("📝 Texto en Bruto (EasyOCR)")
            with st.expander("Ver lectura completa de la IA"):
                st.write(texto_limpio)
