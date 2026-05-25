import streamlit as st
import cv2
import pytesseract
import numpy as np
import re
from PIL import Image

# Configuración de página
st.set_page_config(page_title="OCR Recibos Perú", layout="wide")
st.title("📄 Análisis Inteligente de Recibos de Servicios Peruanos")
st.markdown("Aplicación de OCR y NLP para extracción de entidades en contexto peruano.")

# Funciones de NLP
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-z0-9\s.,/:-]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def extraer_entidades(texto):
    entidades = {"Total a Pagar": "-", "Año de Facturación": "-", "Posible Fecha": "-"}
    
    # NLP Regex estricto para Monto: Ahora EXIGE la palabra "pagar" para ignorar el "subtotal"
    monto = re.search(r'pagar[^\d]*?(\d{2,4}[.,]\d{2}|\d{3,5})', texto)
    if monto:
        val = monto.group(1)
        if not '.' in val and not ',' in val and len(val) >= 3:
            val = val[:-2] + '.' + val[-2:]
        val = val.replace(',', '.') # Normalizar comas a puntos
        entidades["Total a Pagar"] = f"S/ {val}"
        
    # NLP Regex para Año
    ano = re.search(r'(202[4-6])', texto)
    if ano: 
        entidades["Año de Facturación"] = ano.group(1)
        
    # NLP Regex estricto para Fecha: Obliga a que contenga un mes válido o número del 01 al 12, y año 202X
    fecha = re.search(r'(\d{2}[/.-](?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|0[1-9]|1[0-2])[/.-]202[4-6])', texto)
    if fecha: 
        entidades["Posible Fecha"] = fecha.group(1).title()
        
    return entidades

uploaded_file = st.file_uploader("Sube la imagen del recibo (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', width=400)
    
    with st.spinner('Procesando imagen con OCR y NLP...'):
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Preprocesamiento equilibrado
        ancho_base = 1000
        proporcion = ancho_base / gray.shape[1]
        dim = (ancho_base, int(gray.shape[0] * proporcion))
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        
        # Binarización adaptativa con un bloque más grande (21) para manejar mejor las sombras de la foto
        thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
        
        # OCR (SOLUCIÓN CLAVE: psm 11 busca texto disperso en todo el recibo)
        custom_config = r'-l spa --oem 3 --psm 11'
        texto_crudo = pytesseract.image_to_string(thresh, config=custom_config)
        
        # NLP
        texto_limpio = limpiar_texto(texto_crudo)
        entidades = extraer_entidades(texto_limpio)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Texto Extraído (Limpio)")
            st.write(texto_limpio)
            
        with col2:
            st.subheader("🔍 Entidades Extraídas (NLP)")
            st.metric(label="Total a Pagar", value=entidades["Total a Pagar"])
            st.metric(label="Año de Facturación", value=entidades["Año de Facturación"])
            st.metric(label="Fecha Detectada", value=entidades["Posible Fecha"])
