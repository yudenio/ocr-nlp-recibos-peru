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
    
    # NLP Regex para Monto
    monto = re.search(r'(pagar|total).*?(\d{2,4}[.,]\d{2}|\d{3,5})', texto)
    if monto:
        val = monto.group(2)
        if not '.' in val and not ',' in val and len(val) >= 3:
            val = val[:-2] + '.' + val[-2:]
        entidades["Total a Pagar"] = f"S/ {val}"
        
    # NLP Regex para Año (Actualizado a 2026)
    ano = re.search(r'(202[4-6])', texto)
    if ano: 
        entidades["Año de Facturación"] = ano.group(1)
        
    # NLP Regex para Fecha
    fecha = re.search(r'(\d{2}[/.-](?:\d{2}|[a-z]{3})[/.-]\d{2,4})', texto)
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
        
        # SOLUCIÓN: Redimensionar y aplicar umbral de Otsu para estabilizar la lectura
        ancho_base = 1200
        proporcion = ancho_base / gray.shape[1]
        dim = (ancho_base, int(gray.shape[0] * proporcion))
        resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        
        blur = cv2.GaussianBlur(resized, (5,5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR
        custom_config = r'-l spa --oem 3 --psm 6'
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
