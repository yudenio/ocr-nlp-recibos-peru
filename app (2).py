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
    # Permitimos letras, números y guiones/puntos para las fechas y montos
    texto = re.sub(r'[^a-z0-9\s.,/:-]', '', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def extraer_entidades(texto):
    entidades = {"Total a Pagar": "-", "Año de Facturación": "-", "Posible Fecha": "-"}
    
    # NLP Regex para Monto: Busca "pagar" o "total", ignora basura intermedia y busca el número
    monto = re.search(r'(pagar|total).*?(\d{2,4}[.,]\d{2}|\d{3,5})', texto)
    if monto:
        val = monto.group(2)
        # Si el OCR omitió el punto decimal (ej. 8630 en vez de 86.30), lo reconstruimos
        if not '.' in val and not ',' in val and len(val) >= 3:
            val = val[:-2] + '.' + val[-2:]
        entidades["Total a Pagar"] = f"S/ {val}"
        
    # NLP Regex para Año
    ano = re.search(r'(202[4-5])', texto)
    if ano: 
        entidades["Año de Facturación"] = ano.group(1)
        
    # NLP Regex para Fecha: Ahora soporta formatos como "11-mar-2025" o "11/03/2025"
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
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        custom_config = r'-l spa --oem 3 --psm 6'
        texto_crudo = pytesseract.image_to_string(thresh, config=custom_config)
        
        texto_limpio = limpiar_texto(texto_crudo)
        entidades = extraer_entidades(texto_limpio)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Texto Extraído (Limpio)")
            # SOLUCIÓN RED: Usamos st.write en lugar de st.text_area para evitar fallos de Localtunnel
            st.write(texto_limpio)
            
        with col2:
            st.subheader("🔍 Entidades Extraídas (NLP)")
            st.metric(label="Total a Pagar", value=entidades["Total a Pagar"])
            st.metric(label="Año de Facturación", value=entidades["Año de Facturación"])
            st.metric(label="Fecha Detectada", value=entidades["Posible Fecha"])
