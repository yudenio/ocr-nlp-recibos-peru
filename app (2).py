import streamlit as st
import google.generativeai as genai
from PIL import Image
import json

# Configuración de página
st.set_page_config(page_title="IA Multimodal - Recibos Perú", layout="wide")
st.title("📄 Análisis Inteligente de Recibos con Gemini Pro Vision")
st.markdown("Extracción de entidades mediante Modelos Multimodales en la nube.")

# --- BARRA LATERAL PARA LA API KEY ---
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("Pega tu API Key de Google Gemini aquí:", type="password")
st.sidebar.markdown("[Obtén tu API Key gratis aquí](https://aistudio.google.com/app/apikey)")

# --- INTERFAZ WEB ---
uploaded_file = st.file_uploader("Sube la imagen del recibo (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', width=400)
    
    if not api_key:
        st.warning("⚠️ Por favor, ingresa tu API Key en la barra lateral para continuar.")
    else:
        with st.spinner('Analizando recibo con Gemini Pro... (Esto tomará unos 5 segundos)'):
            try:
                # Configurar la API de Gemini
                genai.configure(api_key=api_key)
                
                # Usar el modelo optimizado para visión
                model = genai.GenerativeModel('gemini-pro-vision')
                
                # El Prompt maestro que combina OCR y NLP estructurado
                prompt = """
                Actúa como un experto en extracción de datos de documentos peruanos.
                Analiza esta imagen de un recibo de servicios (como Luz del Sur) y extrae exactamente la siguiente información.
                
                Devuelve ÚNICAMENTE un objeto JSON válido con estas claves exactas (si no encuentras un dato, pon "-"):
                {
                  "suministro": "Número de Suministro o Cliente",
                  "mes_facturado": "El mes y año facturado (ej. Enero 2026)",
                  "total_pagar": "Monto total a pagar incluyendo la moneda (ej. S/ 86.30)",
                  "fecha_vencimiento": "Fecha de vencimiento",
                  "fecha_emision": "Fecha de emisión del recibo",
                  "consumo": "Consumo registrado en el mes (ej. 101.40 kWh)"
                }
                No incluyas formato markdown como ```json, solo el texto del JSON puro.
                """
                
                # Llamada a la IA
                response = model.generate_content([prompt, image])
                texto_respuesta = response.text.strip()
                
                # Limpieza por si Gemini añade formato markdown por costumbre
                if texto_respuesta.startswith("```json"):
                    texto_respuesta = texto_respuesta[7:-3]
                elif texto_respuesta.startswith("```"):
                    texto_respuesta = texto_respuesta[3:-3]
                    
                # Parsear el resultado
                entidades = json.loads(texto_respuesta)
                
                st.success("¡Análisis completado con éxito! Cero impacto en el servidor.")
                
                # Mostrar métricas visuales
                st.subheader("🔍 Entidades Extraídas (IA Multimodal)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total a Pagar", entidades.get("total_pagar", "-"))
                col2.metric("N° de Suministro", entidades.get("suministro", "-"))
                col3.metric("Consumo", entidades.get("consumo", "-"))
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Vencimiento", entidades.get("fecha_vencimiento", "-"))
                col5.metric("Emisión", entidades.get("fecha_emision", "-"))
                col6.metric("Mes Facturado", entidades.get("mes_facturado", "-"))
                
            except Exception as e:
                st.error(f"Ocurrió un error en la conexión o procesamiento: {e}")
