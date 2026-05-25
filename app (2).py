import streamlit as st
import google.generativeai as genai
from PIL import Image
import json

# Configuración de página
st.set_page_config(page_title="IA Multimodal - Recibos Perú", layout="wide")
st.title("📄 Análisis Inteligente de Recibos con IA")
st.markdown("Extracción de entidades usando Modelos Fundacionales Autodetectables.")

# --- BARRA LATERAL PARA LA API KEY ---
st.sidebar.header("🔑 Configuración")
api_key = st.sidebar.text_input("Pega tu API Key de Google aquí:", type="password")
st.sidebar.markdown("[Obtén tu API Key gratis aquí](https://aistudio.google.com/app/apikey)")

# --- INTERFAZ WEB ---
uploaded_file = st.file_uploader("Sube la imagen del recibo (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen Original', width=400)
    
    if not api_key:
        st.warning("⚠️ Por favor, ingresa tu API Key en la barra lateral para continuar.")
    else:
        with st.spinner('Conectando con Google y analizando recibo...'):
            try:
                genai.configure(api_key=api_key)
                
                # 1. RUTINA EXPERTA: Auto-descubrimiento de modelos disponibles
                # Le preguntamos a la API a qué modelos de generación tiene acceso tu llave
                modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                modelo_elegido = None
                
                # Buscamos la versión más moderna (1.5 flash) o la clásica de visión
                for nombre in modelos_disponibles:
                    if '1.5-flash' in nombre or 'pro-vision' in nombre:
                        modelo_elegido = nombre
                        break
                
                # Si no encuentra los preferidos, usa el primer modelo disponible como contingencia
                if not modelo_elegido and modelos_disponibles:
                    modelo_elegido = modelos_disponibles[0]

                st.info(f"💡 Modelo detectado y conectado automáticamente: {modelo_elegido}")
                
                # 2. Inicializar el modelo exacto que tu API Key soporta
                model = genai.GenerativeModel(modelo_elegido)
                
                prompt = """
                Analiza esta imagen de un recibo de servicios peruano (como Luz del Sur).
                Devuelve ÚNICAMENTE un objeto JSON con estas claves exactas (pon "-" si no encuentras el dato):
                {
                  "suministro": "Número de Suministro o Cliente",
                  "mes_facturado": "El mes y año facturado (ej. Enero 2026)",
                  "total_pagar": "Monto total a pagar (ej. S/ 86.30)",
                  "fecha_vencimiento": "Fecha de vencimiento",
                  "fecha_emision": "Fecha de emisión del recibo",
                  "consumo": "Consumo registrado en kWh"
                }
                """
                
                response = model.generate_content([prompt, image])
                texto_respuesta = response.text.strip()
                
                # Limpieza por si la IA añade formato markdown
                if texto_respuesta.startswith("```json"):
                    texto_respuesta = texto_respuesta[7:-3]
                elif texto_respuesta.startswith("```"):
                    texto_respuesta = texto_respuesta[3:-3]
                    
                entidades = json.loads(texto_respuesta)
                
                st.success("¡Análisis NLP completado con éxito!")
                
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
                st.error(f"Error de procesamiento: {e}")
                if 'modelos_disponibles' in locals():
                    st.write("Modelos a los que tu llave tiene acceso actualmente:", modelos_disponibles)
