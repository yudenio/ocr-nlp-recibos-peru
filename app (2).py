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
                
                # Auto-descubrimiento de modelos
                modelos_disponibles = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                modelo_elegido = next((m for m in modelos_disponibles if '1.5-flash' in m or 'pro-vision' in m), modelos_disponibles[0] if modelos_disponibles else None)
                
                st.info(f"💡 Modelo detectado y conectado automáticamente: {modelo_elegido}")
                
                model = genai.GenerativeModel(modelo_elegido)
                
                # Prompt mejorado con los nuevos requerimientos de extracción de datos
                prompt = """
                Analiza esta imagen de un documento peruano.
                Devuelve ÚNICAMENTE un objeto JSON con estas claves exactas (pon "-" si no encuentras el dato):
                {
                  "tipo_documento": "Clasifica el documento (ej. Recibo de Luz, Factura de Agua, DNI, Ticket)",
                  "alerta_pago": "Análisis de estado: indica si está 'Vencido' o 'Vigente'",
                  "suministro": "Número de Suministro o Cliente",
                  "titular": "Nombre completo del cliente o titular del número de suministro",
                  "mes_facturado": "El mes y año facturado (ej. Enero 2026)",
                  "total_mes": "Monto correspondiente al 'TOTAL DEL MES' incluyendo la moneda (ej. S/ 87.33)",
                  "total_pagar": "Monto correspondiente al 'TOTAL A PAGAR S/' incluyendo la moneda (ej. S/ 87.40)",
                  "fecha_vencimiento": "Fecha de vencimiento",
                  "fecha_emision": "Fecha de emisión del recibo",
                  "consumo": "Consumo registrado en kWh (ej. 118.00 kWh)"
                }
                """
                
                response = model.generate_content([prompt, image])
                texto_respuesta = response.text.strip()
                
                # Limpieza de la respuesta
                if texto_respuesta.startswith("```json"):
                    texto_respuesta = texto_respuesta[7:-3]
                elif texto_respuesta.startswith("```"):
                    texto_respuesta = texto_respuesta[3:-3]
                    
                entidades = json.loads(texto_respuesta)
                
                st.success("¡Análisis NLP completado con éxito!")
                
                # Mostrar métricas visuales
                st.subheader("🔍 Entidades Extraídas (IA Multimodal)")
                
                # Datos de cabecera en texto destacado
                st.write(f"**Tipo de Documento:** {entidades.get('tipo_documento', '-')}")
                st.write(f"**Titular del Suministro:** {entidades.get('titular', '-')}")
                st.write(f"**Estado del Recibo:** {entidades.get('alerta_pago', '-')}")
                st.markdown("---")
                
                # Bloque 1 de métricas (Montos e Identificación)
                col1, col2, col3 = st.columns(3)
                col1.metric("Total a Pagar", entidades.get("total_pagar", "-"))
                col2.metric("Total del Mes", entidades.get("total_mes", "-"))
                col3.metric("N° de Suministro", entidades.get("suministro", "-"))
                
                # Bloque 2 de métricas (Fechas y Consumo)
                col4, col5, col6 = st.columns(3)
                col4.metric("Consumo", entidades.get("consumo", "-"))
                col5.metric("Fecha de Vencimiento", entidades.get("fecha_vencimiento", "-"))
                col6.metric("Fecha de Emisión", entidades.get("fecha_emision", "-"))
                
                st.write(f"**Periodo / Mes Facturado:** {entidades.get('mes_facturado', '-')}")
                
                st.markdown("---")
                # Botón de Descarga actualizado con la nueva estructura
                json_string = json.dumps(entidades, indent=4, ensure_ascii=False)
                st.download_button(
                    label="📥 Descargar Resultados (JSON)",
                    file_name="datos_extraidos.json",
                    mime="application/json",
                    data=json_string,
                )
                
            except Exception as e:
                st.error(f"Error de procesamiento: {e}")
