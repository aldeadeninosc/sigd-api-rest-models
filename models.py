from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import joblib
import mysql.connector
import re
import pandas as pd

# Configuración de la conexión a la base de datos
db_config = {
    'user': 'u840740651_aldea',
    'password': 'Evelyn102030',
    'host': 'aldeacristorey.com',
    'database': 'u840740651_aldea'
}

# Cargar el tokenizador y los modelos desde rutas absolutas
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model_subcarpeta = TFDistilBertForSequenceClassification.from_pretrained('/var/www/html/flask_project/recursos/modelo_distilbert_subcarpeta')

# Cargar LabelEncoders desde rutas absolutas
label_encoder_subcarpeta = joblib.load('/var/www/html/flask_project/recursos/LabelEncoder/label_encoder_subcarpeta.pkl')

# Leer archivos CSV y crear conjuntos de nombres y apellidos
nombres_df = pd.read_csv('/var/www/html/flask_project/recursos/nombres.csv')
apellidos_df = pd.read_csv('/var/www/html/flask_project/recursos/apellidos.csv')

nombres = set(nombres_df['nombre'].str.upper().tolist())
apellidos = set(apellidos_df['apellido'].str.upper().tolist())

# Normalización del texto
def normalizar_texto(texto):
    texto_normalizado = texto.upper()
    texto_normalizado = re.sub(r'[^\w\s]', '', texto_normalizado)
    return texto_normalizado

# Buscar la carpeta en la base de datos
def buscar_carpeta_en_bd(carpeta_construct):
    carpeta_construct_normalizada = normalizar_texto(carpeta_construct)
    mejor_coincidencia = None
    max_coincidencias = 0

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = "SELECT folder_name FROM folder_models"
        cursor.execute(query)

        for (folder_name,) in cursor:
            folder_name_normalizado = normalizar_texto(folder_name)
            coincidencias = sum(1 for palabra in carpeta_construct_normalizada.split() if palabra in folder_name_normalizado)
            if coincidencias > max_coincidencias:
                max_coincidencias = coincidencias
                mejor_coincidencia = folder_name
                #print(f"Mejor coincidencia de carpeta: {mejor_coincidencia}")

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")

    if mejor_coincidencia:
        return mejor_coincidencia
    return "No se encontró la carpeta en la base de datos"

# Buscar el documento en la base de datos
def buscar_documento_en_bd(documento_construct):
    documento_construct_normalizado = normalizar_texto(documento_construct)
    mejor_coincidencia = None
    max_coincidencias = 0

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = "SELECT document_name FROM documents_models"
        cursor.execute(query)

        for (document_name,) in cursor:
            document_name_normalizado = normalizar_texto(document_name)
            coincidencias = sum(1 for palabra in documento_construct_normalizado.split() if palabra in document_name_normalizado)
            if coincidencias > max_coincidencias:
                max_coincidencias = coincidencias
                mejor_coincidencia = document_name
                #print(f"Mejor coincidencia de documento: {mejor_coincidencia}")

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"Error al conectar a la base de datos: {err}")

    if mejor_coincidencia:
        return mejor_coincidencia
    return "No se encontró el documento en la base de datos"

# Predicción de la subcarpeta
def predict_subcarpeta(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    logits_subcarpeta = model_subcarpeta(inputs).logits
    predicted_class_id_subcarpeta = tf.argmax(logits_subcarpeta, axis=-1).numpy()[0]
    predicted_label_subcarpeta = label_encoder_subcarpeta.inverse_transform([predicted_class_id_subcarpeta])[0]
    return predicted_label_subcarpeta

# Identificar documento y carpeta mejorado
def identificar_documento_y_carpeta_mejorado(consulta):
    print(f"Consulta: {consulta}")
    palabras = consulta.upper().split()
    nombres_encontrados = []
    apellidos_encontrados = []
    documento_encontrado = []

    # Verificar nombres y apellidos en la consulta
    for palabra in palabras:
        if palabra in nombres:
            nombres_encontrados.append(palabra)
        elif palabra in apellidos:
            apellidos_encontrados.append(palabra)
        else:
            documento_encontrado.append(palabra)

    # Construir la carpeta
    carpeta_construct = ' '.join(nombres_encontrados + apellidos_encontrados).title()
    carpeta = buscar_carpeta_en_bd(carpeta_construct)

    # Construir el documento
    documento_construct = ' '.join(documento_encontrado).title()
    documento = buscar_documento_en_bd(documento_construct)

    # Predecir subcarpeta
    subcarpeta = predict_subcarpeta(consulta)
    
    #print(f"Nombres encontrados: {nombres_encontrados}")  # Verificar nombres encontrados
    #print(f"Apellidos encontrados: {apellidos_encontrados}")  # Verificar apellidos encontrados
    #print(f"Documento encontrado: {documento_encontrado}")  # Verificar documentos encontrados
    
    #print(f"carpeta: {carpeta}")
    #print(f"carpeta: {subcarpeta}")
    #print(f"carpeta: {documento}")

    return carpeta, subcarpeta, documento
