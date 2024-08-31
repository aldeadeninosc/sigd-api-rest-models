from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf
import joblib

# Cargar el tokenizador
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Cargar modelos desde rutas absolutas
model_documento = TFDistilBertForSequenceClassification.from_pretrained('/var/www/html/flask_project/recursos/modelo_distilbert_documentos')
model_subcarpeta = TFDistilBertForSequenceClassification.from_pretrained('/var/www/html/flask_project/recursos/modelo_distilbert_subcarpeta')
model_carpeta = TFDistilBertForSequenceClassification.from_pretrained('/var/www/html/flask_project/recursos/modelo_distilbert_carpeta')

# Cargar LabelEncoders desde rutas absolutas
label_encoder_documento = joblib.load('/var/www/html/flask_project/recursos/LabelEncoder/label_encoder_documento.pkl')
label_encoder_subcarpeta = joblib.load('/var/www/html/flask_project/recursos/LabelEncoder/label_encoder_subcarpeta.pkl')
label_encoder_carpeta = joblib.load('/var/www/html/flask_project/recursos/LabelEncoder/label_encoder_carpeta.pkl')

def predict_labels(text):
    # Tokenizar el texto de entrada
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)

    # Predicción para documento
    logits_documento = model_documento(inputs).logits
    predicted_class_id_documento = tf.argmax(logits_documento, axis=-1).numpy()[0]
    predicted_label_documento = label_encoder_documento.inverse_transform([predicted_class_id_documento])[0]

    # Predicción para subcarpeta
    logits_subcarpeta = model_subcarpeta(inputs).logits
    predicted_class_id_subcarpeta = tf.argmax(logits_subcarpeta, axis=-1).numpy()[0]
    predicted_label_subcarpeta = label_encoder_subcarpeta.inverse_transform([predicted_class_id_subcarpeta])[0]

    # Predicción para carpeta
    logits_carpeta = model_carpeta(inputs).logits
    predicted_class_id_carpeta = tf.argmax(logits_carpeta, axis=-1).numpy()[0]
    predicted_label_carpeta = label_encoder_carpeta.inverse_transform([predicted_class_id_carpeta])[0]

    return predicted_label_carpeta, predicted_label_subcarpeta, predicted_label_documento
