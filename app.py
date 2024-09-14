from flask import Flask, jsonify, request
from models import identificar_documento_y_carpeta_mejorado

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1 style='color: blue;'>API REST MODELS</h1>"

@app.route('/predict', methods=['GET'])
def predict():
    # Obtener el parámetro de la consulta
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is missing"}), 400
    
    # Llamar a la función para predecir las etiquetas utilizando los modelos
    carpeta, subcarpeta, documento = identificar_documento_y_carpeta_mejorado(query)
    
    # Retornar el resultado en formato JSON
    return jsonify({"carpeta": carpeta, "subcarpeta": subcarpeta, "documento": documento})

if __name__ == '__main__':
    app.run(debug=True, port=4000)
