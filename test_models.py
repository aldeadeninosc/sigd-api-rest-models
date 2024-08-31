from models import predict_labels  
# Ejemplo de uso
text = "Acta de nacimiento de Victor Parrales"
carpeta, subcarpeta, documento = predict_labels(text)
print(f"Texto: {text}")
print(f"Carpeta: {carpeta}")
print(f"Subcarpeta: {subcarpeta}")
print(f"Documento: {documento}")