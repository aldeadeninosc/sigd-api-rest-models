import sys
import site

# Añadir la ruta al entorno virtual y la ruta del proyecto al sys.path
site.addsitedir('/var/www/html/flask_project/venv/lib/python3.12/site-packages')

sys.path.insert(0, '/var/www/html/flask_project')

# Establecer la variable de entorno 'PYTHON_EGG_CACHE' para que apunte a un directorio donde Apache tenga permisos de escritura
import os
os.environ['PYTHON_EGG_CACHE'] = '/var/www/html/flask_project/.python-eggs'
os.environ['TRANSFORMERS_CACHE'] = '/var/www/html/flask_project/.cache'

# Importar la aplicación Flask
from app import app as application
