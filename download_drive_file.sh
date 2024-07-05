#!/bin/bash

sudo apt install pip
pip install -r requirements.txt
echo "Paquetes instalados correctamente desde requirements.txt."

sudo apt-get update && sudo apt-get install libgl1

# Descargar modelo
gdown --id 1yCCgM5_TKI5k-HdTqNYKe7UO8Jbmwjmp

# Nombre del archivo a mover
FILENAME="best.pt"

# Directorio de destino
DEST_DIR="modelos"

# Crear el directorio de destino si no existe
if [ ! -d "$DEST_DIR" ]; then
    mkdir "$DEST_DIR"
    echo "Directorio '$DEST_DIR' creado."
fi

# Mover el archivo al directorio de destino
if [ -f "$FILENAME" ]; then
    mv "$FILENAME" "$DEST_DIR/"
    echo "Archivo '$FILENAME' movido a '$DEST_DIR/'."
else
    echo "El archivo '$FILENAME' no existe en el directorio de trabajo."
fi

python3 main.py