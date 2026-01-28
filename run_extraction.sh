#!/bin/bash
# Script de lancement sécurisé pour extraction de landmarks
# Force l'utilisation du venv

# Activer le venv
source venv/bin/activate

# Forcer le PYTHONPATH pour utiliser uniquement le venv
export PYTHONPATH="$(pwd):$(pwd)/venv/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1

# Lancer l'extraction
python scripts/extract_landmarks.py "$@"
