#!/bin/bash
# Script de lancement sécurisé pour l'évaluation
# Force l'utilisation du venv

# Activer le venv
source venv/bin/activate

# Forcer le PYTHONPATH pour utiliser uniquement le venv
export PYTHONPATH="$(pwd):$(pwd)/venv/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1

# Lancer l'évaluation
python scripts/evaluate.py "$@"
