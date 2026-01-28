#!/bin/bash
# Script de lancement sécurisé pour l'entraînement
# Force l'utilisation du venv et gère les conflits numpy

# Activer le venv
source venv/bin/activate

# Forcer le PYTHONPATH pour utiliser uniquement le venv
export PYTHONPATH="$(pwd):$(pwd)/venv/lib/python3.11/site-packages"
export PYTHONNOUSERSITE=1  # Ignore site-packages utilisateur

# Vérifier les versions
echo "=== Environment Check ==="
python -c "import sys; print(f'Python: {sys.executable}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__} from {numpy.__file__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "========================="
echo ""

# Lancer l'entraînement
python scripts/train.py "$@"
