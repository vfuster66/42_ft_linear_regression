# Utilisation d'une image Python 3.10
FROM python:3.10

# Définition du répertoire de travail
WORKDIR /app

# Copier requirements.txt depuis la racine du projet dans /app
COPY requirements.txt /app/

# Installation des dépendances
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copier le dossier app/ dans /app
COPY app/ /app/

# Commande par défaut (modifiable via Makefile)
CMD ["python", "train.py"]
