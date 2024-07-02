# Utilise l'image officielle Python 3.11 slim du Docker Hub
FROM python:3.11-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances nécessaires pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Installer les dépendances supplémentaires pour d'autres librairies
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Copie le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Crée un environnement virtuel Python dans /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Met à jour pip et installe les dépendances listées dans requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Définit une variable d'environnement pour désactiver le buffering en sortie de Python
ENV PYTHONUNBUFFERED 1

# Copie le reste du code de l'application dans le conteneur
COPY . /app

# Expose le port 8501 (port par défaut pour Streamlit)
EXPOSE 8501

# Commande pour démarrer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
