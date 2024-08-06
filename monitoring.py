import streamlit as st
import os
import time
from PIL import Image
from glob import glob
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Configuration des dossiers
FRAMES_DIR = 'frames'
DATA_DIR = 'data'
AGREG_DIR = 'agreg'

# Initialisation des variables
count_data_history = []
latest_image_path = ""

# Fonction pour charger les données de comptage
def load_count_data():
    txt_files = glob(os.path.join(DATA_DIR, '*.txt'))
    if not txt_files:
        return {}
    
    latest_file = max(txt_files, key=os.path.getctime)  # Prendre le fichier le plus récent
    with open(latest_file) as f:
        lines = f.readlines()
        data = {}
        for line in lines:
            key, value = line.strip().split(': ')
            data[key] = int(value)
        return data

# Fonction pour charger les informations d'agrégation
def load_agreg_info():
    agreg_files = glob(os.path.join(AGREG_DIR, '*.txt'))
    if agreg_files:
        with open(agreg_files[0]) as f:
            return eval(f.read())  # Utiliser eval pour convertir la chaîne en dictionnaire
    return "No aggregation info available."

# Fonction pour charger la dernière image
def load_latest_image():
    image_files = glob(os.path.join(FRAMES_DIR, '*.jpg'))
    if image_files:
        latest_image = max(image_files, key=os.path.getctime)  # Prendre le fichier le plus récent
        return latest_image
    return None

# Fonction pour générer le pie chart des données de comptage
def plot_pie_chart(count_data):
    # Génération du pie chart avec les données de comptage affichées
    labels = [key for key in count_data if key != "aggregationId"]
    values = [count_data[key] for key in count_data if key != "aggregationId"]
    
    # Définir les couleurs pour les sections du pie chart
    colors = ['blue', 'lightblue', 'cyan', 'green', 'yellow', 'orange', 'red']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, marker=dict(colors=colors))])
    fig.update_layout(title_text="Count Data Distribution")
    
    return fig

def plot_agreg_map(agreg_info, count_data):
    if isinstance(agreg_info, dict):
        # Récupérer la première paire de coordonnées (latitude1, longitude1)
        latitude = float(agreg_info['latitude1'])
        longitude = float(agreg_info['longitude1'])
        
        # Créer la carte
        m = folium.Map(location=[latitude, longitude], zoom_start=15)
        
        # Calculer la somme des compteurs
        count_sum = sum(count_data[key] for key in count_data if key.endswith('Count'))
        
        # Définir un gradient détaillé pour la heatmap (du bleu au rouge)
        gradient = {
            0.0: 'blue',
            0.1: 'lightblue',
            0.2: 'cyan',
            0.3: 'lightgreen',
            0.4: 'green',
            0.5: 'yellowgreen',
            0.6: 'yellow',
            0.7: 'orange',
            0.8: 'orangered',
            0.9: 'red',
            1.0: 'darkred'
        }
        
        # Définir la valeur maximale pour la heatmap
        max_val = 15
        
        # Calculer l'intensité relative de count_sum par rapport à max_val
        intensity = min(1,count_sum / max_val) if max_val != 0 else 0
        
        # Ajouter un cercle avec heatmap
        HeatMap([[latitude, longitude, intensity]], radius=30, min_opacity=0.2, max_val=1, gradient=gradient).add_to(m)
        
        return m
    return None

# Application Streamlit
def main():
    global count_data_history, latest_image_path

    st.title("Traffic Monitoring Dashboard")

    # Ajout d'un bouton pour accéder au dépôt GitHub
    if st.button('Edit Code on GitHub'):
        st.markdown(
            '<a href="https://github.com/aliamrrr/jetson" target="_blank" style="font-size:18px;">Go to GitHub</a>',
            unsafe_allow_html=True
        )

    # Création des containers vides
    with st.sidebar:
        st.header("Data Count")
        count_data_container = st.empty()

    latest_image_container = st.empty()
    agreg_info_container = st.empty()
    pie_chart_container = st.empty()
    map_container = st.empty()

    # Chargement initial des données
    latest_image = load_latest_image()
    count_data = load_count_data()
    agreg_info = load_agreg_info()

    # Mise à jour des images
    if latest_image:
        latest_image_path = latest_image
        latest_image_container.image(Image.open(latest_image), caption="Latest Frame")

    # Mise à jour des données de comptage
    if count_data:
        count_data_history.append(count_data)
        count_data_container.json(count_data)
        
        # Mise à jour du pie chart
        pie_chart = plot_pie_chart(count_data)
        pie_chart_container.plotly_chart(pie_chart)

    # Mise à jour des informations d'agrégation
    agreg_info_container.text(str(agreg_info))

    # Mise à jour de la carte
    agreg_map = plot_agreg_map(agreg_info, count_data)
    if agreg_map:
        with map_container:
            st_folium(agreg_map, width=700, height=500)

    # Rafraîchir les données toutes les 5 secondes
    while True:
        time.sleep(5)

        new_image = load_latest_image()
        new_count_data = load_count_data()
        new_agreg_info = load_agreg_info()

        # Mise à jour des images si nécessaire
        if new_image and new_image != latest_image_path:
            latest_image_path = new_image
            latest_image_container.image(Image.open(new_image), caption="Latest Frame")

        # Mise à jour des données de comptage si nécessaire
        if new_count_data != count_data:
            count_data = new_count_data
            count_data_history.append(count_data)
            count_data_container.json(count_data)
            
            # Mise à jour du pie chart
            pie_chart = plot_pie_chart(count_data)
            pie_chart_container.plotly_chart(pie_chart)

        # Mise à jour des informations d'agrégation si nécessaire
        if new_agreg_info != agreg_info:
            agreg_info = new_agreg_info
            agreg_info_container.text(str(new_agreg_info))

            # Mise à jour de la carte
            agreg_map = plot_agreg_map(agreg_info, count_data)
            if agreg_map:
                map_container.empty()  # Clear the previous map
                map_container.st_folium(agreg_map, width=700, height=500)

# Fonction de connexion
def login():
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == "jetson" and password == "stage":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid username or password")

# Main application
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    main()
else:
    login()








