import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from streamlit_image_coordinates import streamlit_image_coordinates
from translations import translations

image_directory = "D://UNPAD/Semester 7/Skripshit/websheesh/assets/cosmetics.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Shade Check", 
               "page_icon":image, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

if "language" not in st.session_state:
    st.session_state.language = "en"
    
# Sidebar for language selection
with st.sidebar:
    lang = st.session_state.language
    st.header(translations[lang]["header_language"])
    if st.button("EN"):
        st.session_state.language = "en"
    if st.button("ID"):
        st.session_state.language = "id"


def resize_image(image, max_width=600):
    width_percent = (max_width / float(image.size[0]))
    new_height = int((float(image.size[1]) * float(width_percent)))
    resized_image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return resized_image

# Function to get the hex code from RGB
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Function to classify color using KNN
def classify_color(hex, color_data):
    # Convert hex to RGB
    rgb = [int(hex[i:i+2], 16) for i in (1, 3, 5)]
    # Prepare data for KNN
    X = color_data[['R', 'G', 'B']].values
    y = color_data['season'].values
    knn = KNeighborsClassifier(n_neighbors=3) #change the n_neighbors
    knn.fit(X, y)
    return knn.predict([rgb])[0]

# Load seasonal colors
def load_colors():
    return pd.read_csv('colors.csv')

def shade_check_page():
    lang = st.session_state.language
    st.title(translations[lang]["check_title"])
    
    uploaded_file = st.file_uploader((translations[lang]["upload_shade"]), type=["jpg", "png"])
    
    if uploaded_file is not None:
        # Load and resize image
        image = Image.open(uploaded_file)
        image = resize_image(image, max_width=600)
        
        # Convert image to NumPy array for pixel data
        image_array = np.array(image)

        # Allow the user to click on the image to get coordinates
        coordinates = streamlit_image_coordinates(image_array, key='coords')

        if coordinates:
            # Extract RGB values at the selected coordinates
            x = int(coordinates['x'])
            y = int(coordinates['y'])
            
            # Store the selected color in session state
            st.session_state['selected_color'] = image_array[y, x, :3]

            # Display the selected color in a color picker
            st.color_picker(
                (translations[lang]["select_color"]),
                value='#%02x%02x%02x' % tuple(st.session_state['selected_color']),
            )

            # Convert RGB to Hex code
            hex_color = rgb_to_hex(st.session_state['selected_color'])
            st.write((translations[lang]["hex_code"]), f"{hex_color}")

            # Load seasonal colors
            color_data = load_colors()
            color_data['R'] = color_data['hex'].apply(lambda x: int(x[1:3], 16))
            color_data['G'] = color_data['hex'].apply(lambda x: int(x[3:5], 16))
            color_data['B'] = color_data['hex'].apply(lambda x: int(x[5:7], 16))


            # Classify the color as a seasonal color
            season = classify_color(hex_color, color_data)
            st.write((translations[lang]["seasonal_shade"]),f"{season}")

    else:
        st.warning((translations[lang]["warning"]))

shade_check_page()