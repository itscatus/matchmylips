import streamlit as st
from translations import translations
from PIL import Image

image_directory = "./assets/house-door.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Home", 
               "page_icon":image, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

st.logo('./assets/logo.png', size="large")

# Default language in session state
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

# Main Page Content
def main_page():
    lang = st.session_state.language
     # Membuat dua kolom
    col1, col2 = st.columns([1, 3])  # Proporsi kolom 1:3

    # Menampilkan logo di kolom pertama
    with col1:
        st.image("./assets/logo.png", width=200)  # Ubah ukuran jika perlu

    # Menampilkan judul di kolom kedua
    with col2:
        st.title(translations[lang]["app_title"])
        
    st.subheader(translations[lang]["app_description"])
    st.info(translations[lang]["navigation_prompt"])
    st.header(translations[lang]["main_features"])
    st.write(translations[lang]["feature_1"])
    st.write(translations[lang]["feature_2"])
    st.write(translations[lang]["feature_3"])

main_page()
