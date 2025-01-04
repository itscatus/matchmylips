import streamlit as st
from translations import translations

import streamlit as st 
from PIL import Image

image_directory = "D://UNPAD/Semester 7/Skripshit/websheesh/assets/house-door.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Home", 
               "page_icon":image, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

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
    st.image("logo.svg", width=100)
    st.title(translations[lang]["app_title"])
    st.write(translations[lang]["app_description"])
    st.write(translations[lang]["navigation_prompt"])
    st.header(translations[lang]["main_features"])
    st.write(translations[lang]["feature_1"])
    st.write(translations[lang]["feature_2"])

main_page()
