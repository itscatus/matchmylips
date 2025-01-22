import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from utils.translations import translations
from utils.functions import evaluate

# Streamlit page setup
PAGE_CONFIG = {
    "page_title": "Personal Color Analysis",
    "page_icon": "./assets/person-bounding-box.png",
    "layout": "wide",
    "initial_sidebar_state": "auto"
}
st.set_page_config(**PAGE_CONFIG)

if "language" not in st.session_state:
    st.session_state.language = "en"
lang = st.session_state.language

# Sidebar for language selection
with st.sidebar:
    st.header(translations[lang]["header_language"])
    if st.button("EN"):
        st.session_state.language = "en"
    if st.button("ID"):
        st.session_state.language = "id"

# Initialize MobileNet model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_classes = 4  # Update with your number of classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
MODEL_PATH = "./best_mobilenetv2_model.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load colors CSV
colors_csv_path = "./assets/colors.csv"
colors_df = pd.read_csv(colors_csv_path)

# Extract skin function
def extract_skin(image, parsing_map):
    image = np.array(image)  # Convert PIL Image to NumPy array
    h, w, _ = image.shape

    # Resize parsing map to match image dimensions
    parsing_map_resized = cv2.resize(parsing_map, (w, h), interpolation=cv2.INTER_NEAREST)
    skin_mask = (parsing_map_resized == 1).astype(np.uint8)

    # Mask the image to extract skin region
    image[skin_mask == 0] = [0, 0, 0]

    return image

# Classification and recommendation function
def classify_and_recommend(uploaded_image):
    processed_image = transform(uploaded_image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(processed_image)
        confidences = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]

    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    predicted_index = confidences.argmax()
    predicted_season = seasons[predicted_index]
    confidence_percentage = confidences[predicted_index] * 100

    st.write(translations[lang]["season_type"], f"**{predicted_season}**")
    st.write(translations[lang]["confidence"], f"**{confidence_percentage:.2f}%**")

    season_colors = colors_df[colors_df['season'] == predicted_season]
    colors_per_row = 5
    rows = [season_colors[i:i+colors_per_row] for i in range(0, len(season_colors), colors_per_row)]

    for row in rows:
        columns = st.columns(len(row))
        for col, (_, color) in zip(columns, row.iterrows()):
            hex_code = color['hex']
            with col:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="background-color: {hex_code}; width: 50px; height: 50px; border-radius: 50%; margin: 0 auto;"></div>
                        <div style="margin-top: 5px; font-size: 14px;">{hex_code}</div>
                    </div>
                    """, unsafe_allow_html=True
                )

# Main page logic
def analysis_page():
    st.title(translations[lang]["analysis_title"])
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(translations[lang]["upload_image"], type=["jpg", "png"])
        if uploaded_file is not None:
            # Directly open the uploaded image as PIL Image
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            uploaded_image = ImageOps.exif_transpose(uploaded_image)
            st.image(uploaded_image, use_container_width=True)
        else:
            st.warning(translations[lang]["warning_2"])

    with col2:
        st.write(translations[lang]["analyze_result"])

        if uploaded_file is not None:
            # Directly use uploaded_image in the evaluation process
            parsing_map = evaluate(uploaded_image)
            if parsing_map is not None:
                skin_image = extract_skin(uploaded_image, parsing_map)
                # st.image(skin_image, caption="Extracted Skin", use_container_width=True)

            if st.button("Start Analysis"):
                classify_and_recommend(uploaded_image)
        else:
            st.info(translations[lang]["upload_info"])

analysis_page()
