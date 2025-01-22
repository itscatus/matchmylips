import streamlit as st
import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import pandas as pd
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import facer
from translations import translations

# Initialize MobileNet model
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_classes = 4  # Update with your number of classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
MODEL_PATH = "./best_mobilenetv2_model.pth"
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

# Initialize device and models
device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

# Load colors CSV
colors_csv_path = "./assets/colors.csv"
colors_df = pd.read_csv(colors_csv_path)

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

# Parsing map function
def evaluate(image_path):
    try:
        with torch.no_grad():
            image = Image.open(image_path).convert("RGB")
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
            image_tensor = image_tensor.to(device)

            faces = face_detector(image_tensor)
            if faces['rects'].nelement() == 0:
                return None

            faces['image_ids'] = faces['image_ids'].long() if 'image_ids' in faces else None
            faces_parsed = face_parser(image_tensor, faces)
            seg_logits = faces_parsed['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1).cpu()
            parsing_map = seg_probs.argmax(1).squeeze(0).cpu().numpy()

            return parsing_map
    except Exception as e:
        st.error(f"Error in evaluate: {e}")
        return None

# Extract skin function
def extract_skin(image_path, parsing_map):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    h, w, _ = image.shape

    parsing_map_resized = cv2.resize(parsing_map, (w, h), interpolation=cv2.INTER_NEAREST)
    skin_mask = (parsing_map_resized == 1).astype(np.uint8)

    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    # image_rgba[:, :, 3] = skin_mask * 255

    return image_rgba

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
            uploaded_image_path = "temp_uploaded_image.jpg"
            with open(uploaded_image_path, "wb") as f:
                f.write(uploaded_file.read())

            uploaded_image = Image.open(uploaded_file).convert("RGB")
            uploaded_image = ImageOps.exif_transpose(uploaded_image)
            st.image(uploaded_image, use_container_width=True)
        else:
            st.warning(translations[lang]["warning_2"])

    with col2:
        st.write(translations[lang]["analyze_result"])

        if uploaded_file is not None:
            parsing_map = evaluate(uploaded_image_path)
            if parsing_map is not None:
                skin_image = extract_skin(uploaded_image_path, parsing_map)
                st.image(skin_image, caption="Extracted Skin", use_container_width=True)

            if st.button("Start Analysis"):
                classify_and_recommend(uploaded_image)
        else:
            st.info(translations[lang]["upload_info"])

analysis_page()
