import numpy as np
import torch
from PIL import Image, ImageOps
import cv2
import facer
import streamlit as st
from torchvision import transforms
import pandas as pd
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from translations import translations

# Model and facer initialization
device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

# Streamlit setup
image_directory = "./assets/person-bounding-box.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title": "Personal Color Analysis", 
               "page_icon": image, 
               "layout": "wide", 
               "initial_sidebar_state": "auto"}
st.set_page_config(**PAGE_CONFIG)

st.image('./assets/logo.png', use_column_width=True)

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

# Load model for personal color analysis
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
num_classes = 4  # Adjust based on your dataset
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

MODEL_PATH = "./best_mobilenetv2_model.pth"
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load colors CSV
colors_csv_path = "./assets/colors.csv"
colors_df = pd.read_csv(colors_csv_path)

# Function to evaluate and get parsing map
def evaluate(image):
    try:
        # Convert to tensor and move to device
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).to(torch.uint8).to(device)
        
        # Detect faces
        faces = face_detector(image_tensor)
        if faces['rects'].nelement() == 0:
            st.error(translations[lang]["error_no_faces"])
            return None

        if 'image_ids' in faces:
            faces['image_ids'] = faces['image_ids'].long()

        # Parse faces
        faces_parsed = face_parser(image_tensor, faces)
        seg_logits = faces_parsed['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1).cpu()
        
        # Generate parsing map
        parsing_map = seg_probs.argmax(1).squeeze(0).cpu().numpy()
        return parsing_map
    except Exception as e:
        st.error(f"Error in evaluate: {e}")
        return None

# Function to extract skin from image using parsing map
def extract_skin(image, parsing_map):
    try:
        # Convert image to numpy array
        image_np = np.array(image)
        h, w, _ = image_np.shape

        if parsing_map is None or parsing_map.size == 0:
            raise ValueError(translations[lang]["error_parsing_map"])

        # Resize the parsing map to match image dimensions
        parsing_map_resized = cv2.resize(parsing_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # Create mask for skin
        skin_mask = (parsing_map_resized == 1).astype(np.uint8)

        # Create RGBA image to apply transparency
        image_rgba = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)

        # Set alpha channel based on skin_mask
        image_rgba[:, :, 3] = skin_mask * 255
        return image_rgba
    except Exception as e:
        st.error(f"Error in extract_skin: {e}")
        return None

# Function to process skin extraction in the uploaded image
def upload_img(uploaded_image):
    # Extract parsing map
    parsing_map_result = evaluate(uploaded_image)
    if parsing_map_result is None:
        return None

    # Extract skin using the parsing map
    skin_image = extract_skin(uploaded_image, parsing_map_result)
    if skin_image is not None:
        st.image(skin_image, caption="Extracted Skin Area", use_column_width=True)
    return skin_image

# Function to classify the personal color based on extracted skin image
def classify_spca(processed_skin_image):
    processed_image = transform(processed_skin_image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(processed_image)
        confidences = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]

    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    predicted_index = confidences.argmax()
    predicted_season = seasons[predicted_index]
    confidence_percentage = confidences[predicted_index] * 100

    st.write(translations[lang]["season_type"], f"**{predicted_season}**")
    st.write(translations[lang]["confidence"], f"**{confidence_percentage:.2f}%**")

    # Filter colors based on predicted season
    season_colors = colors_df[colors_df['season'] == predicted_season]

    st.write(translations[lang]["recommendation"], f"{predicted_season}:")

    # Display colors
    colors_per_row = 5
    rows = [season_colors[i:i + colors_per_row] for i in range(0, len(season_colors), colors_per_row)]

    for row in rows:
        columns = st.columns(len(row))
        for col, (_, color) in zip(columns, row.iterrows()):
            hex_code = color['hex']
            with col:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <div style="background-color: {hex_code}; width: 50px; height: 50px; border-radius: 50%; margin: 0 auto; box-shadow: 0 0 5px rgba(0,0,0,0.3);"></div>
                        <div style="margin-top: 5px; font-size: 14px; color: #333;">{hex_code}</div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

# Function for the analysis page
def analysis_page():
    st.title(translations[lang]["analysis_title"])

    # Create two columns
    col1, col2 = st.columns(2)

    # Left column: Upload image
    with col1:
        uploaded_file = st.file_uploader(translations[lang]["upload_image"], type=["jpg", "png"])

        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file).convert("RGB")
            uploaded_image = ImageOps.exif_transpose(uploaded_image)
            st.image(uploaded_image, use_container_width=True)
        else:
            st.warning(translations[lang]["warning_2"])

    # Right column: Analysis results
    with col2:
        st.write(translations[lang]["analyze_result"])
        
        if uploaded_file is not None:
            skin_image = upload_img(uploaded_image)

            if st.button("Start Analysis") and skin_image is not None:
                classify_spca(skin_image)
        else:
            st.info(translations[lang]["upload_info"])

# Run the analysis page
analysis_page()
