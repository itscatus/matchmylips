import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
import matplotlib.pyplot as plt
from translations import translations

image_directory = "./assets/lips.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Try-on Lippies", 
               "page_icon":image, 
               "layout":"wide", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

st.logo('./assets/logo.png', size="large")

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

# Inisialisasi model FARL
device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

# Fungsi untuk menghasilkan parsing map
def evaluate(image):
    with torch.no_grad():
        image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
        image_tensor = image_tensor.to(device)

        faces = face_detector(image_tensor)
        if faces['rects'].nelement() == 0:
            return None

        if 'image_ids' in faces:
            faces['image_ids'] = faces['image_ids'].long()

        faces_parsed = face_parser(image_tensor, faces)
        seg_logits = faces_parsed['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1).cpu()
        parsing_map = seg_probs.argmax(1).squeeze(0).cpu().numpy()
        return parsing_map

# Fungsi untuk mewarnai bibir dengan alpha
def apply_lip_color_with_alpha(image, parsing_map, lip_color, alpha):
    """Apply lip color with transparency."""
    image = np.array(image)
    h, w, _ = image.shape

    # Validasi parsing_map
    if parsing_map is None or parsing_map.size == 0:
        raise ValueError("Error: parsing_map is empty or invalid!")

    # Resize parsing_map
    parsing_map_resized = cv2.resize(parsing_map, (w, h), interpolation=cv2.INTER_NEAREST)

    # Ambil mask bibir atas dan bawah
    mask_upper_lip = (parsing_map_resized == 7).astype(np.uint8)
    mask_lower_lip = (parsing_map_resized == 9).astype(np.uint8)

    # Gabungkan mask bibir atas dan bawah
    mask = mask_upper_lip + mask_lower_lip

    # Pastikan mask antara 0 dan 1 untuk alpha blending
    mask = mask.astype(np.float32)  # Mask menjadi float32 untuk blending
    mask /= mask.max()  # Normalisasi mask agar nilainya antara 0 dan 1

    # Aplikasikan warna lipstik dengan alpha blending
    for c in range(3):  # Iterasi channel warna (RGB)
        image[:, :, c] = np.clip(
            image[:, :, c] * (1 - alpha * mask) + lip_color[c] * (alpha * mask),
            0, 255
        ).astype(np.uint8)

    return image

# Aplikasi Streamlit
def virtual_makeup():
    st.title(translations[lang]["try_title"])
    
    # Initialize processed_image outside columns
    processed_image = None
    
    # Create columns
    col1, col2 = st.columns([1, 1])
    
    # Handle image upload first (col1)
    with col1:
        img_file_buffer = st.file_uploader(translations[lang]["upload_image"], type=["jpg", "jpeg", "png"])
        
        if img_file_buffer:
            # Load and display original image
            image = Image.open(img_file_buffer)
            image = ImageOps.exif_transpose(image)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = np.array(image)
            st.subheader(translations[lang]["original"])
            st.image(image, use_container_width=True)
            
            # Add settings
            st.sidebar.divider()
            st.sidebar.title(translations[lang]["setting"])
            lip_color = st.sidebar.color_picker(translations[lang]["choose_lip"], "#FF69B4")
            lip_color = ImageColor.getcolor(lip_color, "RGB")
            alpha = st.sidebar.slider(translations[lang]["adjust_alpha"], 0.0, 1.0, 0.2)
            apply_button = st.sidebar.button(translations[lang]["apply_button"])
            
            if apply_button:
                with st.spinner(translations[lang]["processing"]):
                    parsing_map = evaluate(image)
                    if parsing_map is not None:
                        try:
                            processed_image = apply_lip_color_with_alpha(image, parsing_map, lip_color, alpha)
                        except Exception as e:
                            st.error(f"{translations[lang]['error_process']}{str(e)}")
                    else:
                        st.error(translations[lang]["error_detect"])
        else:
            st.warning(translations[lang]["warning_2"])

    # Show results in col2
    with col2:
        st.write(translations[lang]["try_result"])
        if processed_image is not None:
            st.subheader(translations[lang]["modified"])
            st.image(processed_image, use_container_width=True)
        else:
            st.info(translations[lang]["upload_info"])
            st.warning(translations[lang]["upload_warn"])
    
# Jalankan aplikasi jika script ini dijalankan
virtual_makeup()    