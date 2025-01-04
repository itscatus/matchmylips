import streamlit as st
from PIL import Image
from translations import translations
import torch
from torchvision import transforms
import facer
import pandas as pd
from torchvision.models import resnet18,ResNet18_Weights

image_directory = "D://UNPAD/Semester 7/Skripshit/websheesh/assets/person-bounding-box.png"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Personal Color Analysis", 
               "page_icon":image, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

# Model and facer initialization
model = resnet18(weights=ResNet18_Weights.DEFAULT) 
num_classes = 4  # Ganti dengan jumlah kelas Anda

# Sesuaikan output layer (fc) dengan jumlah kelas yang diinginkan
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

MODEL_PATH = "best_resnet18_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)  # Muat ke model
model = model.to(device)  # Pindahkan ke device
model.eval()  # Set model ke mode evaluasi

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize facer detectors and parsers
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

# Load colors.csv
colors_csv_path = "D://UNPAD/Semester 7/Skripshit/websheesh/assets/colors.csv"
colors_df = pd.read_csv(colors_csv_path)

# Function for analysis page
def analysis_page():
    st.title("Analisis Seasonal Personal Color dan Warna Bibir")
    uploaded_file = st.file_uploader("Unggah citra wajah untuk analisis:", type=["jpg", "png"])

    if uploaded_file:
        # Save uploaded file locally
        uploaded_image_path = "temp_uploaded_image.jpg"
        with open(uploaded_image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        uploaded_image = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_image, caption="Citra yang diunggah", use_container_width=True)

        # Face detection and personal color analysis
        img_tensor = transforms.ToTensor()(uploaded_image).unsqueeze(0).to(device)
        with torch.no_grad():
            detections = face_detector(img_tensor)

        if len(detections) == 0:
            st.error("Wajah tidak terdeteksi. Harap unggah ulang citra dengan wajah terlihat jelas.")
        else:
            st.success("Wajah terdeteksi! Memulai analisis personal color...")
            processed_image = transform(uploaded_image).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(processed_image)
                confidences = torch.nn.functional.softmax(predictions, dim=1).cpu().numpy()[0]

            # Mapping seasons to predictions
            seasons = ["Spring", "Summer", "Autumn", "Winter"]
            predicted_index = confidences.argmax()
            predicted_season = seasons[predicted_index]
            confidence_percentage = confidences[predicted_index] * 100

            st.write(f"**Hasil:** Tipe Anda adalah **{predicted_season}**!")
            st.write(f"Keyakinan model: **{confidence_percentage:.2f}%**")

            # Filter warna berdasarkan musim yang diprediksi
            season_colors = colors_df[colors_df['season'] == predicted_season]

            st.write(f"Warna yang direkomendasikan untuk tipe {predicted_season}:")
            for _, row in season_colors.iterrows():
                hex_code = row['hex']

                # Display color as a block with the hex code
                color_html = f"""
                <div style="
                    display: inline-block;
                    background-color: {hex_code};
                    width: 50px;
                    height: 50px;
                    border-radius: 50%;
                    margin: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.3);
                    cursor: pointer;
                " title="{hex_code}">
                </div>
                """
                st.markdown(color_html, unsafe_allow_html=True)
                st.write(f"Hex Code: {hex_code}")


# Run the analysis page
analysis_page()
