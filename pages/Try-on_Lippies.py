catus
itscatus_
i'll be your peppered treat

Rizz — 17/01/2025 22:17
mari kita ban selebrasi di nangor
kecuali
catus — 17/01/2025 22:17
pls jgn ke tomoro biarlah ini jd tpt gw pusing ngerjain laporan aja
Rizz — 17/01/2025 22:17
di deket DU
naik bus
menarik"
catus — 17/01/2025 22:17
meh agreee
deket du banyak sihh tp ya gt ntah estetik or notnya
kea yg pernah w unjuk ke lu tu yg public space cafe
yg di belokan sblm ke bebek tu
Rizz — 17/01/2025 22:18
ooooooohhhh iya iya
ah sudah lah nanti kita rapatin di grup ae mau main dimana HAHAHAH
catus — 17/01/2025 22:18
seeplah atur je abang kan panitianya
HAHAHA
Rizz — 17/01/2025 22:19
wakawkawkakwkaw
sok sok dilanjut ci star railnya
catus — 17/01/2025 22:19
ngga udh beres
males anjir main conundrum stress
mening lnjut ngoding
Rizz — 17/01/2025 22:19
ril ril main conundrum isinya gacha power up juga jir
smh hyv
catus — 17/01/2025 22:20
klo lg hoki ez bgt
klo lg ampas yaudala
Rizz — 17/01/2025 22:21
WKWKWKKW untung udah out gue
catus — 17/01/2025 22:22
wkaowkaowka heh anda kmrn main genshin jg stres
Rizz — 17/01/2025 22:22
HEH anjir
kalah 50/50 anjir pasti desperate dimana juga
mana gabisa dipake lagi hdehh weapnya
cukuuuuuuuuuuuuuuup
catus — 17/01/2025 22:23
gw jg abis kalah 50/50
tai hoyo
mana hard pity
Rizz — 17/01/2025 22:23
no sunday???
catus — 17/01/2025 22:23
ngga soalnya sunday not my husbu
ak mau ngambil herta
Rizz — 17/01/2025 22:24
WHAT?
padahal kek tipe lu banget
catus — 17/01/2025 22:24
husbu gwe di hsr cmn danheng anyway
Rizz — 17/01/2025 22:24
om om
catus — 17/01/2025 22:24
kaga ye kocakk
Rizz — 17/01/2025 22:24
KAWKAWKAWK
catus — 17/01/2025 22:24
idk how to determine tp mostly husbu-husbu gwe pny distinctive personality
Rizz — 17/01/2025 22:25
wkwkwkwk suuurree suuuurre
wes wes gue lanjut nulis lagi dah cii
adios
catus — Yesterday at 22:51
anyway there's mucha more of it tp yaudela gwe jg lg males cerita but overal u get the situation rightt
Rizz — Today at 9:46
SBB GA KEBACA
but yes dont worry ci
I'll keep my mouth shut
Rizz — Today at 17:24
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
Expand
message.txt
6 KB
Rizz — Today at 17:32
streamlit
streamlit-image-coordinates
scikit-learn
scikit-build
scikit-image
torch>=1.9.1
torchvision
pandas
pillow
numpy
opencv-python-headless
requests
timm
ipywidgets
matplotlib
validators
pyopengl
scipy
catus — Today at 17:37
2025-01-21 10:35:57.796 Examining the path of torch.classes raised: Tried to instantiate class 'path.path', but it does not exist! Ensure that it is registered via torch::class
Rizz — Today at 17:43
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
Expand
message.txt
7 KB
Rizz — Today at 18:03
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
Expand
message.txt
7 KB
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
Expand
message.txt
6 KB
﻿
import numpy as np
import torch
from PIL import Image, ImageColor, ImageOps
import cv2
import streamlit as st
import facer
import matplotlib.pyplot as plt
from translations import translations

# Page config
image_directory = "./assets/lips.png"
image = Image.open(image_directory)
PAGE_CONFIG = {"page_title":"Try-on Lippies", 
               "page_icon":image, 
               "layout":"wide", 
               "initial_sidebar_state":"auto"}
st.set_page_config(**PAGE_CONFIG)

# Logo
st.logo('./assets/logo.png', size="large")

# Language selection without session state
lang = "en"  # Default language
with st.sidebar:
    st.header(translations[lang]["header_language"])
    if st.button("EN"):
        lang = "en"
    if st.button("ID"):
        lang = "id"

# Initialize models directly
device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = facer.face_detector('retinaface/mobilenet', device=device)
face_parser = facer.face_parser('farl/lapa/448', device=device)

def evaluate(image):
    """Generate parsing map for the image."""
    try:
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
            
            # Clear CUDA cache if using GPU
            if device == "cuda":
                torch.cuda.empty_cache()
                
            return parsing_map
    except Exception as e:
        st.error(f"Error in evaluate: {str(e)}")
        return None

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

def virtual_makeup():
    st.title(translations[lang]["try_title"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        img_file_buffer = st.file_uploader(
            translations[lang]["upload_image"], 
            type=["jpg", "jpeg", "png"]
        )
        
        if img_file_buffer:
            try:
                image = Image.open(img_file_buffer)
                image = ImageOps.exif_transpose(image)
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.array(image)
                st.subheader(translations[lang]["original"])
                st.image(image, use_container_width=True)
                
                st.sidebar.divider()
                st.sidebar.title(translations[lang]["setting"])
                lip_color = st.sidebar.color_picker(translations[lang]["choose_lip"], "#FF69B4")
                lip_color = ImageColor.getcolor(lip_color, "RGB")
                alpha = st.sidebar.slider(translations[lang]["adjust_alpha"], 0.0, 1.0, 0.2)
                
                if st.sidebar.button(translations[lang]["apply_button"]):
                    with st.spinner(translations[lang]["processing"]):
                        parsing_map = evaluate(image)
                        if parsing_map is not None:
                            try:
                                processed_image = apply_lip_color_with_alpha(image, parsing_map, lip_color, alpha)
                                with col2:
                                    st.write(translations[lang]["try_result"])
                                    st.subheader(translations[lang]["modified"])
                                    st.image(processed_image, use_container_width=True)
                            except Exception as e:
                                st.error(f"{translations[lang]['error_process']}{str(e)}")
                        else:
                            st.error(translations[lang]["error_detect"])
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        else:
            st.warning(translations[lang]["warning_2"])

    with col2:
        st.write(translations[lang]["try_result"])
        if not img_file_buffer:
            st.info(translations[lang]["upload_info"])


virtual_makeup()