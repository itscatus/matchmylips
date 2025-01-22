import streamlit as st
import torch
import facer
        
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
