import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

def get_face_embedding_from_pil(img_pil):
    face = mtcnn(img_pil)
    if face is None:
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.squeeze().cpu().numpy()

def load_known_embeddings(people_dir):
    known_embeddings = {}
    for filename in os.listdir(people_dir):
        if filename.endswith('.jpg'):
            name = filename.split('.')[0]
            img = Image.open(os.path.join(people_dir, filename)).convert('RGB')
            embedding = get_face_embedding_from_pil(img)
            if embedding is not None:
                known_embeddings[name] = embedding
    print(f"Loaded embeddings for: {list(known_embeddings.keys())}")
    return known_embeddings

def recognize_face_embedding(embedding, known_embeddings, threshold=0.7):
    best_match = None
    best_similarity = -1
    for name, known_embedding in known_embeddings.items():
        similarity = 1 - cosine(known_embedding, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    if best_similarity > threshold:
        return best_match, best_similarity
    else:
        return None, None

# Load known faces
people_dir = "people/"
known_embeddings = load_known_embeddings(people_dir)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Starting webcam... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Detect face and get embedding
    face = mtcnn(img_pil)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        name, sim = recognize_face_embedding(embedding, known_embeddings)
        if name:
            cv2.putText(frame, f"{name} ({sim:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        else:
            cv2.putText(frame, "Unknown", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
