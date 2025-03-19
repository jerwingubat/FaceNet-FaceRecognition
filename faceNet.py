import os
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
import csv
from datetime import datetime
import time

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Create CSV files if they don't exist
if not os.path.exists("recognition_log.csv"):
    with open("recognition_log.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Similarity", "DateTime"])

if not os.path.exists("current_recognitions.csv"):
    with open("current_recognitions.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Similarity", "DateTime"])

def log_to_history(name, similarity):
    """Append to recognition_log.csv"""
    with open("recognition_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([name, similarity, now])

def log_or_update_current(name, similarity):
    """Update current_recognitions.csv with latest detection for that person."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_records = {}

    # Load existing data
    if os.path.exists("current_recognitions.csv"):
        with open("current_recognitions.csv", mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                current_records[row["Name"]] = {"Similarity": row["Similarity"], "DateTime": row["DateTime"]}

    # Update with new data
    current_records[name] = {"Similarity": f"{similarity:.4f}", "DateTime": now}

    # Save updated data
    with open("current_recognitions.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Similarity", "DateTime"])
        for person, data in current_records.items():
            writer.writerow([person, data["Similarity"], data["DateTime"]])

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

people_dir = "people/"
known_embeddings = load_known_embeddings(people_dir)

last_logged_time = {}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Starting webcam... press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to PIL image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    face = mtcnn(img_pil)
    if face is not None:
        with torch.no_grad():
            embedding = resnet(face.unsqueeze(0).to(device)).squeeze().cpu().numpy()

        name, sim = recognize_face_embedding(embedding, known_embeddings)
        if name and sim > 0.7:
            cv2.putText(frame, f"{name} ({sim:.2f})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            current_time = time.time()

            if name not in last_logged_time or (current_time - last_logged_time[name] > 5):
                log_to_history(name, sim)
                log_or_update_current(name, sim)
                last_logged_time[name] = current_time
        else:
            cv2.putText(frame, "Unknown", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    else:
        cv2.putText(frame, "No Face Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
