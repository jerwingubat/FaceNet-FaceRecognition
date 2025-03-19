# Face Recognition with MTCNN and InceptionResnetV1

This project implements a face recognition system using the MTCNN (Multi-task Cascaded Convolutional Networks) for face detection and InceptionaqResnetV1 for generating face embeddings. The system is capable of identifying known faces from a webcam feed by comparing their embeddings to previously stored known embeddings.

## Requirements

To run this project, you'll need to install the following dependencies:

- Python 3.12.6
- OpenCV (`cv2`)
- PyTorch
- Facenet-pytorch
- NumPy
- SciPy
- PIL (Pillow)

You can install the necessary dependencies via `pip`:

```bash
pip install opencv-python torch facenet-pytorch numpy scipy pillow
```
## Setup
### 1. Install Dependencies
First, make sure you have Python 3.x installed. Then, install the required libraries using pip:
```cmd
pip install opencv-python torch facenet-pytorch numpy scipy pillow
```
### 2. Prepare Pretrained Models
The InceptionResnetV1 and MTCNN models are pretrained on the VGGFace2 dataset and are available through the facenet-pytorch library. These models will be automatically loaded when running the script, so no additional manual setup is required.
### 3. Prepare Known Faces
You need to prepare a directory (e.g., people/) where you store images of known individuals. Each image should be named after the person (e.g., jerwin_gubat.jpg). These images will be used to generate embeddings for recognition.
- Create a folder called people/ inside your project directory.
- Add images of people you want the system to recognize in the people/ directory. Each image should be named after the person (e.g., jerwin_gubat.jpg).
### 4. Directory Structure
Your project directory should look like this:
```
FaceNet/
├── people/
│   ├── jerwin_gubat.jpg
│   └── juan_delacruz.jpg
├── app.py
└── README.md
```
## How it Works
### Face Detection
The MTCNN model detects faces in a webcam feed. If no faces are detected, the program will display a message indicating so.

### Face Embeddings
The face detected by MTCNN is passed to the InceptionResnetV1 model, which generates a 512-dimensional face embedding.

### Face Recognition
The face embedding from the webcam feed is compared against the embeddings of known individuals stored in the people/ directory using cosine similarity. If a match is found, the name and similarity score are displayed on the screen.

## Step-by-Step Process
1. Load known embeddings from images in the people/ directory.
2. Open the webcam feed using OpenCV.
3. For each frame:
     - Detect faces using MTCNN.
     - Generate embeddings for the detected face using InceptionResnetV1.
     - Compare the generated embedding with the stored embeddings of known individuals.
     - Display the recognized person's name and similarity score if a match is found.
     - Exit the webcam feed by pressing the 'q' key.
## Usage
Place images of the known individuals in the people/ directory.

Run the script:
```
python app.py
```
The webcam feed will open, and the system will try to recognize faces in the video feed. If it detects a known face, it will display the name and similarity score on the screen. If the face is not recognized, it will show "Unknown."

Press 'q' to quit the webcam feed.

## Troubleshooting
- No face detected: Ensure the webcam is working and that faces are clearly visible in the frame.
- Low similarity score: If the system fails to recognize faces with high accuracy, you can adjust the threshold for recognition in the code (default is 0.7).
## License
This project is licensed under the MIT License.
