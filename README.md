# Face Preprocessing API - Modular Face Recognition System Component

This repository contains the **Face Preprocessor** module for a modular face recognition system. The complete system consists of four major components:

1. **Preprocessor (this repo)** – Processes raw face images and prepares them for embedding extraction.
2. **Embedding Generator** – Generates 128D/512D feature vectors from preprocessed face tensors.
3. **Spring Boot Hub** – Acts as a central API orchestrator, calling other modules and managing control flow.
4. **Database System** – Stores embeddings, face IDs, and associated metadata.

This modular architecture enables flexibility, scalability, and maintainability for enterprise-level face recognition applications.

---

## 🧠 What This Module Does

This FastAPI-based service accepts an uploaded face image and returns a **base64-encoded `.npy` tensor** that is ready to be fed into a face embedding model.

- **CLAHE** enhancement is applied to boost contrast
- **MediaPipe Face Mesh** detects faces using landmarks (CPU-efficient)
- **Alignment** is done based on eye coordinates for consistent orientation
- The face is **cropped**, **resized** to `(160, 160)` and **normalized** to [-1, 1] range
- The output is serialized as a `.npy` NumPy array and base64-encoded for transport

This prepares data for deep learning-based face embedding extraction (e.g., FaceNet, ArcFace).

---

## 📦 Requirements

```
fastapi==0.110.0
uvicorn==0.29.0
opencv-python==4.9.0.80
mediapipe==0.10.9
numpy==1.24.4
python-multipart==0.0.9
```

---

## 🚀 How to Set Up

### 1. Create Virtual Environment
```
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the API Server
```
uvicorn app:app --reload
```
Then open your browser to: `http://127.0.0.1:8000/docs`

---

## 🧪 Using the API

### POST `/preprocess`
**Input:** Form-data with a key `file` and a JPEG/PNG image.

**Output:** JSON with base64-encoded `.npy` array:
```
{
  "embedding_input": "<base64-string>"
}
```

### Decode in Python for Validation
```python
import base64, io, numpy as np
from matplotlib import pyplot as plt

with open("response.json") as f:
    data = json.load(f)
    b64_str = data["embedding_input"]

decoded = base64.b64decode(b64_str)
face_array = np.load(io.BytesIO(decoded))
image = ((face_array * 128.0) + 127.5).astype(np.uint8)
plt.imshow(image)
plt.axis("off")
plt.show()
```

---

## 💡 Why It Matters (And Why It’s Impressive)

This module represents a clean, scalable entry point to a facial recognition system. Key qualities:

- **Modular**: Can plug into any modern microservice architecture
- **Lightweight**: MediaPipe is fast, CPU-friendly, and doesn’t require a GPU
- **Deployable**: Easily containerized for cloud or edge deployments
- **Production-Ready**: Input validation, structured outputs, clean format for ML models
- **Real-World Use Case**: Aligns with modern security systems, biometric login flows, attendance tracking, and more

As a fresher, this demonstrates my ability to:
- Design modular APIs
- Work with computer vision and deep learning tooling
- Build real microservices with FastAPI
- Integrate image processing pipelines in Python

---

## 🔮 Next Steps in the Project

- Build the **Embedding Generator** microservice that loads a FaceNet model and returns 128D vectors
- Implement a **Spring Boot Hub** API to coordinate the pipeline and act as the central service
- Connect a **Database** (PostgreSQL or MongoDB) to store user info and face vectors
- Build face **verification and search** endpoints using cosine similarity
- Optional: Add Docker and CI/CD for deployment

---

## 🤝 Contact

This project is part of a larger modular face recognition system built as a job-seeking showcase. Feel free to connect or contribute if you're working on related ideas.

> Built with FastAPI, MediaPipe, and every ounce of patience.
