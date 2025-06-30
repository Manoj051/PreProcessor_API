import cv2
import numpy as np
import mediapipe as mp
import math
import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def align_face(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

def preprocess_face_image_bytes(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image data"}

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enhanced = apply_clahe(frame_rgb)

    results = face_mesh.process(enhanced)
    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    face_landmarks = results.multi_face_landmarks[0]
    ih, iw, _ = enhanced.shape

    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    left_eye_coords = (int(left_eye.x * iw), int(left_eye.y * ih))
    right_eye_coords = (int(right_eye.x * iw), int(right_eye.y * ih))

    aligned = align_face(enhanced, left_eye_coords, right_eye_coords)

    xs = [lm.x * iw for lm in face_landmarks.landmark]
    ys = [lm.y * ih for lm in face_landmarks.landmark]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    cropped_face = aligned[y1:y2, x1:x2]
    if cropped_face.size == 0:
        return {"error": "Cropped face area is empty"}

    try:
        resized = cv2.resize(cropped_face, (160, 160), interpolation=cv2.INTER_AREA)
    except:
        return {"error": "Failed to resize face"}

    normalized = resized.astype('float32')
    normalized = (normalized - 127.5) / 128.0

    buffer = io.BytesIO()
    np.save(buffer, normalized)
    buffer.seek(0)
    b64_npy = base64.b64encode(buffer.read()).decode('utf-8')

    return {"embedding_input": b64_npy}

@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = preprocess_face_image_bytes(contents)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
