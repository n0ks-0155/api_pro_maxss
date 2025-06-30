import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import logging
from PIL import Image
import io
import aiohttp
import tensorflow as tf

#Config
API_KEY = "123123"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

CATEGORIES = {
    "weapon": {
        "name": "Оружие",
        "subcategories": ["Пистолеты", "Винтовки", "Дробовики", "Гранатомёты"]
    },
    "equipment": {
        "name": "Экипировка",
        "subcategories": ["Тактические жилеты", "Шлемы", "Перчатки", "Очки"]
    },
    "ammo": {
        "name": "Боеприпасы",
        "subcategories": ["Патроны", "Гранаты", "Газовые баллончики"]
    }
}

IMAGE_SIZE = (224, 224)
THRESHOLD = 0.5
TTA_STEPS = 5 

def f1_score(y_true, y_pred, threshold=0.5):      #Заглушка
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    precision = tp / (tf.reduce_sum(y_pred) + 1e-7)
    recall = tp / (tf.reduce_sum(y_true) + 1e-7)
    return 2 * (precision * recall) / (precision + recall + 1e-7)

try:                                                #Загрузка моделей
    model_image = load_model(
        "./multi_label_classifier_improved.h5",
        custom_objects={'f1_score': f1_score}
    )
    model_text = joblib.load("./logistic_regression_model.joblib")
    vectorizer = joblib.load("./tfidf_vectorizer.joblib")
    label_encoder = joblib.load("./label_encoder.joblib")

    mlb_subcat_classes = [                                   
        "Пистолеты", "Винтовки", "Дробовики", "Гранатомёты",       #Определение классов
        "Тактические жилеты", "Шлемы", "Перчатки", "Очки",
        "Патроны", "Гранаты", "Газовые баллончики"
    ]

    mlb_cat_classes = ["weapon", "equipment", "ammo"]

except Exception as e:
    logging.error(f"Error loading models or classes: {e}")
    raise RuntimeError("Failed to load models or class labels")
    
app = FastAPI(                                            #Сам API-сервер
    title="StrikeGear Classifier API",
    description="API для классификации изображений и текста вооружения",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

                             
class PhotoUrl(BaseModel):              #Pydantic
    photo_id: str
    url: str


class PredictionResult(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]


class PostRequestUrl(BaseModel):
    post_id: str
    text: Optional[str] = None
    photos: Optional[List[PhotoUrl]] = None


class PostResponse(BaseModel):
    post_id: str
    predictions: List[PredictionResult]

def get_api_key(api_key: str = Depends(api_key_header)):        #Authorisation
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key


                                                                #Обработчик
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={"Access-Control-Allow-Origin": "*"}
    )

async def download_image(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.read()
                return None
    except Exception as e:
        logging.error(f"Error downloading image from {url}: {e}")
        return None


async def process_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image)
        image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
        if image_array.shape != (*IMAGE_SIZE, 3):
            raise ValueError(f"Unexpected image shape: {image_array.shape}")
        return image_array
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")


def predict_text(text: str):
    try:
        text_vector = vectorizer.transform([text])
        probabilities = model_text.predict_proba(text_vector)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = probabilities[predicted_idx]

        category = next(
            (cat for cat, data in CATEGORIES.items()
             if predicted_label in data["subcategories"]),
            "ammo"
        )

        return {
            "category": CATEGORIES[category]["name"],
            "subcategory": predicted_label,
            "confidence": float(confidence),
            "source": "text"
        }
    except Exception as e:
        logging.error(f"Text prediction error: {e}")
        return None


def predict_image(image_array, tta_steps=TTA_STEPS, threshold=THRESHOLD):
    subcat_preds = []
    cat_preds = []

    for _ in range(tta_steps):
        augmented_img = tf.image.random_flip_left_right(image_array)
        augmented_img = tf.image.random_brightness(augmented_img, 0.1)
        augmented_img = tf.expand_dims(augmented_img, axis=0)
        subcat_pred, cat_pred = model_image.predict(augmented_img)
        subcat_preds.append(subcat_pred)
        cat_preds.append(cat_pred)

    subcat_pred_avg = np.mean(subcat_preds, axis=0)
    cat_pred_avg = np.mean(cat_preds, axis=0)

    subcat_indices = np.where(subcat_pred_avg[0] > threshold)[0]
    cat_indices = np.where(cat_pred_avg[0] > threshold)[0]

    predicted_subcats = [mlb_subcat_classes[i] for i in subcat_indices]
    predicted_cats = [mlb_cat_classes[i] for i in cat_indices]

    subcat_probs = {mlb_subcat_classes[i]: float(subcat_pred_avg[0][i]) for i in range(subcat_pred_avg.shape[1])}
    cat_probs = {mlb_cat_classes[i]: float(cat_pred_avg[0][i]) for i in range(cat_pred_avg.shape[1])}

    if len(subcat_indices) > 0:
        max_idx = subcat_indices[np.argmax(subcat_pred_avg[0][subcat_indices])]
        predicted_subcat = mlb_subcat_classes[max_idx]
        confidence = float(subcat_pred_avg[0][max_idx])
    else:
        predicted_subcat = "Неизвестно"
        confidence = 0.0

    if len(cat_indices) > 0:
        max_cat_idx = cat_indices[np.argmax(cat_pred_avg[0][cat_indices])]
        predicted_cat = mlb_cat_classes[max_cat_idx]
    else:
        predicted_cat = "Неизвестно"

    return {
        "category": predicted_cat,
        "subcategory": predicted_subcat,
        "confidence": confidence,
        "source": "image"
    }


                                                                #Эндроипнты

@app.post("/predict/url", response_model=PostResponse)
async def predict_from_url(
        request: PostRequestUrl,
        api_key: str = Depends(get_api_key)
):
    if not request.text and not request.photos:
        raise HTTPException(status_code=400, detail="Необходимо предоставить текст или изображения")

    predictions = []

                                                            #Предсказание по тексту
    if request.text:
        text_pred = predict_text(request.text)
        if text_pred:
            predictions.append(PredictionResult(
                object_id=f"text_1",
                category=text_pred["category"],
                subcategory=text_pred["subcategory"],
                confidence=text_pred["confidence"],
                photo_ids=[]
            ))

                                                            #Предсказание по URL изображения
    if request.photos:
        for idx, photo in enumerate(request.photos):
            try:
                image_data = await download_image(photo.url)
                if image_data:
                    image_array = await process_image(image_data)
                    img_pred = predict_image(image_array)
                    predictions.append(PredictionResult(
                        object_id=f"img_{idx + 1}",
                        category=img_pred["category"],
                        subcategory=img_pred["subcategory"],
                        confidence=img_pred["confidence"],
                        photo_ids=[photo.photo_id]
                    ))
            except Exception as e:
                logging.error(f"Error processing photo URL {photo.photo_id}: {e}")

    if not predictions:
        raise HTTPException(status_code=400, detail="Не удалось обработать текст и изображение")

    return PostResponse(post_id=request.post_id, predictions=predictions)


@app.post("/predict/upload", response_model=PostResponse)
async def predict_from_upload(
        post_id: str = Form(...),
        photos: List[UploadFile] = File(...),
        api_key: str = Depends(get_api_key)
):
    if not photos:
        raise HTTPException(status_code=400, detail="Необходимо предоставить изображения")

    predictions = []

    for idx, photo in enumerate(photos):
        try:
            image_data = await photo.read()
            image_array = await process_image(image_data)
            img_pred = predict_image(image_array)
            predictions.append(PredictionResult(
                object_id=f"img_{idx + 1}",
                category=img_pred["category"],
                subcategory=img_pred["subcategory"],
                confidence=img_pred["confidence"],
                photo_ids=[photo.filename or f"photo_{idx + 1}"]
            ))
        except Exception as e:
            logging.error(f"Error processing uploaded photo {photo.filename}: {e}")
            predictions.append(PredictionResult(
                object_id=f"img_{idx + 1}",
                category="Неизвестно",
                subcategory="Неизвестно",
                confidence=0.0,
                photo_ids=[photo.filename or f"photo_{idx + 1}"]
            ))

    return PostResponse(post_id=post_id, predictions=predictions)


@app.get("/health")
async def health_check():
    return {"status": "OK", "models_loaded": True}
