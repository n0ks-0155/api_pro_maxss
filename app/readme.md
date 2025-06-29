# Структура FastAPI 

## 1. Проверка /health (GET)

### Описание: Проверка работоспособности
### Responce: 

```
{
  "status": "OK",
  "models_loaded": true
}

```

## 2. Проверка /health (POST)

### Описание: Классификация по тексту или URL-изображения
### Обязательный заголовок: ```X-API-Key: 123123```
### Request (JSON):
```
{
  "post_id": "string",
  "text": "string (optional)",
  "photos": [
    {
      "photo_id": "string",
      "url": "string"
    }
  ]
}
```

## Pydantic (schemas.py)

### PhotoSchema
```
class PhotoSchema(BaseModel):
    photo_id: str
    url: str
```

### PredictionSchema
```
class PredictionSchema(BaseModel):
    object_id: str
    category: str
    subcategory: str
    confidence: float
    photo_ids: List[str]
```

### PostRequestSchema
```
class PostRequestSchema(BaseModel):
    post_id: str
    text: str
    photos: List[PhotoSchema]
```

### PostResponseSchema
```
class PostResponseSchema(BaseModel):
    post_id: str
    predictions: List[PredictionSchema]
```
## Управление текстовыми символами (utils.py)

```
def extract_objects(text: str):
    parts = re.split(r'[и,.\-]', text.lower())
    objects = [p.strip() for p in parts if len(p.strip()) > 3]
    return objects
```
## Наиболее частые ошибки:

400: Неправильные входные данные

403: Неверный API-ключ

500: Ошибки при обработке со стороны сервера

## Безопасность:

При отсутствии или неверном ключе возвращается ошибка 403

Значение, заданное по умолчанию - ```123123```
