from contextlib import asynccontextmanager
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Сервіс бере модель з того самого сервера MLflow, на який ми логували запуски
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')

# Словник для завантаженої моделі: заповнюється один раз під час запуску сервера
models = {}


@asynccontextmanager
async def lifespan(app):
    # До yield: завантажуємо з реєстру ту версію моделі, яка має аліас champion
    models['champion'] = mlflow.pyfunc.load_model(
        model_uri='models:/bigmart_sales@champion')
    yield
    # Після yield: звільняємо пам'ять під час зупинки сервера
    models.clear()


app = FastAPI(title='BigMart Sales API', lifespan=lifespan)


class Item(BaseModel):
    # Схема запиту повторює ознаки, на яких навчено конвеєр (сигнатуру моделі)
    Item_Weight: float
    Item_Fat_Content: str
    Item_Visibility: float
    Item_Type: str
    Item_MRP: float
    Outlet_Identifier: str
    Outlet_Establishment_Year: int
    Outlet_Size: str
    Outlet_Location_Type: str
    Outlet_Type: str


@app.post('/predict')
def predict(item: Item):
    # Один товар перетворюємо на таблицю з одним рядком, як очікує конвеєр
    X = pd.DataFrame([item.model_dump()])
    # Повертаємо прогноз продажів числом
    return {'Item_Outlet_Sales': float(models['champion'].predict(X)[0])}
