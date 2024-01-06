from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Request
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.services.utils import setup_logger
from src.services.training import train_iris_model
from src.schemas.iris import IrisModel, IrisModelPrediction
from src.services.firebase_setup import initialize_firestore
from src.services.rate_limiter import limiter, get_remote_address

router = APIRouter()
logger = setup_logger()

@router.post("/token")
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Generate a token for authentication."""
    if form_data.username == "cyril" and form_data.password == "cyril":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/download-iris")
async def download_iris_dataset():
    """Download the Iris dataset."""
    try:
        dataset_url = 'https://www.kaggle.com/datasets/uciml/iris'
        # Code to download the dataset
        # od.download(dataset_url, 'src/data/')
        return {"message": "Dataset downloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An internal error occurred while downloading the dataset")

@router.get("/load-iris")
async def load_iris_dataset():
    """Load the Iris dataset into memory."""
    try:
        df = pd.read_csv('src/data/iris/Iris.csv')
        return df.to_dict(orient='records')
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found")

@router.post("/process-iris")
async def process_iris_dataset(iris_data_list: List[IrisModel]):
    """Process the Iris dataset."""
    try:
        df = pd.DataFrame([iris.dict() for iris in iris_data_list])
        df_numeric = df.drop(columns=['species'])
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        return pd.DataFrame(df_scaled, columns=df_numeric.columns).to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/split-iris")
async def split_iris_dataset(df: List[IrisModel], test_size: float = 0.2):
    """Split the Iris dataset into training and testing sets."""
    try:
        X_train, X_test = train_test_split(df, test_size=test_size)
        return {"train": X_train, "test": X_test}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model(data: List[IrisModel]):
    """Train a machine learning model using the Iris dataset."""
    try:
        df = pd.DataFrame([item.dict() for item in data])
        X = df.drop(columns=['species'])
        y = df['species']
        model = train_iris_model(X, y)
        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
@limiter.limit("5/minute")
async def make_prediction(request: Request, data: List[IrisModelPrediction]):
    """Make predictions using a trained model."""
    try:
        model = joblib.load('src/models/iris_model.pkl')
        df = pd.DataFrame([item.dict() for item in data])
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

db = initialize_firestore()

@router.get("/get-parameters")
async def get_parameters():
    """Retrieve parameters from Firestore."""
    try:
        params_ref = db.collection('parameters').document('parameters')
        params = params_ref.get()
        if params.exists:
            return params.to_dict()
        else:
            return {"message": "No parameters found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-parameters")
async def update_parameters(new_params: dict):
    """Update parameters in Firestore."""
    try:
        params_ref = db.collection('parameters').document('parameters')
        params_ref.update(new_params)
        return {"message": "Parameters updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-parameters")
async def add_parameters(new_params: dict):
    """Add new parameters to Firestore."""
    try:
        params_ref = db.collection('parameters').document('parameters')
        params_ref.set(new_params)
        return {"message": "Parameters added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
