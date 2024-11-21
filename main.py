from fastapi import FastAPI
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
import os
import uvicorn
import numpy as np
app = FastAPI()
iris = load_iris()
X = iris.data
y = iris.target
print(X)
model = RandomForestClassifier(n_estimators=100)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model.fit(X_train,y_train)

class IrisFeatures(BaseModel):
    sepal_length:float
    sepal_width :float
    petal_length:float
    petal_width:float

@app.post("/predictiris")
def predict_iris(features:IrisFeatures):
    features = np.array([[features.sepal_length,features.sepal_width,features.petal_length,features.petal_width]])
    prediction = model.predict(features)
    return {'Species':iris.target_names[prediction[0]]}
if __name__ == "__main__":
    port = int(os.getenv("PORT",8000))
    uvicorn.run("main:app",host="0.0.0.0",port=port,reload=True)