from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS so React can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CalculationData(BaseModel):
    a: float
    b: float

@app.post("/add")
async def add_numbers(data: CalculationData):
    return {"result": data.a + data.b}

# Added a variety feature: Subtraction
@app.post("/subtract")
async def subtract_numbers(data: CalculationData):
    return {"result": data.a - data.b}