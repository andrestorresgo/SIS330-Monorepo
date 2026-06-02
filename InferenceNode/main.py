import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from collections import deque
import logging
from models.architectures import HemoScout, VentGuardian, MicroTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_node")

app = FastAPI(title="Inference Node (CTDE Architecture)")

# --- Global State ---
# Models will be loaded during the startup event
MODELS = {
    "hemo": None,
    "vent": None,
    "transformer": None
}

# Buffer to store the last 60 seconds of Tier 1 probabilities per patient
# Key: patient_id, Value: deque of [hemo_prob, vent_prob]
patient_buffers: Dict[str, deque] = {}

# --- Pydantic Schemas ---
class SensorData(BaseModel):
    patient_id: str
    timestamp: float
    pleth_wave: List[float] # Exactly 100 floats (1 second at 100Hz)
    co2_wave: List[float]   # Exactly 100 floats (1 second at 100Hz)

class Tier1Response(BaseModel):
    hemo_risk: float
    vent_risk: float

class Tier2Response(BaseModel):
    system_risk: float
    alarm_suppressed: bool

class InferenceResponse(BaseModel):
    patient_id: str
    status: str # "CALIBRANDO" or "MONITORIZANDO"
    tier_1: Tier1Response
    tier_2: Tier2Response

# --- Lifespan / Startup ---
@app.on_event("startup")
async def load_models():
    """
    Fase 3: Carga de Modelos (Arranque en Frío)
    Carga los pesos en memoria RAM/VRAM una sola vez.
    """
    try:
        # Instantiate architectures
        MODELS["hemo"] = HemoScout()
        MODELS["vent"] = VentGuardian()
        MODELS["transformer"] = MicroTransformer()

        # Load weights
        MODELS["hemo"].load_state_dict(torch.load("hemo_demo.pth", map_location=torch.device('gpu')))
        MODELS["vent"].load_state_dict(torch.load("vent_demo.pth", map_location=torch.device('gpu')))
        MODELS["transformer"].load_state_dict(torch.load("transformer_demo.pth", map_location=torch.device('gpu')))

        # Set to evaluation mode
        for model in MODELS.values():
            model.eval()
        
        logger.info("All models loaded successfully in eval mode.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        # TODO: In production, we might want to exit here
        raise e

# --- Inference Logic ---
@app.post("/predict", response_model=InferenceResponse)
async def predict(data: SensorData):
    """
    Fase 4: Diseño del API y Lógica de Inferencia (Pipeline Secuencial)
    """
    # 1. Validation of input size
    if len(data.pleth_wave) != 100 or len(data.co2_wave) != 100:
        raise HTTPException(status_code=400, detail="Each wave must contain exactly 100 samples (1 second at 100Hz).")

    # Initialize buffer for new patients
    if data.patient_id not in patient_buffers:
        patient_buffers[data.patient_id] = deque(maxlen=60)

    # 2. Sequential Inference Pipeline
    with torch.no_grad():
        # Paso A: Preprocessing
        # Convert to tensors [Batch, Channel, Length] -> [1, 1, 100]
        pleth_tensor = torch.tensor(data.pleth_wave, dtype=torch.float32).view(1, 1, 100)
        co2_tensor = torch.tensor(data.co2_wave, dtype=torch.float32).view(1, 1, 100)

        # Paso B: Tier 1 (Sentidos)
        hemo_output = MODELS["hemo"](pleth_tensor)
        vent_output = MODELS["vent"](co2_tensor)

        # Paso C: Probabilities and Buffer Update
        hemo_prob = torch.sigmoid(hemo_output).item()
        vent_prob = torch.sigmoid(vent_output).item()

        # Update patient state
        patient_buffers[data.patient_id].append([hemo_prob, vent_prob])

        # Paso D: Tier 2 (Cerebro)
        status = "CALIBRANDO"
        system_risk = 0.0
        alarm_suppressed = True

        if len(patient_buffers[data.patient_id]) == 60:
            status = "MONITORIZANDO"
            # Convert deque to tensor [Batch, Sequence, Features] -> [1, 60, 2]
            buffer_data = list(patient_buffers[data.patient_id])
            transformer_input = torch.tensor(buffer_data, dtype=torch.float32).view(1, 60, 2)
            
            transformer_output = MODELS["transformer"](transformer_input)
            system_risk = torch.sigmoid(transformer_output).item()
            alarm_suppressed = False

        # Paso E: Return response
        return InferenceResponse(
            patient_id=data.patient_id,
            status=status,
            tier_1=Tier1Response(hemo_risk=hemo_prob, vent_risk=vent_prob),
            tier_2=Tier2Response(system_risk=system_risk, alarm_suppressed=alarm_suppressed)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
