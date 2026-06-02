import asyncio
import httpx
import time
import random
import numpy as np

# Configuración
URL = "http://localhost:8000/predict"
PATIENT_ID = "paciente_test_stress"
TOTAL_REQUESTS = 500  # Para la prueba
CONCURRENT_REQUESTS = 20 # Número de tareas concurrentes para alcanzar la tasa deseada

async def send_inference_request(client, patient_id):
    """Genera datos sintéticos y envía una petición al nodo."""
    data = {
        "patient_id": patient_id,
        "timestamp": time.time(),
        "pleth_wave": [random.uniform(-1, 1) for _ in range(100)],
        "co2_wave": [random.uniform(-1, 1) for _ in range(100)]
    }
    
    start_time = time.time()
    try:
        response = await client.post(URL, json=data)
        latency = (time.time() - start_time) * 1000
        if response.status_code == 200:
            res_json = response.json()
            print(f"[OK] Status: {res_json['status']} | Latency: {latency:.2f}ms | Hemo: {res_json['tier_1']['hemo_risk']:.4f}")
        else:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[EXCEPT] {e}")

async def run_stress_test():
    """Ejecuta ráfagas de peticiones para simular carga."""
    print(f"Iniciando Stress Test contra {URL}...")
    print("Simulando ~100 peticiones por segundo...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_test = time.time()
        
        for i in range(TOTAL_REQUESTS // CONCURRENT_REQUESTS):
            tasks = [send_inference_request(client, f"paciente_{random.randint(1, 10)}") for _ in range(CONCURRENT_REQUESTS)]
            await asyncio.gather(*tasks)
            # Pequeña pausa para no saturar el event loop local y mantener la tasa
            await asyncio.sleep(0.1) 
            
        total_time = time.time() - start_test
        print("\n" + "="*30)
        print(f"Prueba completada.")
        print(f"Total peticiones: {TOTAL_REQUESTS}")
        print(f"Tiempo total: {total_time:.2f}s")
        print(f"Promedio: {TOTAL_REQUESTS/total_time:.2f} req/s")
        print("="*30)

if __name__ == "__main__":
    try:
        asyncio.run(run_stress_test())
    except KeyboardInterrupt:
        pass
