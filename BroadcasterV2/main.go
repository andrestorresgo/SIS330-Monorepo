package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"
)

// Define the payload expected by your FastAPI server
type WaveInput struct {
	WaveData []float64 `json:"wave_data"`
	Horizon  int       `json:"horizon"`
}

// Define the response from your FastAPI server
type PredictionResponse struct {
	Status        string    `json:"status"`
	Horizon       int       `json:"horizon"`
	PredictedWave []float64 `json:"predicted_wave"`
}

const (
	apiURL     = "http://127.0.0.1:8000/predict"
	windowSize = 256 // Number of past data points to send as context
	horizon    = 64  // How many points into the future to predict
)

func main() {
	file, err := os.Open("pleth_wave.csv")
	if err != nil {
		log.Fatalf("Failed to open CSV: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, _ = reader.Read()

	var fullWave []float64
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		val, _ := strconv.ParseFloat(record[0], 64)
		if val != val {
			val = 0.0
		}
		fullWave = append(fullWave, val)
	}

	fmt.Printf("Loaded %d points from VitalDB.\n", len(fullWave))
	fmt.Println("Starting real-time broadcast simulation...")

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	currentIndex := windowSize
	client := &http.Client{Timeout: 2 * time.Second}

	for range ticker.C {
		if currentIndex >= len(fullWave) {
			fmt.Println("End of wave data reached.")
			break
		}

		currentWindow := fullWave[currentIndex-windowSize : currentIndex]

		go fetchPrediction(client, currentWindow, currentIndex)

		currentIndex += 50
	}
}

func fetchPrediction(client *http.Client, window []float64, index int) {
	payload := WaveInput{
		WaveData: window,
		Horizon:  horizon,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		log.Printf("JSON marshal error: %v", err)
		return
	}

	resp, err := client.Post(apiURL, "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Failed to reach API: %v", err)
		return
	}
	defer resp.Body.Close()

	// Parse the response
	var result PredictionResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		log.Printf("Failed to decode response: %v", err)
		return
	}

	// Print a clean readout of what the model thinks is coming next
	// fmt.Printf("Broadcast at point %d -> Predicted next %d points. (First val: %.3f, Last val: %.3f)\n",
	// 	index, result.Horizon, result.PredictedWave[0], result.PredictedWave[len(result.PredictedWave)-1])

	// Print in this format, as a mock, map result.PredictedWave[len(result.PredictedWave)-1 to hemo risk between 0 and 1:
	/*
			{
		  "patient_id": "paciente_123",
		  "status": "MONITORIZANDO",
		  "tier_1": {
		    "hemo_risk": 0.12,
		    "vent_risk": 0.05
		  },
		  "tier_2": {
		    "system_risk": 0.08,
		    "alarm_suppressed": false
		  }
		}
	*/

	systemRisk := result.PredictedWave[len(result.PredictedWave)-1] * 0.01

	response := map[string]interface{}{
		"patient_id": "paciente_123",
		"status":     "MONITORIZANDO",
		"tier_2": map[string]interface{}{
			"system_risk":      systemRisk,
			"alarm_suppressed": false,
		},
	}

	jsonResponse, err := json.Marshal(response)
	if err != nil {
		log.Printf("JSON marshal error: %v", err)
		return
	}

	fmt.Println(string(jsonResponse))
}
