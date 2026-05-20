/**
 * Represents one vital-sign payload from the Go broadcaster.
 * Arrives at 100Hz over WebSocket.
 */
export interface VitalPayload {
  time: number;
  pleth: number;
  co2: number;
}

/**
 * A 1-second buffered chunk (100 data points at 100Hz).
 * Sent to the AI gateway for inference.
 */
export interface VitalChunk {
  pleth_window: number[];
  co2_window: number[];
  timestamp: string;
}

/**
 * AI prediction result returned by the ML engine (or mock).
 */
export interface AiPrediction {
  hypotension_risk: number;
  state: string;
  timestamp: string;
}
