import { Injectable, Logger } from '@nestjs/common';
import type {
  AiPrediction,
  VitalChunk,
} from '../ingestion/ingestion.interfaces.js';

@Injectable()
export class MockAiService {
  private readonly logger = new Logger(MockAiService.name);

  /**
   * Simulate ML inference with a 50ms delay.
   * Returns a fake hypotension prediction.
   */
  async predict(chunk: VitalChunk): Promise<AiPrediction> {
    await new Promise((resolve) => setTimeout(resolve, 50));

    const prediction: AiPrediction = {
      hypotension_risk: +(Math.random() * 0.3).toFixed(4),
      state: Math.random() > 0.85 ? 'warning' : 'stable',
      timestamp: chunk.timestamp,
    };

    this.logger.log(
      `🤖 Mock prediction: risk=${prediction.hypotension_risk}, state=${prediction.state}`,
    );

    return prediction;
  }
}
