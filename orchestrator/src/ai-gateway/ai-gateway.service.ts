import { Injectable, Logger } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { OnEvent } from '@nestjs/event-emitter';
import { EventEmitter2 } from '@nestjs/event-emitter';
import { firstValueFrom } from 'rxjs';
import { MockAiService } from './mock-ai.service.js';
import type {
  VitalChunk,
  AiPrediction,
} from '../ingestion/ingestion.interfaces.js';

@Injectable()
export class AiGatewayService {
  private readonly logger = new Logger(AiGatewayService.name);
  private readonly mlEndpoint: string | undefined;

  constructor(
    private readonly httpService: HttpService,
    private readonly mockAi: MockAiService,
    private readonly eventEmitter: EventEmitter2,
  ) {
    this.mlEndpoint = process.env.ML_ENDPOINT;
    if (this.mlEndpoint) {
      this.logger.log(`ML endpoint configured: ${this.mlEndpoint}`);
    } else {
      this.logger.warn('No ML_ENDPOINT set — using AiService for predictions');
    }
  }

  /**
   * Fires asynchronously on each 100-point chunk.
   * Does NOT block the ingestion/buffer pipeline.
   */
  @OnEvent('vital.chunk', { async: true })
  async handleChunk(chunk: VitalChunk) {
    try {
      let prediction: AiPrediction;

      if (this.mlEndpoint) {
        // ── Real ML inference via HTTP POST ──
        const response = await firstValueFrom(
          this.httpService.post<AiPrediction>(this.mlEndpoint, chunk),
        );
        prediction = response.data;
      } else {
        // ── Mock inference ──
        prediction = await this.mockAi.predict(chunk);
      }

      // Broadcast prediction to downstream consumers (dashboard)
      this.eventEmitter.emit('vital.prediction', prediction);
    } catch (err) {
      this.logger.error(`AI inference failed: ${err}`);
    }
  }
}
