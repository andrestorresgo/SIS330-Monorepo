import { Injectable, Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import { EventEmitter2 } from '@nestjs/event-emitter';
import type {
  VitalPayload,
  VitalChunk,
} from '../ingestion/ingestion.interfaces.js';

@Injectable()
export class BufferService {
  private readonly logger = new Logger(BufferService.name);

  private plethBuffer: number[] = [];
  private co2Buffer: number[] = [];

  private chunkCount = 0;

  constructor(private readonly eventEmitter: EventEmitter2) {}

  @OnEvent('vital.raw')
  handleVitalRaw(payload: VitalPayload) {
    this.plethBuffer.push(payload.pleth);
    this.co2Buffer.push(payload.co2);

    if (this.plethBuffer.length >= 100) {
      // ── 100-Point Lock ──
      // Snapshot and clear in one synchronous tick — the Node.js event loop
      // guarantees no interleaving, so the WS ingestion thread is never blocked.
      const chunk: VitalChunk = {
        pleth_window: [...this.plethBuffer],
        co2_window: [...this.co2Buffer],
        timestamp: new Date().toISOString(),
      };

      // Reset buffers
      this.plethBuffer = [];
      this.co2Buffer = [];

      this.chunkCount++;
      this.logger.log(
        `📦 Chunk #${this.chunkCount} assembled — ` +
          `pleth[${chunk.pleth_window.length}] co2[${chunk.co2_window.length}]`,
      );

      // Fire chunk event for downstream consumers (AI gateway, etc.)
      this.eventEmitter.emit('vital.chunk', chunk);
    }
  }
}
