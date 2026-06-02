import {
  Injectable,
  Logger,
  OnModuleInit,
  OnModuleDestroy,
} from '@nestjs/common';
import { EventEmitter2 } from '@nestjs/event-emitter';
import WebSocket from 'ws';
import type { VitalPayload } from './ingestion.interfaces.js';

@Injectable()
export class IngestionService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(IngestionService.name);
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private readonly maxReconnectDelay = 30_000; // 30s cap
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private isShuttingDown = false;

  // Throughput tracking
  private messageCount = 0;
  private throughputInterval: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly eventEmitter: EventEmitter2) {}

  onModuleInit() {
    const host = process.env.GO_WS_HOST ?? 'localhost';
    const port = process.env.GO_WS_PORT ?? '8080';
    this.logger.log(`Targeting Go broadcaster at ws://${host}:${port}/vitals`);
    this.connect(host, port);
    this.startThroughputLogger();
  }

  onModuleDestroy() {
    this.isShuttingDown = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.throughputInterval) clearInterval(this.throughputInterval);
    if (this.ws) this.ws.close();
  }

  private connect(host: string, port: string) {
    const url = `ws://${host}:${port}/vitals`;

    try {
      this.ws = new WebSocket(url);
    } catch (err) {
      this.logger.error(`Failed to create WebSocket: ${err}`);
      this.scheduleReconnect(host, port);
      return;
    }

    this.ws.on('open', () => {
      this.reconnectAttempts = 0;
      this.logger.log(`✅ Connected to Go broadcaster at ${url}`);
    });

    this.ws.on('message', (data: WebSocket.Data) => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-base-to-string
        const payload: VitalPayload = JSON.parse(data.toString());
        this.messageCount++;
        this.eventEmitter.emit('vital.raw', payload);
      } catch (err) {
        this.logger.warn(`Failed to parse payload: ${err}`);
      }
    });

    this.ws.on('close', (code, reason) => {
      if (this.isShuttingDown) return;
      this.logger.warn(
        `Connection closed (code=${code}, reason=${reason.toString()}). Reconnecting...`,
      );
      this.scheduleReconnect(host, port);
    });

    this.ws.on('error', (err) => {
      // 'close' event will fire after this, triggering reconnect
      this.logger.error(`WebSocket error: ${err.message}`);
    });
  }

  private scheduleReconnect(host: string, port: string) {
    if (this.isShuttingDown) return;

    const delay = Math.min(
      1000 * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay,
    );
    this.reconnectAttempts++;
    this.logger.log(
      `Reconnecting in ${delay}ms (attempt #${this.reconnectAttempts})...`,
    );

    this.reconnectTimer = setTimeout(() => {
      this.connect(host, port);
    }, delay);
  }

  private startThroughputLogger() {
    this.throughputInterval = setInterval(() => {
      if (this.messageCount > 0) {
        this.logger.debug(`Throughput: ${this.messageCount} msg/s`);
        this.messageCount = 0;
      }
    }, 1000);
  }
}
