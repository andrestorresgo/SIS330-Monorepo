import { Logger } from '@nestjs/common';
import { OnEvent } from '@nestjs/event-emitter';
import {
  WebSocketGateway,
  WebSocketServer,
  OnGatewayConnection,
  OnGatewayDisconnect,
} from '@nestjs/websockets';
import { Server, Socket } from 'socket.io';
import type {
  VitalPayload,
  AiPrediction,
} from '../ingestion/ingestion.interfaces.js';

@WebSocketGateway({
  cors: {
    origin: '*',
  },
})
export class DashboardGateway
  implements OnGatewayConnection, OnGatewayDisconnect
{
  private readonly logger = new Logger(DashboardGateway.name);

  @WebSocketServer()
  server!: Server;

  private clientCount = 0;

  handleConnection(client: Socket) {
    this.clientCount++;
    this.logger.log(
      `🖥️  Dashboard client connected (id=${client.id}, total=${this.clientCount})`,
    );
  }

  handleDisconnect(client: Socket) {
    this.clientCount--;
    this.logger.log(
      `🖥️  Dashboard client disconnected (id=${client.id}, total=${this.clientCount})`,
    );
  }

  // ── Raw 100Hz pass-through ──
  @OnEvent('vital.raw')
  handleRawVital(payload: VitalPayload) {
    this.server?.emit('vital:raw', payload);
  }

  // ── AI prediction (once per second) ──
  @OnEvent('vital.prediction')
  handlePrediction(prediction: AiPrediction) {
    this.server?.emit('vital:prediction', prediction);
    this.logger.log(
      `📡 Broadcast prediction to ${this.clientCount} client(s): risk=${prediction.hypotension_risk}`,
    );
  }
}
