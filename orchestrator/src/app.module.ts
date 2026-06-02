import { Module } from '@nestjs/common';
import { EventEmitterModule } from '@nestjs/event-emitter';
import { IngestionModule } from './ingestion/ingestion.module.js';
import { BufferModule } from './buffer/buffer.module.js';
import { AiGatewayModule } from './ai-gateway/ai-gateway.module.js';
import { DashboardModule } from './dashboard/dashboard.module.js';

@Module({
  imports: [
    EventEmitterModule.forRoot(),
    IngestionModule,
    BufferModule,
    AiGatewayModule,
    DashboardModule,
  ],
})
export class AppModule {}
