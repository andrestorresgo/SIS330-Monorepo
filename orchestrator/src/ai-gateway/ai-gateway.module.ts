import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { AiGatewayService } from './ai-gateway.service.js';
import { MockAiService } from './mock-ai.service.js';

@Module({
  imports: [HttpModule],
  providers: [AiGatewayService, MockAiService],
  exports: [AiGatewayService],
})
export class AiGatewayModule {}
