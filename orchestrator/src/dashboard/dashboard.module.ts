import { Module } from '@nestjs/common';
import { DashboardGateway } from './dashboard.gateway.js';

@Module({
  providers: [DashboardGateway],
  exports: [DashboardGateway],
})
export class DashboardModule {}
