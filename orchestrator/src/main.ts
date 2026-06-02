import { NestFactory } from '@nestjs/core';
import { Logger } from '@nestjs/common';
import { AppModule } from './app.module.js';

async function bootstrap() {
  const logger = new Logger('Bootstrap');

  logger.log('╔══════════════════════════════════════════════╗');
  logger.log('║   Anesthesia AI — NestJS Orchestrator        ║');
  logger.log('╚══════════════════════════════════════════════╝');

  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: '*',
  });

  const port = process.env.PORT ?? 3000;
  await app.listen(port);

  logger.log(`🚀 Orchestrator listening on port ${port}`);
  logger.log(`📡 Dashboard WS: ws://localhost:${port}`);
}
bootstrap();
