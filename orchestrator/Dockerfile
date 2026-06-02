# ── Stage 1: Build ──
FROM node:22-alpine AS builder

WORKDIR /build

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ── Stage 2: Runtime ──
FROM node:22-alpine

WORKDIR /app

# Copy only production dependencies
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Copy compiled JavaScript from builder
COPY --from=builder /build/dist ./dist

EXPOSE 3000

CMD ["node", "dist/main.js"]
