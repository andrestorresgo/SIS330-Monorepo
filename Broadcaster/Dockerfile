# ── Stage 1: Build ──
FROM golang:1.23-alpine AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o broadcaster .

# ── Stage 2: Runtime ──
FROM alpine:3.20

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /build/broadcaster .

# production_sim_data.csv is mounted at runtime via docker-compose volume
EXPOSE 8080

CMD ["./broadcaster"]
