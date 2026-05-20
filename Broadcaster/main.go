package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/websocket"
)

// ──────────────────────────────────────────────
// Data types
// ──────────────────────────────────────────────

// VitalSign represents one row from production_sim_data.csv.
type VitalSign struct {
	Time  float64 `json:"time"`
	Pleth float64 `json:"pleth"`
	CO2   float64 `json:"co2"`
}

// HealthInfo is returned by the /debug/health endpoint.
type HealthInfo struct {
	Uptime      string  `json:"uptime"`
	Clients     int     `json:"clients"`
	TickIndex   int     `json:"tick_index"`
	TotalRows   int     `json:"total_rows"`
	LoopCount   int     `json:"loop_count"`
	HeapAllocMB float64 `json:"heap_alloc_mb"`
	SysMemMB    float64 `json:"sys_mem_mb"`
}

// ──────────────────────────────────────────────
// CSV Loader
// ──────────────────────────────────────────────

func loadCSV(path string) ([]VitalSign, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)

	// Skip header
	if _, err := reader.Read(); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	var data []VitalSign
	for {
		record, err := reader.Read()
		if err != nil {
			break // EOF or error
		}
		if len(record) < 3 {
			continue
		}

		t, _ := strconv.ParseFloat(record[0], 64)
		pleth, _ := strconv.ParseFloat(record[1], 64)
		co2, _ := strconv.ParseFloat(record[2], 64)

		data = append(data, VitalSign{Time: t, Pleth: pleth, CO2: co2})
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("csv is empty")
	}

	return data, nil
}

// ──────────────────────────────────────────────
// Hub — manages WebSocket clients & fan-out
// ──────────────────────────────────────────────

type Client struct {
	conn *websocket.Conn
	send chan []byte
}

type Hub struct {
	mu         sync.RWMutex
	clients    map[*Client]bool
	register   chan *Client
	unregister chan *Client
	broadcast  chan []byte
}

func newHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		register:   make(chan *Client),
		unregister: make(chan *Client),
		broadcast:  make(chan []byte, 256),
	}
}

func (h *Hub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("[HUB] Client connected (%d total)", h.clientCount())

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
			log.Printf("[HUB] Client disconnected (%d total)", h.clientCount())

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					// Client too slow, evict
					go func(c *Client) {
						h.unregister <- c
					}(client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

func (h *Hub) clientCount() int {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return len(h.clients)
}

// ──────────────────────────────────────────────
// Client write pump
// ──────────────────────────────────────────────

func (c *Client) writePump() {
	defer c.conn.Close()
	for msg := range c.send {
		if err := c.conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			return
		}
	}
}

// ──────────────────────────────────────────────
// 100Hz Ticker
// ──────────────────────────────────────────────

type Broadcaster struct {
	hub       *Hub
	data      []VitalSign
	index     int
	loopCount int
	startTime time.Time
}

func newBroadcaster(hub *Hub, data []VitalSign) *Broadcaster {
	return &Broadcaster{
		hub:       hub,
		data:      data,
		startTime: time.Now(),
	}
}

func (b *Broadcaster) run() {
	ticker := time.NewTicker(10 * time.Millisecond) // 100 Hz
	defer ticker.Stop()

	log.Printf("[TICKER] Started 100Hz broadcast (%d rows, infinite loop)", len(b.data))

	for range ticker.C {
		// Read current row
		vital := b.data[b.index]

		// Marshal to JSON
		payload, err := json.Marshal(vital)
		if err != nil {
			log.Printf("[TICKER] Marshal error: %v", err)
			continue
		}

		// Broadcast to all connected clients
		b.hub.broadcast <- payload

		// Advance index with wrap-around
		b.index++
		if b.index >= len(b.data) {
			b.index = 0
			b.loopCount++
			log.Printf("[TICKER] CSV loop #%d complete, rewinding to row 0", b.loopCount)
		}
	}
}

// ──────────────────────────────────────────────
// WebSocket handler
// ──────────────────────────────────────────────

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin:     func(r *http.Request) bool { return true },
}

func handleVitals(hub *Hub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Printf("[WS] Upgrade error: %v", err)
			return
		}

		client := &Client{
			conn: conn,
			send: make(chan []byte, 512),
		}

		hub.register <- client

		// Write pump in goroutine
		go client.writePump()

		// Read pump — just drain to detect disconnects
		for {
			if _, _, err := conn.ReadMessage(); err != nil {
				hub.unregister <- client
				break
			}
		}
	}
}

// ──────────────────────────────────────────────
// Health endpoint
// ──────────────────────────────────────────────

func handleHealth(hub *Hub, bc *Broadcaster) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		info := HealthInfo{
			Uptime:      time.Since(bc.startTime).Round(time.Second).String(),
			Clients:     hub.clientCount(),
			TickIndex:   bc.index,
			TotalRows:   len(bc.data),
			LoopCount:   bc.loopCount,
			HeapAllocMB: float64(m.HeapAlloc) / 1024 / 1024,
			SysMemMB:    float64(m.Sys) / 1024 / 1024,
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(info)
	}
}

// ──────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────

func main() {
	csvPath := "production_sim_data.csv"
	if len(os.Args) > 1 {
		csvPath = os.Args[1]
	}

	log.Println("[INIT] Starting Broadcaster")

	// Load CSV
	log.Printf("[INIT] Loading CSV: %s", csvPath)
	data, err := loadCSV(csvPath)
	if err != nil {
		log.Fatalf("[INIT] Failed to load CSV: %v", err)
	}
	log.Printf("[INIT] Loaded %d rows (%.1f min at 100Hz)", len(data), float64(len(data))/100/60)

	// Start hub
	hub := newHub()
	go hub.run()

	// Start 100Hz broadcaster
	bc := newBroadcaster(hub, data)
	go bc.run()

	// HTTP routes
	http.HandleFunc("/vitals", handleVitals(hub))
	http.HandleFunc("/debug/health", handleHealth(hub, bc))

	addr := ":8080"
	log.Printf("[SERVER] WebSocket endpoint: ws://localhost%s/vitals", addr)
	log.Printf("[SERVER] Health endpoint:    http://localhost%s/debug/health", addr)
	log.Printf("[SERVER] Waiting for connections...")

	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("[SERVER] Fatal: %v", err)
	}
}
