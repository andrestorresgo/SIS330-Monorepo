package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gorilla/websocket"
)

// VitalSign matches the broadcaster's JSON payload.
type VitalSign struct {
	Time  float64 `json:"time"`
	Pleth float64 `json:"pleth"`
	CO2   float64 `json:"co2"`
}

// HealthInfo matches the broadcaster's /debug/health response.
type HealthInfo struct {
	Uptime      string  `json:"uptime"`
	Clients     int     `json:"clients"`
	TickIndex   int     `json:"tick_index"`
	TotalRows   int     `json:"total_rows"`
	LoopCount   int     `json:"loop_count"`
	HeapAllocMB float64 `json:"heap_alloc_mb"`
	SysMemMB    float64 `json:"sys_mem_mb"`
}

func main() {
	wsURL := "ws://localhost:8080/vitals"
	healthURL := "http://localhost:8080/debug/health"

	if len(os.Args) > 1 {
		wsURL = os.Args[1]
	}

	fmt.Println("╔══════════════════════════════════════════════╗")
	fmt.Println("║   Broadcaster Throughput Monitor             ║")
	fmt.Println("╚══════════════════════════════════════════════╝")
	fmt.Printf("  WebSocket: %s\n", wsURL)
	fmt.Printf("  Health:    %s\n", healthURL)
	fmt.Println("  Press Ctrl+C to stop and generate report.\n")

	// Connect to WebSocket
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	fmt.Println("  ✅ Connected!\n")

	// Signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Tracking variables
	var (
		totalMessages   int
		intervals       []float64 // inter-message intervals in ms
		lastReceiveTime time.Time
		startTime       = time.Now()
		secondCounts    []int // messages per second
		currentSecond   int
		currentCount    int
		healthSnapshots []HealthInfo
		done            = make(chan struct{})
		closeOnce       sync.Once
		closeDone       = func() { closeOnce.Do(func() { close(done) }) }
	)

	// Health poller — every 30 seconds
	go func() {
		ticker := time.NewTicker(30 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if h, err := fetchHealth(healthURL); err == nil {
					healthSnapshots = append(healthSnapshots, h)
				}
			case <-done:
				return
			}
		}
	}()

	// Progress printer — every 5 seconds
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				elapsed := time.Since(startTime).Seconds()
				rate := float64(totalMessages) / elapsed
				fmt.Printf("  [%6.0fs] %d messages | avg %.1f msg/s\n", elapsed, totalMessages, rate)
			case <-done:
				return
			}
		}
	}()

	// Message receiver
	go func() {
		for {
			_, msg, err := conn.ReadMessage()
			if err != nil {
				closeDone()
				return
			}

			now := time.Now()
			totalMessages++

			// Track inter-message interval
			if !lastReceiveTime.IsZero() {
				interval := now.Sub(lastReceiveTime).Seconds() * 1000 // ms
				intervals = append(intervals, interval)
			}
			lastReceiveTime = now

			// Track per-second counts
			sec := int(now.Sub(startTime).Seconds())
			if sec != currentSecond {
				if currentSecond > 0 || currentCount > 0 {
					secondCounts = append(secondCounts, currentCount)
				}
				currentSecond = sec
				currentCount = 0
			}
			currentCount++

			// Validate JSON shape (lightweight)
			var v VitalSign
			if err := json.Unmarshal(msg, &v); err != nil {
				log.Printf("Invalid JSON: %s", string(msg))
			}
		}
	}()

	// Wait for signal or read error
	select {
	case <-sigChan:
		closeDone()
	case <-done:
	}
	elapsed := time.Since(startTime)

	// Final health snapshot
	if h, err := fetchHealth(healthURL); err == nil {
		healthSnapshots = append(healthSnapshots, h)
	}

	// Flush last second count
	if currentCount > 0 {
		secondCounts = append(secondCounts, currentCount)
	}

	// Generate report
	generateReport(elapsed, totalMessages, intervals, secondCounts, healthSnapshots)
}

func fetchHealth(url string) (HealthInfo, error) {
	var h HealthInfo
	resp, err := http.Get(url)
	if err != nil {
		return h, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	err = json.Unmarshal(body, &h)
	return h, err
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func generateReport(elapsed time.Duration, totalMessages int, intervals []float64, secondCounts []int, health []HealthInfo) {
	sort.Float64s(intervals)

	var meanInterval, minInterval, maxInterval, p99Interval, stddevInterval float64
	if len(intervals) > 0 {
		sum := 0.0
		for _, v := range intervals {
			sum += v
		}
		meanInterval = sum / float64(len(intervals))
		minInterval = intervals[0]
		maxInterval = intervals[len(intervals)-1]
		p99Interval = percentile(intervals, 99)

		// Stddev
		sqSum := 0.0
		for _, v := range intervals {
			sqSum += (v - meanInterval) * (v - meanInterval)
		}
		stddevInterval = math.Sqrt(sqSum / float64(len(intervals)))
	}

	var meanRate, minRate, maxRate float64
	if len(secondCounts) > 0 {
		sortedCounts := make([]int, len(secondCounts))
		copy(sortedCounts, secondCounts)
		sort.Ints(sortedCounts)

		sum := 0
		for _, v := range sortedCounts {
			sum += v
		}
		meanRate = float64(sum) / float64(len(sortedCounts))
		minRate = float64(sortedCounts[0])
		maxRate = float64(sortedCounts[len(sortedCounts)-1])
	}

	sep := strings.Repeat("═", 55)

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  BROADCASTER THROUGHPUT REPORT")
	fmt.Println(sep)
	fmt.Println()
	fmt.Println("  GENERAL")
	fmt.Printf("    Duration:          %s\n", elapsed.Round(time.Second))
	fmt.Printf("    Total messages:    %d\n", totalMessages)
	fmt.Printf("    Overall rate:      %.2f msg/s\n", float64(totalMessages)/elapsed.Seconds())
	fmt.Println()
	fmt.Println("  THROUGHPUT (per second)")
	fmt.Printf("    Mean:              %.1f msg/s\n", meanRate)
	fmt.Printf("    Min:               %.0f msg/s\n", minRate)
	fmt.Printf("    Max:               %.0f msg/s\n", maxRate)
	fmt.Printf("    Samples:           %d seconds\n", len(secondCounts))
	fmt.Println()
	fmt.Println("  JITTER (inter-message interval)")
	fmt.Printf("    Expected:          10.00 ms\n")
	fmt.Printf("    Mean:              %.3f ms\n", meanInterval)
	fmt.Printf("    Stddev:            %.3f ms\n", stddevInterval)
	fmt.Printf("    Min:               %.3f ms\n", minInterval)
	fmt.Printf("    Max:               %.3f ms\n", maxInterval)
	fmt.Printf("    P99:               %.3f ms\n", p99Interval)
	fmt.Println()

	if len(health) > 0 {
		fmt.Println("  MEMORY (broadcaster process)")
		fmt.Println("    Snapshot            HeapAlloc(MB)   Sys(MB)   Clients   Loops")
		for i, h := range health {
			fmt.Printf("    [%2d] %-14s  %8.2f      %8.2f    %3d      %3d\n",
				i+1, h.Uptime, h.HeapAllocMB, h.SysMemMB, h.Clients, h.LoopCount)
		}

		first := health[0]
		last := health[len(health)-1]
		drift := last.HeapAllocMB - first.HeapAllocMB
		fmt.Printf("\n    Heap drift:        %+.2f MB", drift)
		if math.Abs(drift) < 2 {
			fmt.Println(" ✅ (stable)")
		} else {
			fmt.Println(" ⚠️  (possible leak)")
		}
	} else {
		fmt.Println("  MEMORY: No health snapshots collected (run longer for memory data)")
	}

	fmt.Println()
	fmt.Println(sep)
	fmt.Println("  VERDICT")

	overallRate := float64(totalMessages) / elapsed.Seconds()
	rateOk := math.Abs(overallRate-100) < 3
	jitterOk := p99Interval < 15

	if rateOk {
		fmt.Println("    Rate:    ✅ PASS (within ±3 of 100 Hz)")
	} else {
		fmt.Printf("    Rate:    ❌ FAIL (%.2f Hz, expected ~100 Hz)\n", overallRate)
	}
	if jitterOk {
		fmt.Println("    Jitter:  ✅ PASS (P99 < 15ms)")
	} else {
		fmt.Printf("    Jitter:  ❌ FAIL (P99 = %.3fms, expected < 15ms)\n", p99Interval)
	}

	fmt.Println(sep)
	fmt.Println()
}
