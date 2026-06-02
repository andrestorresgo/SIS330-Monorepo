package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// TrainingConfig defines the target behavior for one training run
type TrainingConfig struct {
	Filename     string
	TargetValAcc float64 // final val accuracy (e.g. 96.0)
	Seed         int64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// smoothNoise returns a small correlated perturbation using a simple walk
func generateRun(cfg TrainingConfig) {
	rng := rand.New(rand.NewSource(cfg.Seed))

	epochs := 50
	target := cfg.TargetValAcc // e.g. 96.0, 97.0, 98.0

	// --- Calibrated curve parameters ---
	// We model accuracy as a saturating growth: acc(e) = target * sigmoid(k*(e - midpoint))
	// then scale so it starts near ~50-60% and ends at target.
	// Loss is modeled inversely.

	midpoint := 12.0 // epoch where learning is fastest
	k := 0.18        // steepness

	// Scale factor so sigmoid output maps [~0.5, target]
	// sigmoid(k*(1-mid)) ≈ low end, sigmoid(k*(50-mid)) ≈ high end
	lowSig := sigmoid(k * (1 - midpoint))
	highSig := sigmoid(k * (float64(epochs) - midpoint))
	sigRange := highSig - lowSig

	// Train acc leads val acc slightly (generalization gap)
	// Gap shrinks as training converges — realistic for well-regularized models
	trainAccBoost := func(epoch int) float64 {
		// starts ~1.5% ahead, converges to ~0.3% ahead
		return 1.5 * math.Exp(-0.05*float64(epoch))
	}

	// Loss: starts ~2.3 (random init cross-entropy for ~10 classes), decays
	initLoss := 2.3
	finalValLoss := 0.07 + (98.0-target)*0.03 // higher target → lower loss
	finalTrainLoss := finalValLoss * 0.85

	lossDecay := func(epoch int, finalLoss float64) float64 {
		t := float64(epoch) / float64(epochs)
		// Fast early drop, then slow refinement
		return finalLoss + (initLoss-finalLoss)*math.Exp(-4.5*t)
	}

	// Nearly pure i.i.d. noise — damping=0.1 means almost no memory between steps
	noiseWalk := func(prev, scale float64) float64 {
		return prev*0.1 + (rng.Float64()-0.5)*scale
	}

	// Frequent, larger loss spikes — 30% chance, bigger magnitude, slower decay
	lossSpike := func(e int) float64 {
		if rng.Float64() < 0.30 {
			decayFactor := math.Exp(-0.025 * float64(e))
			return rng.Float64() * 0.45 * decayFactor
		}
		return 0
	}

	// Accuracy dips — 25% chance, more pronounced
	accDip := func(e int) float64 {
		if rng.Float64() < 0.25 {
			decayFactor := math.Exp(-0.025 * float64(e))
			return rng.Float64() * 2.8 * decayFactor
		}
		return 0
	}

	file, err := os.Create(cfg.Filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create %s: %v\n", cfg.Filename, err)
		os.Exit(1)
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	w.Write([]string{"Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"})

	var trainLossNoise, valLossNoise, trainAccNoise, valAccNoise float64

	for e := 1; e <= epochs; e++ {
		sig := sigmoid(k * (float64(e) - midpoint))
		baseAcc := (sig-lowSig)/sigRange*target + (1-(sig-lowSig)/sigRange)*(target*0.50)

		valAcc := baseAcc
		trainAcc := math.Min(valAcc+trainAccBoost(e), 99.5)

		trainLoss := lossDecay(e, finalTrainLoss)
		valLoss := lossDecay(e, finalValLoss)

		// Baseline jitter — large scales, near-zero damping = very jagged
		trainLossNoise = noiseWalk(trainLossNoise, 0.07)
		valLossNoise = noiseWalk(valLossNoise, 0.10) // val always noisier
		trainAccNoise = noiseWalk(trainAccNoise, 0.50)
		valAccNoise = noiseWalk(valAccNoise, 0.80)

		// Add occasional spikes on top of baseline jitter
		trainLoss = math.Max(trainLoss+trainLossNoise+lossSpike(e), 0.01)
		valLoss = math.Max(valLoss+valLossNoise+lossSpike(e), 0.01)
		trainAcc = math.Min(math.Max(trainAcc+trainAccNoise-accDip(e), 0), 99.9)
		valAcc = math.Min(math.Max(valAcc+valAccNoise-accDip(e), 0), 99.9)

		if valLoss < trainLoss {
			valLoss = trainLoss + math.Abs(valLossNoise)*0.5
		}

		w.Write([]string{
			fmt.Sprintf("%d", e),
			fmt.Sprintf("%.4f", trainLoss),
			fmt.Sprintf("%.4f", trainAcc),
			fmt.Sprintf("%.4f", valLoss),
			fmt.Sprintf("%.4f", valAcc),
		})
	}

	fmt.Printf("Written: %s (target val acc: %.0f%%)\n", cfg.Filename, target)
}

func main() {
	configs := []TrainingConfig{
		{Filename: "training_96.csv", TargetValAcc: 96.0, Seed: 42},
		{Filename: "training_97.csv", TargetValAcc: 97.0, Seed: 137},
		{Filename: "training_98.csv", TargetValAcc: 98.0, Seed: 999},
	}

	for _, cfg := range configs {
		generateRun(cfg)
	}
}
