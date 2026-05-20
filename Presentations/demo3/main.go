package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// TransformerConfig defines one training run for a binary classifier
type TransformerConfig struct {
	Filename     string
	TargetValAcc float64 // final val accuracy in [0,1], e.g. 0.96
	TargetValAUC float64 // final AUC-ROC, e.g. 0.97
	TargetSens   float64 // final sensitivity (crash recall)
	TargetSpec   float64 // final specificity (safe recall)
	Seed         int64
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// saturate models a metric that rises from `start` to `end` over `epochs`
// using a sigmoid with given midpoint and steepness.
func saturate(epoch, epochs int, start, end, midpoint, k float64) float64 {
	lowSig := sigmoid(k * (1 - midpoint))
	highSig := sigmoid(k * (float64(epochs) - midpoint))
	sigRange := highSig - lowSig
	sig := sigmoid(k * (float64(epoch) - midpoint))
	t := (sig - lowSig) / sigRange
	return start + t*(end-start)
}

func generateTransformerRun(cfg TransformerConfig) {
	rng := rand.New(rand.NewSource(cfg.Seed))

	epochs := 50

	// Shared noise primitive — near-zero damping for jagged curves
	noiseWalk := func(prev, scale float64) float64 {
		return prev*0.1 + (rng.Float64()-0.5)*scale
	}

	// Loss spike: 30% chance, decays in magnitude as training stabilises
	lossSpike := func(e int) float64 {
		if rng.Float64() < 0.30 {
			return rng.Float64() * 0.45 * math.Exp(-0.025*float64(e))
		}
		return 0
	}

	// Metric dip: 25% chance, decays over time
	metricDip := func(e int, mag float64) float64 {
		if rng.Float64() < 0.25 {
			return rng.Float64() * mag * math.Exp(-0.025*float64(e))
		}
		return 0
	}

	// -------------------------------------------------------------------------
	// Loss parameters
	// Binary cross-entropy random init ≈ ln(2) ≈ 0.693 for balanced classes.
	// Train loss ends slightly lower than val loss (no overfitting, well-regularized).
	initLoss := 0.693
	finalValLoss := 0.08 + (0.98-cfg.TargetValACC())*0.4
	finalTrainLoss := finalValLoss * 0.85

	lossDecay := func(e int, finalLoss float64) float64 {
		t := float64(e) / float64(epochs)
		return finalLoss + (initLoss-finalLoss)*math.Exp(-4.5*t)
	}

	// -------------------------------------------------------------------------
	// Accuracy: [0,1], train leads val by a shrinking gap
	trainAccBoost := func(e int) float64 {
		return 0.015 * math.Exp(-0.05*float64(e))
	}

	// -------------------------------------------------------------------------
	// Sens/Spec tradeoff:
	// Early training: model is biased — sensitivity shoots up first (predicting
	// "crash" aggressively) while specificity lags. As the model refines its
	// decision boundary both converge toward their targets.
	// This models the classic precision-recall tension students need to see.
	initSens := 0.55 // model starts recall-heavy (biased toward positive class)
	initSpec := 0.40 // specificity starts low as a result

	// -------------------------------------------------------------------------
	// AUC: starts around 0.70 (better than random, worse than good), rises steadily.
	// AUC is smoother than acc/loss because it's threshold-independent.
	initAUC := 0.68

	// -------------------------------------------------------------------------
	file, err := os.Create(cfg.Filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot create %s: %v\n", cfg.Filename, err)
		os.Exit(1)
	}
	defer file.Close()

	w := csv.NewWriter(file)
	defer w.Flush()

	w.Write([]string{
		"Epoch",
		"Train Loss", "Train Acc",
		"Val Loss", "Val Acc",
		"Val AUC",
		"Val Sens", "Val Spec",
	})

	var tlNoise, vlNoise, taNoise, vaNoise float64
	var aucNoise, sensNoise, specNoise float64

	for e := 1; e <= epochs; e++ {
		// --- Loss ---
		trainLoss := lossDecay(e, finalTrainLoss)
		valLoss := lossDecay(e, finalValLoss)

		tlNoise = noiseWalk(tlNoise, 0.07)
		vlNoise = noiseWalk(vlNoise, 0.10)

		trainLoss = math.Max(trainLoss+tlNoise+lossSpike(e), 0.01)
		valLoss = math.Max(valLoss+vlNoise+lossSpike(e), 0.01)
		if valLoss < trainLoss {
			valLoss = trainLoss + math.Abs(vlNoise)*0.5
		}

		// --- Accuracy [0,1] ---
		baseAcc := saturate(e, epochs, 0.50, cfg.TargetValAcc, 12.0, 0.18)
		valAcc := baseAcc
		trainAcc := math.Min(valAcc+trainAccBoost(e), 0.999)

		taNoise = noiseWalk(taNoise, 0.005)
		vaNoise = noiseWalk(vaNoise, 0.008)

		trainAcc = math.Min(math.Max(trainAcc+taNoise-metricDip(e, 0.028), 0), 0.999)
		valAcc = math.Min(math.Max(valAcc+vaNoise-metricDip(e, 0.028), 0), 0.999)

		// --- AUC — smoother, starts at initAUC ---
		baseAUC := saturate(e, epochs, initAUC, cfg.TargetValAUC, 14.0, 0.16)
		aucNoise = noiseWalk(aucNoise, 0.006)
		valAUC := math.Min(math.Max(baseAUC+aucNoise-metricDip(e, 0.015), 0), 1.0)

		// --- Sensitivity ---
		// Starts high (model over-predicts positives), then stabilises at target.
		// Uses a gentler curve so the early-high / late-stable shape is visible.
		baseSens := saturate(e, epochs, initSens, cfg.TargetSens, 10.0, 0.20)
		sensNoise = noiseWalk(sensNoise, 0.008)
		valSens := math.Min(math.Max(baseSens+sensNoise-metricDip(e, 0.020), 0), 1.0)

		// --- Specificity ---
		// Starts low (false alarms are high), rises as boundary refines.
		// Lags sensitivity by design — crosses over around epoch 15-20.
		baseSpec := saturate(e, epochs, initSpec, cfg.TargetSpec, 18.0, 0.20)
		specNoise = noiseWalk(specNoise, 0.009)
		valSpec := math.Min(math.Max(baseSpec+specNoise-metricDip(e, 0.022), 0), 1.0)

		w.Write([]string{
			fmt.Sprintf("%d", e),
			fmt.Sprintf("%.4f", trainLoss),
			fmt.Sprintf("%.4f", trainAcc),
			fmt.Sprintf("%.4f", valLoss),
			fmt.Sprintf("%.4f", valAcc),
			fmt.Sprintf("%.4f", valAUC),
			fmt.Sprintf("%.4f", valSens),
			fmt.Sprintf("%.4f", valSpec),
		})
	}

	fmt.Printf("Written: %s (acc=%.2f AUC=%.2f sens=%.2f spec=%.2f)\n",
		cfg.Filename, cfg.TargetValAcc, cfg.TargetValAUC, cfg.TargetSens, cfg.TargetSpec)
}

// TargetValACC is a helper so lossDecay can reference it without a pointer receiver
func (c TransformerConfig) TargetValACC() float64 { return c.TargetValAcc }

func main() {
	cfg := TransformerConfig{
		Filename:     "transformer_training.csv",
		TargetValAcc: 0.97,
		TargetValAUC: 0.975,
		TargetSens:   0.95,
		TargetSpec:   0.96,
		Seed:         137,
	}
	generateTransformerRun(cfg)
}
