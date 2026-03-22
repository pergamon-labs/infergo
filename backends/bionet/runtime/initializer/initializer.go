package initializer

import (
	"math"
	"math/rand/v2"

	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
)

type Initializer func(t *tensor.Tensor)

func fanIn(shape []int) int {
	if len(shape) == 0 {
		return 0
	}

	if len(shape) == 1 {
		return shape[0]
	}

	fan := shape[1]
	for _, dim := range shape[2:] {
		fan *= dim
	}

	return fan
}

// KaimingUniform initializes weights using Kaiming uniform initialization.
// This method is particularly well-suited for layers with ReLU activation.
// It initializes weights to a uniform distribution within the range [-bound, bound],
// where bound is sqrt(3 / fanIn).  The tensor is modified in-place.
func KaimingUniform(t *tensor.Tensor) {
	if t == nil || len(t.Shape()) == 0 {
		return
	}

	fanIn := fanIn(t.Shape())
	if fanIn <= 0 {
		return
	}

	bound := math.Sqrt(6.0 / float64(fanIn))

	for i := range t.Values() {
		randomValue := rand.Float64()*2*bound - bound
		t.SetFlatValue(i, randomValue)
	}
}

// KaimingNormal initializes weights using Kaiming normal initialization.
// This method is particularly well-suited for layers with ReLU activation.
// It initializes weights to a normal distribution with standard deviation std,
// where std is sqrt(1 / fanIn).  The tensor is modified in-place.
func KaimingNormal(t *tensor.Tensor) {
	if t == nil || len(t.Shape()) == 0 {
		return
	}

	fanIn := fanIn(t.Shape())
	if fanIn <= 0 {
		return
	}

	std := math.Sqrt(2.0 / float64(fanIn))

	for i := range t.Values() {
		t.SetFlatValue(i, rand.NormFloat64()*std)
	}
}
