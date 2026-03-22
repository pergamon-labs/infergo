package initializer

import (
	"math"
	"testing"

	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
	"github.com/stretchr/testify/assert"
)

func TestKaimingUniform(t *testing.T) {
	// Test case 1: 2D tensor
	t.Run("2D tensor", func(t *testing.T) {
		dims := []int{64, 128}
		tensor := tensor.Zeros(dims)
		KaimingUniform(&tensor)

		assert.Equal(t, dims, tensor.Shape(), "Shape should remain unchanged")
		assert.NotEqual(t, tensor.Values(), make([]float64, len(tensor.Values())), "Values should be initialized")

		// Check if values are within the expected range
		bound := math.Sqrt(float64(6.0 / float64(fanIn(dims))))
		for _, v := range tensor.Values() {
			assert.GreaterOrEqual(t, v, -bound, "Value should be greater than or equal to -bound")
			assert.LessOrEqual(t, v, bound, "Value should be less than or equal to bound")
		}
	})

	// Test case 2: 1D tensor
	t.Run("1D tensor", func(t *testing.T) {
		dims := []int{512}
		tensor := tensor.Zeros(dims)
		KaimingUniform(&tensor)

		assert.Equal(t, dims, tensor.Shape(), "Shape should remain unchanged")
		assert.NotEqual(t, tensor.Values(), make([]float64, len(tensor.Values())), "Values should be initialized")

		// Check if values are within the expected range
		bound := math.Sqrt(float64(6.0 / float64(fanIn(dims))))
		for _, v := range tensor.Values() {
			assert.GreaterOrEqual(t, v, -bound, "Value should be greater than or equal to -bound")
			assert.LessOrEqual(t, v, bound, "Value should be less than or equal to bound")
		}
	})

	// Test case 3: Empty tensor
	t.Run("Empty tensor", func(t *testing.T) {
		tensor := tensor.Zeros([]int{})
		KaimingUniform(&tensor)

		assert.Empty(t, tensor.Shape(), "Shape should remain empty")
		assert.Empty(t, tensor.Values(), "Values should remain empty")
	})

	t.Run("Nil tensor", func(t *testing.T) {
		var tensor *tensor.Tensor
		KaimingUniform(tensor)
		assert.Nil(t, tensor, "Tensor should remain nil")
	})
}

func TestKaimingNormal(t *testing.T) {
	// Test case 1: 2D tensor
	t.Run("2D tensor", func(t *testing.T) {
		dims := []int{64, 128}
		tensor := tensor.Zeros(dims)
		KaimingNormal(&tensor)

		assert.Equal(t, dims, tensor.Shape(), "Shape should remain unchanged")
		assert.NotEqual(t, tensor.Values(), make([]float64, len(tensor.Values())), "Values should be initialized")

		expectedStd := math.Sqrt(2.0 / float64(fanIn(dims)))
		var sum float64
		for _, v := range tensor.Values() {
			assert.True(t, !math.IsNaN(v) && !math.IsInf(v, 0), "Value should be finite")
			sum += v
		}
		mean := sum / float64(len(tensor.Values()))

		var sqDiff float64
		for _, v := range tensor.Values() {
			sqDiff += (v - mean) * (v - mean)
		}
		sampleStd := math.Sqrt(sqDiff / float64(len(tensor.Values())))

		assert.InDelta(t, 0.0, mean, expectedStd*0.2, "Sample mean should remain close to zero")
		assert.InDelta(t, expectedStd, sampleStd, expectedStd*0.25, "Sample std should remain close to expected Kaiming std")
	})

	// Test case 2: 1D tensor
	t.Run("1D tensor", func(t *testing.T) {
		dims := []int{512}
		tensor := tensor.Zeros(dims)
		KaimingNormal(&tensor)

		assert.Equal(t, dims, tensor.Shape(), "Shape should remain unchanged")
		assert.NotEqual(t, tensor.Values(), make([]float64, len(tensor.Values())), "Values should be initialized")

		expectedStd := math.Sqrt(2.0 / float64(fanIn(dims)))
		var sum float64
		for _, v := range tensor.Values() {
			assert.True(t, !math.IsNaN(v) && !math.IsInf(v, 0), "Value should be finite")
			sum += v
		}
		mean := sum / float64(len(tensor.Values()))

		var sqDiff float64
		for _, v := range tensor.Values() {
			sqDiff += (v - mean) * (v - mean)
		}
		sampleStd := math.Sqrt(sqDiff / float64(len(tensor.Values())))

		assert.InDelta(t, 0.0, mean, expectedStd*0.3, "Sample mean should remain close to zero")
		assert.InDelta(t, expectedStd, sampleStd, expectedStd*0.3, "Sample std should remain close to expected Kaiming std")
	})

	// Test case 3: Empty tensor
	t.Run("Empty tensor", func(t *testing.T) {
		tensor := tensor.Zeros([]int{})
		KaimingNormal(&tensor)

		assert.Empty(t, tensor.Shape(), "Shape should remain empty")
		assert.Empty(t, tensor.Values(), "Values should remain empty")
	})

	t.Run("Nil tensor", func(t *testing.T) {
		var tensor *tensor.Tensor
		KaimingNormal(tensor)
		assert.Nil(t, tensor, "Tensor should remain nil")
	})
}
