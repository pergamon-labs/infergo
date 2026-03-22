package tensor

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTensorSetValues(t *testing.T) {
	testCases := []struct {
		name        string
		tensor      Tensor
		newValues   []float64
		expectedErr bool
	}{
		{
			name: "Valid set values",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			newValues:   []float64{5, 6, 7, 8},
			expectedErr: false,
		},
		{
			name: "Invalid set values - wrong length",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			newValues:   []float64{5, 6, 7},
			expectedErr: true,
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			newValues:   []float64{},
			expectedErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.tensor.SetValues(tc.newValues)

			if tc.expectedErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.newValues, tc.tensor.values)
			}
		})
	}
}

func TestTensorString(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		expected string
	}{
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expected: "[]",
		},
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{1.5, 2.7, 3.14},
				dimensions: []int{3},
			},
			expected: "[1.5000, 2.7000, 3.1400]",
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected: "[\n  [1.0000, 2.0000],\n  [3.0000, 4.0000]\n]",
		},
		{
			name: "3D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			expected: "[\n  [\n    [1.0000, 2.0000],\n    [3.0000, 4.0000]\n  ],\n  [\n    [5.0000, 6.0000],\n    [7.0000, 8.0000]\n  ]\n]",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.tensor.String()
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestTensorGetValue(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		indices  []int
		expected float64
		hasError bool
	}{
		{
			name: "1D tensor - valid index",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			indices:  []int{1},
			expected: 2,
			hasError: false,
		},
		{
			name: "2D tensor - valid indices",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			indices:  []int{1, 0},
			expected: 3,
			hasError: false,
		},
		{
			name: "3D tensor - valid indices",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			indices:  []int{1, 0, 1},
			expected: 6,
			hasError: false,
		},
		{
			name: "4D tensor - valid indices",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
				dimensions: []int{2, 2, 2, 2},
			},
			indices:  []int{1, 0, 1, 0},
			expected: 11,
			hasError: false,
		},
		{
			name: "Invalid number of indices",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			indices:  []int{1},
			expected: 0,
			hasError: true,
		},
		{
			name: "Index out of bounds",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			indices:  []int{3},
			expected: 0,
			hasError: true,
		},
		{
			name: "Index out of dimension length",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3, 3},
			},
			indices:  []int{1, 4},
			expected: 0,
			hasError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := tc.tensor.GetValue(tc.indices)
			if tc.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, result)
			}
		})
	}
}

func TestSetValue(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		indices  []int
		value    float64
		expected []float64
		hasError bool
	}{
		{
			name: "Set value in 1D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			indices:  []int{1},
			value:    5,
			expected: []float64{1, 5, 3},
			hasError: false,
		},
		{
			name: "Set value in 2D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			indices:  []int{1, 0},
			value:    7,
			expected: []float64{1, 2, 7, 4},
			hasError: false,
		},
		{
			name: "Set value in 3D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			indices:  []int{1, 0, 1},
			value:    9,
			expected: []float64{1, 2, 3, 4, 5, 9, 7, 8},
			hasError: false,
		},
		{
			name: "Invalid number of indices",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			indices:  []int{1},
			value:    5,
			expected: []float64{1, 2, 3, 4},
			hasError: true,
		},
		{
			name: "Index out of bounds",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			indices:  []int{3},
			value:    5,
			expected: []float64{1, 2, 3},
			hasError: true,
		},
		{
			name: "Index out of dimension length",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			indices:  []int{1, 2},
			value:    5,
			expected: []float64{1, 2, 3, 4},
			hasError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.tensor.SetValue(tc.indices, tc.value)
			if tc.hasError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected, tc.tensor.values)
			}
		})
	}
}

func TestTensorFill(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		value    float64
		expected []float64
	}{
		{
			name: "Fill 1D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			value:    5,
			expected: []float64{5, 5, 5},
		},
		{
			name: "Fill 2D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			value:    3.14,
			expected: []float64{3.14, 3.14, 3.14, 3.14},
		},
		{
			name: "Fill empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			value:    7,
			expected: []float64{},
		},
		{
			name: "Fill with zero",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{5},
			},
			value:    0,
			expected: []float64{0, 0, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.tensor.Fill(tc.value)
			assert.Equal(t, tc.expected, tc.tensor.values)
		})
	}
}

func TestTensorReshape(t *testing.T) {
	testCases := []struct {
		name           string
		initialTensor  Tensor
		newDimensions  []int
		expectedTensor Tensor
	}{
		{
			name: "Reshape 1D to 2D",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			newDimensions: []int{2, 2},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Reshape 2D to 1D",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			newDimensions: []int{4},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
		},
		{
			name: "Reshape to remove empty dimension",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{6, 1},
			},
			newDimensions: []int{6},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{6},
			},
		},
		{
			name: "Reshape to add empty dimension",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{6},
			},
			newDimensions: []int{6, 1},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{6, 1},
			},
		},
		{
			name: "Reshape to larger size",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			newDimensions: []int{2, 3},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 0, 0, 0},
				dimensions: []int{2, 3},
			},
		},
		{
			name: "Reshape to smaller size",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			newDimensions: []int{3, 1},
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3, 1},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.initialTensor.Reshape(tc.newDimensions)

			assert.Equal(t, tc.expectedTensor.values, tc.initialTensor.values)
			assert.Equal(t, tc.expectedTensor.dimensions, tc.initialTensor.dimensions)
		})
	}
}

func TestSqueeze(t *testing.T) {
	testCases := []struct {
		name           string
		initialTensor  Tensor
		dim            int
		expectedTensor Tensor
	}{
		{
			name: "Squeeze middle dimension",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 1, 2},
			},
			dim: 1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Squeeze all dimensions of size 1",
			initialTensor: Tensor{
				values:     []float64{1},
				dimensions: []int{1, 1, 1},
			},
			dim: -1,
			expectedTensor: Tensor{
				values:     []float64{1},
				dimensions: []int{},
			},
		},
		{
			name: "Squeeze specific dimension",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{3, 1, 2},
			},
			dim: 1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{3, 2},
			},
		},
		{
			name: "Squeeze non-existent dimension",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim: 1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.initialTensor.Squeeze(tc.dim)

			assert.Equal(t, tc.expectedTensor.values, result.values)
			assert.Equal(t, tc.expectedTensor.dimensions, result.dimensions)
		})
	}
}

func TestTensorUnsqueeze(t *testing.T) {
	testCases := []struct {
		name           string
		initialTensor  Tensor
		dim            int
		expectedTensor Tensor
	}{
		{
			name: "Unsqueeze 1D tensor at beginning",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: 0,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{1, 3},
			},
		},
		{
			name: "Unsqueeze 1D tensor at end",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: 1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3, 1},
			},
		},
		{
			name: "Unsqueeze 2D tensor in middle",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim: 1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 1, 2},
			},
		},
		{
			name: "Unsqueeze scalar",
			initialTensor: Tensor{
				values:     []float64{5},
				dimensions: []int{},
			},
			dim: 0,
			expectedTensor: Tensor{
				values:     []float64{5},
				dimensions: []int{1},
			},
		},
		{
			name: "Unsqueeze with invalid dimension (too large)",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: 2,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
		},
		{
			name: "Unsqueeze with invalid dimension (negative)",
			initialTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: -1,
			expectedTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := tc.initialTensor.Unsqueeze(tc.dim)

			assert.Equal(t, tc.expectedTensor.values, result.values)
			assert.Equal(t, tc.expectedTensor.dimensions, result.dimensions)
		})
	}
}

func TestElementMultiply(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		expected  Tensor
		expectErr bool
	}{
		{
			name: "1D tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{2, 3, 4},
				dimensions: []int{3},
			},
			expected: Tensor{
				values:     []float64{2, 6, 12},
				dimensions: []int{3},
			},
			expectErr: false,
		},
		{
			name: "2D tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 3, 4, 5},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{2, 6, 12, 20},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Mismatched values length",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ElementMultiply(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestElementAdd(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		expected  Tensor
		expectErr bool
	}{
		{
			name: "Valid addition",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{6, 8, 10, 12},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Mismatched values length",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ElementAdd(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestElementSubtract(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		expected  Tensor
		expectErr bool
	}{
		{
			name: "Valid subtraction",
			t1: Tensor{
				values:     []float64{5, 7, 9, 11},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{4, 5, 6, 7},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Mismatched values length",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ElementSubtract(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestElementDivide(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		expected  Tensor
		expectErr bool
	}{
		{
			name: "Valid division",
			t1: Tensor{
				values:     []float64{10, 20, 30, 40},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 4, 5, 8},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{5, 5, 6, 5},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Division by zero",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 0, 3, 4},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{1, math.Inf(1), 1, 1},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Mismatched values length",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			expected:  Tensor{},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ElementDivide(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestScalarDimAdd(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		dim       int
		expected  Tensor
		expectErr bool
	}{
		{
			name: "2D addition along dim 1",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{2, 4, 4, 6},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "2D addition along dim 0",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{2, 3, 5, 6},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "1D addition along dim 0",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{2, 4, 6, 8},
				dimensions: []int{4},
			},
			expectErr: false,
		},
		{
			name: "3D addition along dim 2",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				dimensions: []int{2, 2, 3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: 2,
			expected: Tensor{
				values:     []float64{2, 4, 6, 5, 7, 9, 8, 10, 12, 11, 13, 15},
				dimensions: []int{2, 2, 3},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim:       1,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Invalid dimension",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim:       2,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty scalar",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			dim:       0,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty tensors with dims",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{0},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ScalarDimAdd(tc.t1, tc.t2, tc.dim)
			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestScalarDimSubtract(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		dim       int
		expected  Tensor
		expectErr bool
	}{
		{
			name: "2D subtraction along dim 1",
			t1: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{4, 4, 6, 6},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "2D subtraction along dim 0",
			t1: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{4, 5, 5, 6},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "1D subtraction along dim 0",
			t1: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{4},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{4, 4, 4, 4},
				dimensions: []int{4},
			},
			expectErr: false,
		},
		{
			name: "3D subtraction along dim 2",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				dimensions: []int{2, 2, 3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim: 2,
			expected: Tensor{
				values:     []float64{0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9},
				dimensions: []int{2, 2, 3},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim:       1,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Invalid dimension",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim:       2,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty scalar",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			dim:       0,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty tensors with dims",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{0},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ScalarDimSubtract(tc.t1, tc.t2, tc.dim)
			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestScalarDimMultiply(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		dim       int
		expected  Tensor
		expectErr bool
	}{
		{
			name: "2D multiplication along dim 1",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 3},
				dimensions: []int{2},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{2, 6, 6, 12},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "2D multiplication along dim 0",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 3},
				dimensions: []int{2},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{2, 4, 9, 12},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "1D multiplication along dim 0",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			t2: Tensor{
				values:     []float64{2, 3, 4, 5},
				dimensions: []int{4},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{2, 6, 12, 20},
				dimensions: []int{4},
			},
			expectErr: false,
		},
		{
			name: "3D multiplication along dim 2",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				dimensions: []int{2, 2, 3},
			},
			t2: Tensor{
				values:     []float64{2, 3, 4},
				dimensions: []int{3},
			},
			dim: 2,
			expected: Tensor{
				values:     []float64{2, 6, 12, 8, 15, 24, 14, 24, 36, 20, 33, 48},
				dimensions: []int{2, 2, 3},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim:       1,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Invalid dimension",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim:       2,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty scalar",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			dim:       0,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty tensors with dims",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{0},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ScalarDimMultiply(tc.t1, tc.t2, tc.dim)
			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestScalarDimDivide(t *testing.T) {
	testCases := []struct {
		name      string
		t1        Tensor
		t2        Tensor
		dim       int
		expected  Tensor
		expectErr bool
	}{
		{
			name: "2D division along dim 1",
			t1: Tensor{
				values:     []float64{6, 8, 10, 12},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 4},
				dimensions: []int{2},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{3, 2, 5, 3},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "2D division along dim 0",
			t1: Tensor{
				values:     []float64{6, 8, 10, 12},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{2, 5},
				dimensions: []int{2},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{3, 4, 2, 2.4},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "1D division along dim 0",
			t1: Tensor{
				values:     []float64{10, 20, 30, 40},
				dimensions: []int{4},
			},
			t2: Tensor{
				values:     []float64{2, 4, 5, 8},
				dimensions: []int{4},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{5, 5, 6, 5},
				dimensions: []int{4},
			},
			expectErr: false,
		},
		{
			name: "3D division along dim 2",
			t1: Tensor{
				values:     []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24},
				dimensions: []int{2, 2, 3},
			},
			t2: Tensor{
				values:     []float64{2, 2, 3},
				dimensions: []int{3},
			},
			dim: 2,
			expected: Tensor{
				values:     []float64{1, 2, 2, 4, 5, 4, 7, 8, 6, 10, 11, 8},
				dimensions: []int{2, 2, 3},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim:       1,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Invalid dimension",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			dim:       2,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty scalar",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			dim:       0,
			expected:  Tensor{},
			expectErr: true,
		},
		{
			name: "Empty tensors with dims",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{0},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{0, 0},
			},
			expectErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ScalarDimDivide(tc.t1, tc.t2, tc.dim)
			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestValidateDimensionMatch(t *testing.T) {
	testCases := []struct {
		name      string
		t1        *Tensor
		t2        *Tensor
		expectErr bool
	}{
		{
			name: "Matching dimensions and values",
			t1: &Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: &Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			expectErr: false,
		},
		{
			name: "Mismatched dimensions",
			t1: &Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: &Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expectErr: true,
		},
		{
			name: "Mismatched values length",
			t1: &Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: &Tensor{
				values:     []float64{1, 2},
				dimensions: []int{2},
			},
			expectErr: true,
		},
		{
			name: "Empty tensors",
			t1: &Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: &Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expectErr: false,
		},
		{
			name: "Same value count but different exact shapes",
			t1: &Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: &Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4, 1},
			},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateDimensionMatch(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestScalarAdd(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		scalar   float64
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			scalar: 2,
			expected: Tensor{
				values:     []float64{3, 4, 5},
				dimensions: []int{3},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			scalar: 1.5,
			expected: Tensor{
				values:     []float64{2.5, 3.5, 4.5, 5.5},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			scalar: 3,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := ScalarAdd(tc.tensor, tc.scalar)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestScalarSubtract(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		scalar   float64
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{3, 4, 5},
				dimensions: []int{3},
			},
			scalar: 2,
			expected: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{2.5, 3.5, 4.5, 5.5},
				dimensions: []int{2, 2},
			},
			scalar: 1.5,
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			scalar: 3,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := ScalarSubtract(tc.tensor, tc.scalar)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestScalarMultiply(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		scalar   float64
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			scalar: 2,
			expected: Tensor{
				values:     []float64{2, 4, 6},
				dimensions: []int{3},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			scalar: 1.5,
			expected: Tensor{
				values:     []float64{1.5, 3, 4.5, 6},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			scalar: 3,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := ScalarMultiply(tc.tensor, tc.scalar)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestScalarDivide(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		scalar   float64
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{2, 4, 6},
				dimensions: []int{3},
			},
			scalar: 2,
			expected: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{1.5, 3, 4.5, 6},
				dimensions: []int{2, 2},
			},
			scalar: 1.5,
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			scalar: 3,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := ScalarDivide(tc.tensor, tc.scalar)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestTensorAbs(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		expected Tensor
	}{
		{
			name: "1D tensor with positive and negative values",
			tensor: Tensor{
				values:     []float64{-1, 2, -3, 4},
				dimensions: []int{4},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
		},
		{
			name: "2D tensor with positive and negative values",
			tensor: Tensor{
				values:     []float64{-1.5, 2.5, -3.5, 4.5},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{1.5, 2.5, 3.5, 4.5},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.tensor.Abs()
			assert.Equal(t, tc.expected.values, tc.tensor.values)
			assert.Equal(t, tc.expected.dimensions, tc.tensor.dimensions)
		})
	}
}

func TestTensorSqrt(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{1, 4, 9, 16},
				dimensions: []int{4},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{1, 4, 9, 16},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.tensor.Sqrt()
			assert.InDeltaSlice(t, tc.expected.values, tc.tensor.values, 1e-6)
			assert.Equal(t, tc.expected.dimensions, tc.tensor.dimensions)
		})
	}
}

func TestTensorClip(t *testing.T) {
	testCases := []struct {
		name     string
		tensor   Tensor
		min      float64
		max      float64
		expected Tensor
	}{
		{
			name: "1D tensor",
			tensor: Tensor{
				values:     []float64{-1, 2, 3, 6},
				dimensions: []int{4},
			},
			min: 0,
			max: 5,
			expected: Tensor{
				values:     []float64{0, 2, 3, 5},
				dimensions: []int{4},
			},
		},
		{
			name: "2D tensor",
			tensor: Tensor{
				values:     []float64{-1.5, 2.5, 3.5, 6.5},
				dimensions: []int{2, 2},
			},
			min: 0,
			max: 5,
			expected: Tensor{
				values:     []float64{0, 2.5, 3.5, 5},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "Empty tensor",
			tensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			min: 0,
			max: 5,
			expected: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tc.tensor.Clip(tc.min, tc.max)
			assert.Equal(t, tc.expected.values, tc.tensor.values)
			assert.Equal(t, tc.expected.dimensions, tc.tensor.dimensions)
		})
	}
}

func TestConcatTensors(t *testing.T) {
	testCases := []struct {
		name      string
		dim       int
		tensors   []Tensor
		expected  Tensor
		expectErr bool
	}{
		{
			name: "Concat 1D tensors",
			dim:  0,
			tensors: []Tensor{
				{values: []float64{1, 2}, dimensions: []int{2}},
				{values: []float64{3, 4}, dimensions: []int{2}},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			expectErr: false,
		},
		{
			name: "Concat 2D tensors along dim 0",
			dim:  0,
			tensors: []Tensor{
				{values: []float64{1, 2, 3, 4}, dimensions: []int{2, 2}},
				{values: []float64{5, 6, 7, 8}, dimensions: []int{2, 2}},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{4, 2},
			},
			expectErr: false,
		},
		{
			name: "Concat 2D tensors along dim 1",
			dim:  1,
			tensors: []Tensor{
				{values: []float64{1, 2, 3, 4}, dimensions: []int{2, 2}},
				{values: []float64{5, 6, 7, 8}, dimensions: []int{2, 2}},
			},
			expected: Tensor{
				values:     []float64{1, 2, 5, 6, 3, 4, 7, 8},
				dimensions: []int{2, 4},
			},
			expectErr: false,
		},
		{
			name: "Concat 3D tensors along dim 1",
			dim:  1,
			tensors: []Tensor{
				{values: []float64{1, 2, 3, 4, 5, 6, 7, 8}, dimensions: []int{2, 2, 2}},
				{values: []float64{9, 10, 11, 12, 13, 14, 15, 16}, dimensions: []int{2, 2, 2}},
			},
			expected: Tensor{
				values:     []float64{1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16},
				dimensions: []int{2, 4, 2},
			},
			expectErr: false,
		},
		{
			name: "Incompatible dimensions",
			dim:  1,
			tensors: []Tensor{
				{values: []float64{1, 2, 3, 4}, dimensions: []int{2, 2}},
				{values: []float64{3, 4, 5, 6, 7, 8}, dimensions: []int{3, 2}},
			},
			expectErr: true,
		},
		{
			name:      "No tensors provided",
			dim:       0,
			tensors:   []Tensor{},
			expectErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := ConcatTensors(tc.dim, tc.tensors...)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestDot(t *testing.T) {
	testCases := []struct {
		name         string
		t1           Tensor
		t2           Tensor
		expected     Tensor
		expectErr    bool
		expectScalar bool
	}{
		{
			name: "2D x 2D tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{19, 22, 43, 50},
				dimensions: []int{2, 2},
			},
			expectErr:    false,
			expectScalar: false,
		},
		{
			name: "2D x 1D tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{5, 6},
				dimensions: []int{2},
			},
			expected: Tensor{
				values:     []float64{17, 39},
				dimensions: []int{2},
			},
			expectErr:    false,
			expectScalar: false,
		},
		{
			name: "2D x 1D tensors of different sizes where dimensions are compatible",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{3, 2},
			},
			t2: Tensor{
				values:     []float64{5, 6},
				dimensions: []int{2},
			},
			expected: Tensor{
				values:     []float64{17, 39, 61},
				dimensions: []int{3},
			},
			expectErr:    false,
			expectScalar: false,
		},
		{
			name: "1D x 1D tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{4, 5, 6},
				dimensions: []int{3},
			},
			expected: Tensor{
				values:     []float64{32},
				dimensions: []int{},
			},
			expectErr:    false,
			expectScalar: true,
		},
		{
			name: "Incompatible dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{4, 5},
				dimensions: []int{2},
			},
			expectErr:    true,
			expectScalar: false,
		},
		{
			name: "Empty tensors",
			t1: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			t2: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expectErr:    true,
			expectScalar: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Dot(tc.t1, tc.t2)

			if tc.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
				assert.Equal(t, tc.expectScalar, result.IsScalar())
			}
		})
	}
}

func TestTranspose(t *testing.T) {
	testCases := []struct {
		name        string
		input       Tensor
		dim1        int
		dim2        int
		expected    Tensor
		expectError bool
	}{
		{
			name: "2D tensor transpose",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			dim1: 0,
			dim2: 1,
			expected: Tensor{
				values:     []float64{1, 4, 2, 5, 3, 6},
				dimensions: []int{3, 2},
			},
			expectError: false,
		},
		{
			name: "3D tensor transpose",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			dim1: 0,
			dim2: 2,
			expected: Tensor{
				values:     []float64{1, 5, 3, 7, 2, 6, 4, 8},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "No change when dimensions are the same",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim1: 1,
			dim2: 1,
			expected: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "Invalid dimension",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim1:        0,
			dim2:        2,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Transpose(tc.input, tc.dim1, tc.dim2)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestMatMul(t *testing.T) {
	testCases := []struct {
		name        string
		leftTensor  Tensor
		rightTensor Tensor
		expected    Tensor
		expectError bool
	}{
		{
			name: "Valid matrix multiplication",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			rightTensor: Tensor{
				values:     []float64{5, 6, 7, 8},
				dimensions: []int{2, 2},
			},
			expected: Tensor{
				values:     []float64{19, 22, 43, 50},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "Different batch and rows dimensions",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			rightTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{3, 2},
			},
			expected: Tensor{
				values:     []float64{22, 28, 49, 64},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "Transpose dimensions",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			rightTensor: Tensor{
				values:     []float64{1, 4, 2, 5, 3, 6},
				dimensions: []int{3, 2},
			},
			expected: Tensor{
				values:     []float64{14, 32, 32, 77},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "1Dx1D tensor multiplication",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			rightTensor: Tensor{
				values:     []float64{4, 5, 6},
				dimensions: []int{3},
			},
			expectError: true,
		},
		{
			name: "1Dx2D tensor multiplication",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			rightTensor: Tensor{
				values:     []float64{4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 2},
			},
			expected: Tensor{
				values:     []float64{40, 46},
				dimensions: []int{2},
			},
			expectError: false,
		},
		{
			name: "Incompatible dimensions",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3, 1},
			},
			rightTensor: Tensor{
				values:     []float64{4, 5, 6},
				dimensions: []int{3, 1},
			},
			expected:    Tensor{},
			expectError: true,
		},
		{
			name: "3D tensor multiplication",
			leftTensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			rightTensor: Tensor{
				values:     []float64{9, 10, 11, 12, 13, 14, 15, 16},
				dimensions: []int{2, 2, 2},
			},
			expected: Tensor{
				values:     []float64{31, 34, 71, 78, 155, 166, 211, 226},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "Empty tensors",
			leftTensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			rightTensor: Tensor{
				values:     []float64{},
				dimensions: []int{},
			},
			expected:    Tensor{},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := MatMul(tc.leftTensor, tc.rightTensor)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}
func TestSlice(t *testing.T) {
	testCases := []struct {
		name        string
		input       Tensor
		start       []int
		end         []int
		expected    Tensor
		expectError bool
	}{
		{
			name: "Slice 1D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{5},
			},
			start: []int{1},
			end:   []int{4},
			expected: Tensor{
				values:     []float64{2, 3, 4},
				dimensions: []int{3},
			},
			expectError: false,
		},
		{
			name: "Slice 2D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			start: []int{0, 1},
			end:   []int{2, 3},
			expected: Tensor{
				values:     []float64{2, 3, 5, 6},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "Slice 3D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
				dimensions: []int{2, 2, 4},
			},
			start: []int{0, 1, 1},
			end:   []int{2, 2, 3},
			expected: Tensor{
				values:     []float64{6, 7, 14, 15},
				dimensions: []int{2, 1, 2},
			},
			expectError: false,
		},
		{
			name: "Invalid start index",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{-1, 0},
			end:         []int{2, 2},
			expectError: true,
		},
		{
			name: "Invalid end index",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{0, 0},
			end:         []int{2, 3},
			expectError: true,
		},
		{
			name: "Start index greater than or equal to end index",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{1, 1},
			end:         []int{1, 2},
			expectError: true,
		},
		{
			name: "Mismatched number of indices",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{0},
			end:         []int{2, 2},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Slice(tc.input, tc.start, tc.end)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestSetSlice(t *testing.T) {
	testCases := []struct {
		name        string
		input       Tensor
		start       []int
		end         []int
		values      []float64
		expected    Tensor
		expectError bool
	}{
		{
			name: "Set slice in 1D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{5},
			},
			start:  []int{1},
			end:    []int{4},
			values: []float64{6, 7, 8},
			expected: Tensor{
				values:     []float64{1, 6, 7, 8, 5},
				dimensions: []int{5},
			},
			expectError: false,
		},
		{
			name: "Set slice in 2D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			start:  []int{0, 1},
			end:    []int{2, 3},
			values: []float64{7, 8, 9, 10},
			expected: Tensor{
				values:     []float64{1, 7, 8, 4, 9, 10},
				dimensions: []int{2, 3},
			},
			expectError: false,
		},
		{
			name: "Set slice in 3D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			start:  []int{0, 1, 0},
			end:    []int{2, 2, 2},
			values: []float64{9, 10, 11, 12},
			expected: Tensor{
				values:     []float64{1, 2, 9, 10, 5, 6, 11, 12},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "Invalid start index",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{-1, 0},
			end:         []int{2, 2},
			values:      []float64{5, 6, 7, 8},
			expectError: true,
		},
		{
			name: "Invalid end index",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{0, 0},
			end:         []int{2, 3},
			values:      []float64{5, 6, 7, 8},
			expectError: true,
		},
		{
			name: "Mismatched number of values",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			start:       []int{0, 0},
			end:         []int{2, 2},
			values:      []float64{5, 6, 7},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.input.SetSlice(tc.start, tc.end, tc.values)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, tc.input.values)
				assert.Equal(t, tc.expected.dimensions, tc.input.dimensions)
			}
		})
	}
}

func TestRotate180(t *testing.T) {
	testCases := []struct {
		name        string
		input       Tensor
		dim1        int
		dim2        int
		expected    Tensor
		expectError bool
	}{
		{
			name: "2D tensor rotation 2x2",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim1: 0,
			dim2: 1,
			expected: Tensor{
				values:     []float64{4, 3, 2, 1},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "2D tensor rotation 3x3",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			dim1: 0,
			dim2: 1,
			expected: Tensor{
				values:     []float64{9, 8, 7, 6, 5, 4, 3, 2, 1},
				dimensions: []int{3, 3},
			},
			expectError: false,
		},
		{
			name: "2D tensor rotation 2x3",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			dim1: 0,
			dim2: 1,
			expected: Tensor{
				values:     []float64{6, 5, 4, 3, 2, 1},
				dimensions: []int{2, 3},
			},
			expectError: false,
		},
		{
			name: "3D tensor rotation",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8},
				dimensions: []int{2, 2, 2},
			},
			dim1: 0,
			dim2: 1,
			expected: Tensor{
				values:     []float64{7, 8, 5, 6, 3, 4, 1, 2},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "Invalid dimensions",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim1:        0,
			dim2:        2,
			expectError: true,
		},
		{
			name: "Same dimension",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim1:        0,
			dim2:        0,
			expectError: true,
		},
		{
			name: "1D tensor",
			input: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			dim1:        0,
			dim2:        0,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Rotate180(tc.input, tc.dim1, tc.dim2)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)

				twoFoldResult, err := Rotate180(result, tc.dim1, tc.dim2)
				assert.NoError(t, err)
				assert.Equal(t, tc.input.values, twoFoldResult.values)
				assert.Equal(t, tc.input.dimensions, twoFoldResult.dimensions)
			}
		})
	}
}

func TestFlip(t *testing.T) {
	testCases := []struct {
		name        string
		input       Tensor
		dim         int
		expected    Tensor
		expectError bool
	}{
		{
			name: "1D tensor flip",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{4, 3, 2, 1},
				dimensions: []int{4},
			},
			expectError: false,
		},
		{
			name: "2D tensor flip along dim 0",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{4, 5, 6, 1, 2, 3},
				dimensions: []int{2, 3},
			},
			expectError: false,
		},
		{
			name: "2D tensor flip along dim 1",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{3, 2, 1, 6, 5, 4},
				dimensions: []int{2, 3},
			},
			expectError: false,
		},
		{
			name: "3D tensor flip along dim 1",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				dimensions: []int{2, 3, 2},
			},
			dim: 1,
			expected: Tensor{
				values:     []float64{5, 6, 3, 4, 1, 2, 11, 12, 9, 10, 7, 8},
				dimensions: []int{2, 3, 2},
			},
			expectError: false,
		},
		{
			name: "3D tensor flip along dim 0",
			input: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
				dimensions: []int{2, 3, 2},
			},
			dim: 0,
			expected: Tensor{
				values:     []float64{7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6},
				dimensions: []int{2, 3, 2},
			},
			expectError: false,
		},
		{
			name: "Invalid dimension",
			input: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			dim:         2,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Flip(tc.input, tc.dim)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)

				twoFoldResult, err := Flip(result, tc.dim)
				assert.NoError(t, err)
				assert.Equal(t, tc.input.values, twoFoldResult.values)
				assert.Equal(t, tc.input.dimensions, twoFoldResult.dimensions)
			}
		})
	}
}

func TestConvolve(t *testing.T) {
	testCases := []struct {
		name        string
		tensor      Tensor
		kernel      Tensor
		padding     []int
		stride      []int
		expected    Tensor
		expectError bool
	}{
		{
			name: "1D convolution",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{1, 1, 1, 5},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1},
				dimensions: []int{1, 1, 1, 3},
			},
			padding: []int{0, 0},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-2, -2, -2},
				dimensions: []int{1, 1, 1, 3},
			},
			expectError: false,
		},
		{
			name: "2D convolution",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{1, 1, 3, 3},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{1, 1, 3, 3},
			},
			padding: []int{0, 0},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-8},
				dimensions: []int{1, 1, 1, 1},
			},
			expectError: false,
		},
		{
			name: "2D convolution with padding",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{1, 1, 3, 3},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{1, 1, 3, 3},
			},
			padding: []int{1, 1},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-9, -6, 9, -20, -8, 20, -21, -6, 21},
				dimensions: []int{1, 1, 3, 3},
			},
			expectError: false,
		},
		{
			name: "2D convolution with stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
				dimensions: []int{1, 1, 5, 5},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{1, 1, 3, 3},
			},
			padding: []int{0, 0},
			stride:  []int{2, 2},
			expected: Tensor{
				values:     []float64{-8, -8, -8, -8},
				dimensions: []int{1, 1, 2, 2},
			},
			expectError: false,
		},
		{
			name: "2D convolution with padding and stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
				dimensions: []int{1, 1, 5, 5},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{1, 1, 3, 3},
			},
			padding: []int{1, 1},
			stride:  []int{2, 2},
			expected: Tensor{
				values:     []float64{-11, -6, 17, -48, -8, 56, -61, -6, 67},
				dimensions: []int{1, 1, 3, 3},
			},
			expectError: false,
		},
		{
			name: "2D convolution with batch size",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
				dimensions: []int{2, 1, 3, 3},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{1, 1, 3, 3},
			},
			padding: []int{0, 0},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-8, -8},
				dimensions: []int{2, 1, 1, 1},
			},
			expectError: false,
		},
		{
			name: "2D convolution with batch size and channel size",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
				dimensions: []int{2, 2, 3, 3},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2, 1, 0, -1, 2, 0, -2},
				dimensions: []int{2, 2, 3, 3},
			},
			padding: []int{0, 0},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-18, -18, -18, -18},
				dimensions: []int{2, 2, 1, 1},
			},
			expectError: false,
		},
		{
			name: "2D convolution with channel upsampling",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{1, 1, 3, 3},
			},
			kernel: Tensor{
				values:     []float64{1, 0, -1, 2, 0, -2, 1, 0, -1, 1, 0, -1, 2, 0, -2, 1, 0, -1},
				dimensions: []int{2, 1, 3, 3},
			},
			padding: []int{0, 0},
			stride:  []int{1, 1},
			expected: Tensor{
				values:     []float64{-8, -8},
				dimensions: []int{1, 2, 1, 1},
			},
			expectError: false,
		},
		{
			name: "Mismatched dimensions",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{1, 1, 2, 2},
			},
			kernel: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{1, 1, 3},
			},
			padding:     []int{0, 0},
			stride:      []int{1, 1},
			expectError: true,
		},
		{
			name: "Invalid dimensions for convolution",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			kernel: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			padding:     []int{0, 0},
			stride:      []int{1, 1},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := Convolve(tc.tensor, tc.kernel, tc.padding, tc.stride)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestMaxPool(t *testing.T) {
	testCases := []struct {
		name        string
		tensor      Tensor
		kernelSize  []int
		stride      []int
		expected    Tensor
		expectError bool
	}{
		{
			name: "1D Max Pooling with default stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{5},
			},
			kernelSize: []int{2},
			stride:     []int{0},
			expected: Tensor{
				values:     []float64{2, 4},
				dimensions: []int{2},
			},
			expectError: false,
		},
		{
			name: "2D Max Pooling with half stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			kernelSize: []int{2, 2},
			stride:     []int{1, 1},
			expected: Tensor{
				values:     []float64{5, 6, 8, 9},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "2D Max Pooling with default stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			kernelSize: []int{2, 2},
			stride:     []int{0, 0},
			expected: Tensor{
				values:     []float64{5},
				dimensions: []int{1, 1},
			},
			expectError: false,
		},
		{
			name: "2D Max Pooling with over stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
				dimensions: []int{5, 5},
			},
			kernelSize: []int{2, 2},
			stride:     []int{3, 3},
			expected: Tensor{
				values:     []float64{7, 10, 22, 25},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "3D Max Pooling",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
				dimensions: []int{2, 2, 4},
			},
			kernelSize: []int{1, 1, 2},
			stride:     []int{1, 1, 2},
			expected: Tensor{
				values:     []float64{2, 4, 6, 8, 10, 12, 14, 16},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "Mismatched dimensions",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			kernelSize:  []int{2},
			stride:      []int{1, 1},
			expectError: true,
		},
		{
			name: "Invalid dimensions for max pooling",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			kernelSize:  []int{4},
			stride:      []int{1},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := MaxPool(tc.tensor, tc.kernelSize, tc.stride)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestAvgPool(t *testing.T) {
	testCases := []struct {
		name        string
		tensor      Tensor
		kernelSize  []int
		stride      []int
		expected    Tensor
		expectError bool
	}{
		{
			name: "1D Avg Pooling with default stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5},
				dimensions: []int{5},
			},
			kernelSize: []int{2},
			stride:     []int{0},
			expected: Tensor{
				values:     []float64{1.5, 3.5},
				dimensions: []int{2},
			},
			expectError: false,
		},
		{
			name: "2D Avg Pooling with half stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			kernelSize: []int{2, 2},
			stride:     []int{1, 1},
			expected: Tensor{
				values:     []float64{3, 4, 6, 7},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "2D Avg Pooling with default stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
				dimensions: []int{3, 3},
			},
			kernelSize: []int{2, 2},
			stride:     []int{0, 0},
			expected: Tensor{
				values:     []float64{3},
				dimensions: []int{1, 1},
			},
			expectError: false,
		},
		{
			name: "2D Avg Pooling with over stride",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
				dimensions: []int{5, 5},
			},
			kernelSize: []int{2, 2},
			stride:     []int{3, 3},
			expected: Tensor{
				values:     []float64{4, 7, 19, 22},
				dimensions: []int{2, 2},
			},
			expectError: false,
		},
		{
			name: "3D Avg Pooling",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
				dimensions: []int{2, 2, 4},
			},
			kernelSize: []int{1, 1, 2},
			stride:     []int{1, 1, 2},
			expected: Tensor{
				values:     []float64{1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5},
				dimensions: []int{2, 2, 2},
			},
			expectError: false,
		},
		{
			name: "Mismatched dimensions",
			tensor: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			kernelSize:  []int{2},
			stride:      []int{1, 1},
			expectError: true,
		},
		{
			name: "Invalid dimensions for avg pooling",
			tensor: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			kernelSize:  []int{4},
			stride:      []int{1},
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := AvgPool(tc.tensor, tc.kernelSize, tc.stride)

			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expected.values, result.values)
				assert.Equal(t, tc.expected.dimensions, result.dimensions)
			}
		})
	}
}

func TestOnes(t *testing.T) {
	testCases := []struct {
		name     string
		dims     []int
		expected Tensor
	}{
		{
			name: "1D Tensor",
			dims: []int{3},
			expected: Tensor{
				values:     []float64{1, 1, 1},
				dimensions: []int{3},
			},
		},
		{
			name: "2D Tensor",
			dims: []int{2, 2},
			expected: Tensor{
				values:     []float64{1, 1, 1, 1},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "3D Tensor",
			dims: []int{2, 2, 2},
			expected: Tensor{
				values:     []float64{1, 1, 1, 1, 1, 1, 1, 1},
				dimensions: []int{2, 2, 2},
			},
		},
		{
			name:     "Empty Tensor",
			dims:     []int{},
			expected: Tensor{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := Ones(tc.dims)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestZeros(t *testing.T) {
	testCases := []struct {
		name     string
		dims     []int
		expected Tensor
	}{
		{
			name: "1D Tensor",
			dims: []int{3},
			expected: Tensor{
				values:     []float64{0, 0, 0},
				dimensions: []int{3},
			},
		},
		{
			name: "2D Tensor",
			dims: []int{2, 2},
			expected: Tensor{
				values:     []float64{0, 0, 0, 0},
				dimensions: []int{2, 2},
			},
		},
		{
			name: "3D Tensor",
			dims: []int{2, 2, 2},
			expected: Tensor{
				values:     []float64{0, 0, 0, 0, 0, 0, 0, 0},
				dimensions: []int{2, 2, 2},
			},
		},
		{
			name:     "Empty Tensor",
			dims:     []int{},
			expected: Tensor{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := Zeros(tc.dims)
			assert.Equal(t, tc.expected.values, result.values)
			assert.Equal(t, tc.expected.dimensions, result.dimensions)
		})
	}
}

func TestRand(t *testing.T) {
	testCases := []struct {
		name string
		dims []int
		min  float64
		max  float64
	}{
		{
			name: "1D Tensor",
			dims: []int{5},
			min:  0,
			max:  1,
		},
		{
			name: "2D Tensor",
			dims: []int{3, 3},
			min:  -1,
			max:  1,
		},
		{
			name: "3D Tensor",
			dims: []int{2, 2, 2},
			min:  0,
			max:  10,
		},
		{
			name: "Empty Tensor",
			dims: []int{},
			min:  0,
			max:  1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := Rand(tc.dims, tc.min, tc.max)

			// Check dimensions
			assert.Equal(t, tc.dims, result.dimensions)

			// Check number of values
			expectedLength := 1
			for _, dim := range tc.dims {
				expectedLength *= dim
			}
			assert.Equal(t, expectedLength, len(result.values))

			// Check if values are within the specified range
			for _, v := range result.values {
				assert.GreaterOrEqual(t, v, tc.min)
				assert.Less(t, v, tc.max)
			}

			// Check if at least one value is different.  This could theoretically fail, but
			// it's extremely unlikely.  Should you be so unlucky, do run the test again
			if len(result.values) > 1 {
				allSame := true
				for i := 1; i < len(result.values); i++ {
					if result.values[i] != result.values[0] {
						allSame = false
						break
					}
				}
				assert.False(t, allSame, "All values in the random tensor are the same")
			}
		})
	}
}

func TestEqual(t *testing.T) {
	testCases := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected bool
	}{
		{
			name: "Equal tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			expected: true,
		},
		{
			name: "Different values",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 4},
				dimensions: []int{3},
			},
			expected: false,
		},
		{
			name: "Different dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			expected: false,
		},
		{
			name:     "Empty tensors",
			t1:       Tensor{},
			t2:       Tensor{},
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := Equal(tc.t1, tc.t2)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestEqualValues(t *testing.T) {
	testCases := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected bool
	}{
		{
			name: "Equal values",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			expected: true,
		},
		{
			name: "Different values",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 4},
				dimensions: []int{3},
			},
			expected: false,
		},
		{
			name: "Equal values, different dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			expected: true,
		},
		{
			name:     "Empty tensors",
			t1:       Tensor{},
			t2:       Tensor{},
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := EqualValues(tc.t1, tc.t2)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestInDelta(t *testing.T) {
	testCases := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		delta    float64
		expected bool
	}{
		{
			name: "Equal tensors",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			delta:    1e-6,
			expected: true,
		},
		{
			name: "Values within delta",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1.0001, 1.9999, 3.0002},
				dimensions: []int{3},
			},
			delta:    0.001,
			expected: true,
		},
		{
			name: "Values outside delta",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1.002, 2, 3.003},
				dimensions: []int{3},
			},
			delta:    0.001,
			expected: false,
		},
		{
			name: "Different dimensions",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{1, 3},
			},
			delta:    1e-6,
			expected: false,
		},
		{
			name:     "Empty tensors",
			t1:       Tensor{},
			t2:       Tensor{},
			delta:    1e-6,
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := InDelta(tc.t1, tc.t2, tc.delta)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestInDeltaValues(t *testing.T) {
	testCases := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		delta    float64
		expected bool
	}{
		{
			name: "Equal values",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			delta:    1e-6,
			expected: true,
		},
		{
			name: "Values within delta",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1.0001, 1.9999, 3.0002},
				dimensions: []int{3},
			},
			delta:    0.001,
			expected: true,
		},
		{
			name: "Values outside delta",
			t1: Tensor{
				values:     []float64{1, 2, 3},
				dimensions: []int{3},
			},
			t2: Tensor{
				values:     []float64{1.002, 2, 3.003},
				dimensions: []int{3},
			},
			delta:    0.001,
			expected: false,
		},
		{
			name: "Different dimensions, equal values",
			t1: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{2, 2},
			},
			t2: Tensor{
				values:     []float64{1, 2, 3, 4},
				dimensions: []int{4},
			},
			delta:    1e-6,
			expected: true,
		},
		{
			name:     "Empty tensors",
			t1:       Tensor{},
			t2:       Tensor{},
			delta:    1e-6,
			expected: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := InDeltaValues(tc.t1, tc.t2, tc.delta)
			assert.Equal(t, tc.expected, result)
		})
	}
}
