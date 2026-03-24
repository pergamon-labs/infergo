package functional

import (
	"testing"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"

	"github.com/stretchr/testify/assert"
)

func TestActivationTypeString(t *testing.T) {
	tests := []struct {
		name     string
		aType    ActivationType
		expected string
	}{
		{ActivationNone.String(), ActivationNone, "None"},
		{ActivationReLU.String(), ActivationReLU, "ReLU"},
		{ActivationPReLU.String(), ActivationPReLU, "PReLU"},
		{ActivationELU.String(), ActivationELU, "ELU"},
		{ActivationSigmoid.String(), ActivationSigmoid, "Sigmoid"},
		{ActivationRangeSigmoid.String(), ActivationRangeSigmoid, "RangeSigmoid"},
		{ActivationTanH.String(), ActivationTanH, "TanH"},
		{ActivationSwish.String(), ActivationSwish, "Swish"},
		{ActivationSoftmax.String(), ActivationSoftmax, "Softmax"},
		{ActivationLayerNormalization.String(), ActivationLayerNormalization, "LayerNormalization"},
		{ActivationType(999).String(), ActivationType(999), "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.aType.String(), "ActivationType.String() should match expected value")
		})
	}
}

func TestGetActivationFunction(t *testing.T) {
	tests := []struct {
		name        string
		aType       ActivationType
		expectError bool
	}{
		{"None", ActivationNone, false},
		{"ReLU", ActivationReLU, false},
		{"Sigmoid", ActivationSigmoid, false},
		{"RangeSigmoid", ActivationRangeSigmoid, false},
		{"TanH", ActivationTanH, false},
		{"Swish", ActivationSwish, false},
		{"Softmax", ActivationSoftmax, false},
		{"LayerNormalization", ActivationLayerNormalization, false},
		{"Unknown", ActivationType(999), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetActivationFunction(tt.aType)
			if tt.expectError {
				assert.Error(t, err, "GetActivationFunction() should return an error")
				assert.Nil(t, got, "GetActivationFunction() should return nil when error is expected")
			} else {
				assert.NoError(t, err, "GetActivationFunction() should not return an error")
				assert.NotNil(t, got, "GetActivationFunction() should return a non-nil function")
			}
		})
	}
}

func TestGetActivationFunctionReturnType(t *testing.T) {
	validTypes := []ActivationType{
		ActivationNone, ActivationReLU,
		ActivationSigmoid, ActivationRangeSigmoid, ActivationTanH,
		ActivationSwish, ActivationSoftmax, ActivationLayerNormalization,
	}

	for _, aType := range validTypes {
		t.Run(aType.String(), func(t *testing.T) {
			got, err := GetActivationFunction(aType)
			assert.NoError(t, err, "GetActivationFunction() should not return an error")

			// Check if the returned function is of type ActivationFn
			_, ok := interface{}(got).(ActivationFn)
			assert.True(t, ok, "GetActivationFunction() should return an ActivationFn")
		})
	}
}

func TestGetActivationFunctionPrime(t *testing.T) {
	tests := []struct {
		name        string
		aType       ActivationType
		expectError bool
	}{
		{"None", ActivationNone, false},
		{"ReLU", ActivationReLU, false},
		{"Sigmoid", ActivationSigmoid, false},
		{"RangeSigmoid", ActivationRangeSigmoid, false},
		{"TanH", ActivationTanH, false},
		{"Swish", ActivationSwish, false},
		{"Softmax", ActivationSoftmax, false},
		{"LayerNormalization", ActivationLayerNormalization, true},
		{"Unknown", ActivationType(999), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetActivationFunctionPrime(tt.aType)
			if tt.expectError {
				assert.Error(t, err, "GetActivationFunctionPrime() should return an error")
				assert.Nil(t, got, "GetActivationFunctionPrime() should return nil when error is expected")
			} else {
				assert.NoError(t, err, "GetActivationFunctionPrime() should not return an error")
				assert.NotNil(t, got, "GetActivationFunctionPrime() should return a non-nil function")
			}
		})
	}
}

func TestGetActivationFunctionPrimeReturnType(t *testing.T) {
	validTypes := []ActivationType{
		ActivationNone, ActivationReLU,
		ActivationSigmoid, ActivationRangeSigmoid, ActivationTanH,
		ActivationSwish, ActivationSoftmax,
	}

	for _, aType := range validTypes {
		t.Run(aType.String(), func(t *testing.T) {
			got, err := GetActivationFunctionPrime(aType)
			assert.NoError(t, err, "GetActivationFunctionPrime() should not return an error")

			// Check if the returned function is of type ActivationFn
			_, ok := interface{}(got).(ActivationFn)
			assert.True(t, ok, "GetActivationFunctionPrime() should return an ActivationFn")
		})
	}
}

func TestIdentity(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Single value",
			input: tensor.New([]float64{5.0}, []int{1}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{5.0}, []int{1}),
		},
		{
			name:  "Multiple values",
			input: tensor.New([]float64{-2.0, 0.0, 3.5}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-2.0, 0.0, 3.5}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Identity(tt.input, &tt.params)
			assert.NoError(t, err, "Identity() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestIdentityPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Single value",
			input: tensor.New([]float64{5.0}, []int{1}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{1.0}, []int{1}),
		},
		{
			name:  "Multiple values",
			input: tensor.New([]float64{-2.0, 0.0, 3.5}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{1.0, 1.0, 1.0}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := IdentityPrime(tt.input, &tt.params)
			assert.NoError(t, err, "IdentityPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.5, 0.7310585786300049, 0.8807970779778823}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, 0.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.11920292202211755, 0.2689414213699951, 0.5}, []int{3}),
		},
		{
			name:  "Mixed values with alpha",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.5,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.2689414213699951, 0.5, 0.8807970779778823}, []int{3}),
		},
		{
			name:  "Mixed values with alpha of zero",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.5, 0.5, 0.8807970779778823}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Sigmoid(tt.input, &tt.params)
			assert.NoError(t, err, "Sigmoid() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestSigmoidPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.25, 0.19661193324148185, 0.10499358540350662}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, 0.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.10499358540350662, 0.19661193324148185, 0.25}, []int{3}),
		},
		{
			name:  "Mixed values with alpha",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.5,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.09830596662074093, 0.25, 0.10499358540350662}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SigmoidPrime(tt.input, &tt.params)
			assert.NoError(t, err, "SigmoidPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestRangeSigmoid(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0, 0.4621171572600098, 0.7615941559557646}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, 0.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.7615941559557649, -0.4621171572600098, 0.0}, []int{3}),
		},
		{
			name:  "Mixed values with alpha and signalGain",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.5,
				SignalGain: 2.0,
			},
			expected: tensor.New([]float64{-0.7615941559557649, 0.0, 0.9640275800758169}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := RangeSigmoid(tt.input, &tt.params)
			assert.NoError(t, err, "RangeSigmoid() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestRangeSigmoidPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.5, 0.39322386648296376, 0.209987170807013}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, 0.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.20998717080701304, 0.3932238664829637, 0.5}, []int{3}),
		},
		{
			name:  "Mixed values with alpha and signalGain",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.5,
				SignalGain: 2.0,
			},
			expected: tensor.New([]float64{0.20998717080701304, 1.0, 0.07065082485316446}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := RangeSigmoidPrime(tt.input, &tt.params)
			assert.NoError(t, err, "RangeSigmoidPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestReLU(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.0, 1.0, 2.0}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, 0.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.2, -0.1, 0.0}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.2, 0.0, 2.0}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ReLU(tt.input, &tt.params)
			assert.NoError(t, err, "ReLU() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestReLUPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.1, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{1.0, 1.0, 1.0}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, -0.1}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.1, 0.1, 0.1}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.1, 0.1, 1.0}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := ReLUPrime(tt.input, &tt.params)
			assert.NoError(t, err, "ReLUPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestTanh(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.1, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.09966799462495582, 0.7615941559557649, 0.9640275800758169}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, -0.1}, []int{3}),
			params: ActivationParams{
				Alpha:      1.5,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.9950547536867305, -0.9051482536448664, -0.148885033623318}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.2,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.9836748576936802, 0.0, 0.9640275800758169}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Tanh(tt.input, &tt.params)
			assert.NoError(t, err, "Tanh() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestTanhPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.1, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.9900663297653179, 0.41997434161402614, 0.07065082485316443}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, -0.1}, []int{3}),
			params: ActivationParams{
				Alpha:      1.5,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.014799055748160317, 0.27105995838547287, 1.4667498701444752}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.2,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.038860529209581475, 1.0, 0.07065082485316443}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := TanhPrime(tt.input, &tt.params)
			assert.NoError(t, err, "TanhPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestSwish(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.1, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.052497918747894, 0.7310585786300049, 1.7615941559557646}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, -0.1}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.2384058440442351, -0.2689414213699951, -0.047502081252106004}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.23840584404423526, 0.0, 1.7615941559557649}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Swish(tt.input, &tt.params)
			assert.NoError(t, err, "Swish() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestSwishPrime(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Positive values",
			input: tensor.New([]float64{0.1, 1.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.5499167914982293, 0.9276705118714866, 1.0907842487848955}, []int{3}),
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-2.0, -1.0, -0.1}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.09078424878489545, 0.07232948812851328, 0.45008320850177086}, []int{3}),
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-2.0, 0.0, 2.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.09078424878489545, 0.5, 1.0907842487848955}, []int{3}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SwishPrime(tt.input, &tt.params)
			assert.NoError(t, err, "SwishPrime() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestSoftmax(t *testing.T) {
	tests := []struct {
		name      string
		input     tensor.Tensor
		params    ActivationParams
		expected  tensor.Tensor
		expectErr bool
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New(nil, nil),
			expectErr: true,
		},
		{
			name:  "Single value",
			input: tensor.New([]float64{1.0}, []int{1}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{1.0}, []int{1}),
			expectErr: false,
		},
		{
			name:  "Multiple values",
			input: tensor.New([]float64{1.0, 2.0, 3.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748219}, []int{3}),
			expectErr: false,
		},
		{
			name:  "Same values",
			input: tensor.New([]float64{1.0, 1.0, 1.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.3333333333333333, 0.3333333333333333, 0.3333333333333333}, []int{3}),
			expectErr: false,
		},
		{
			name:  "Batch dimension",
			input: tensor.New([]float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, []int{2, 3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.09003057317038043, 0.24472847105479759, 0.6652409557748217, 0.09003057317038043, 0.24472847105479759, 0.6652409557748217}, []int{2, 3}),
			expectErr: false,
		},
		{
			name:  "Batch dimension with same values",
			input: tensor.New([]float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, []int{2, 3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.33333333333333337, 0.33333333333333337, 0.33333333333333337, 0.33333333333333337, 0.33333333333333337, 0.33333333333333337}, []int{2, 3}),
			expectErr: false,
		},
		{
			name: "Sequence and batch dimension",
			input: tensor.New([]float64{
				1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4,
				1, 2, 3, 4, 1, 2, 3, 4,
			}, []int{3, 2, 4}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
				0.032058603280084974, 0.08714431874203253, 0.23688281808991005, 0.643914259887972,
			}, []int{3, 2, 4}),
			expectErr: false,
		},
		{
			name:  "Negative values",
			input: tensor.New([]float64{-1.0, -2.0, -3.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.6652409557748219, 0.24472847105479764, 0.09003057317038046}, []int{3}),
			expectErr: false,
		},
		{
			name:  "Mixed values",
			input: tensor.New([]float64{-1.0, 0.0, 1.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.09003057317038046, 0.24472847105479764, 0.6652409557748219}, []int{3}),
			expectErr: false,
		},
		{
			name:  "Large values",
			input: tensor.New([]float64{100.0, 200.0, 300.0}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected:  tensor.New([]float64{0.0, 0.0, 1.0}, []int{3}),
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Softmax(tt.input, &tt.params)
			if tt.expectErr {
				assert.Error(t, err, "Softmax() should return an error")
			} else {
				assert.NoError(t, err, "Softmax() should not return an error")
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestSoftmaxPrimePartial(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
	}{
		{
			name:  "Empty tensor",
			input: tensor.New(nil, nil),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New(nil, nil),
		},
		{
			name:  "Single value",
			input: tensor.New([]float64{0.5}, []int{1}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.25}, []int{1}),
		},
		{
			name:  "Multiple values",
			input: tensor.New([]float64{0.1, 0.2, 0.7}, []int{3}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.09, 0.16, 0.21}, []int{3}),
		},
		{
			name:  "Values close to 0 and 1",
			input: tensor.New([]float64{0.01, 0.99}, []int{2}),
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.0099, 0.0099}, []int{2}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := SoftmaxPrimePartial(tt.input, &tt.params)
			assert.NoError(t, err, "SoftmaxPrimePartial() should not return an error")
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}

func TestLinear(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name:  "1D input on 2D weights",
			input: tensor.New([]float64{1, 2, 3}, []int{3}),
			params: ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3}),
				Bias:    tensor.New([]float64{0.1, 0.2}, []int{2}),
			},
			expected: tensor.New([]float64{1.5, 3.4}, []int{2}),
			wantErr:  false,
		},
		{
			name:  "Input with batch size 2",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
			params: ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
				Bias:    tensor.New([]float64{0.1, 0.2}, []int{2}),
			},
			expected: tensor.New([]float64{14.1, 32.2, 32.1, 77.2}, []int{2, 2}),
			wantErr:  false,
		},
		{
			name:  "Input with batch size 3",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, []int{3, 5}),
			params: ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{2, 5}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3}, []int{2}),
			},
			expected: tensor.New([]float64{55.1, 55.2, 130.1, 130.2, 205.1, 205.2}, []int{3, 2}),
			wantErr:  false,
		},
		{
			name:  "Input with length and batch dimensions",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 2, 5}),
			params: ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, []int{4, 5}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{4}),
			},
			expected: tensor.New([]float64{35.1, 34.2, 37.3, 44.4, 35.1, 34.2, 37.3, 44.4, 35.1, 34.2, 37.3, 44.4, 35.1, 34.2, 37.3, 44.4, 35.1, 34.2, 37.3, 44.4, 35.1, 34.2, 37.3, 44.4}, []int{3, 2, 4}),
			wantErr:  false,
		},
		{
			name:  "Sequence dimension outputs should be independent",
			input: tensor.New([]float64{5, 6, 5, 6}, []int{2, 1, 2}),
			params: ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3}, []int{3}),
			},
			expected: tensor.New([]float64{17.1, 39.2, 61.3, 17.1, 39.2, 61.3}, []int{2, 1, 3}),
			wantErr:  false,
		},
		{
			name:  "Batch dimension outputs should be independent",
			input: tensor.New([]float64{5, 6, 5, 6}, []int{1, 2, 2}),
			params: ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3}, []int{3}),
			},
			expected: tensor.New([]float64{17.1, 39.2, 61.3, 17.1, 39.2, 61.3}, []int{1, 2, 3}),
			wantErr:  false,
		},
		{
			name:  "Incompatible dimensions",
			input: tensor.New([]float64{1, 2, 3, 4}, []int{4}),
			params: ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3}, []int{1, 3}),
				Bias:    tensor.New([]float64{0.1}, []int{1}),
			},
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Linear(tt.input, &tt.params)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestLinearGrad(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name:  "1D input",
			input: tensor.New([]float64{1, 2}, []int{2}),
			params: ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3}),
				Bias:    tensor.New([]float64{0.1, 0.2}, []int{2}),
			},
			expected: tensor.New([]float64{0.9, 1.2, 1.5}, []int{3}),
			wantErr:  false,
		},
		{
			name:  "Input with batch size 2",
			input: tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
			params: ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{2, 3}),
				Bias:    tensor.New([]float64{0.1, 0.2}, []int{2}),
			},
			expected: tensor.New([]float64{0.9, 1.2, 1.5, 1.9000000000000001, 2.6, 3.3}, []int{2, 3}),
			wantErr:  false,
		},
		{
			name:  "Incompatible dimensions",
			input: tensor.New([]float64{1, 2, 3}, []int{3}),
			params: ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{2, 2}),
				Bias:    tensor.New([]float64{0.1, 0.2}, []int{2}),
			},
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := LinearGrad(tt.input, &tt.params)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestConvolution(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name:  "1D convolution",
			input: tensor.New([]float64{1, 2, 3, 4, 5}, []int{1, 1, 1, 5}),
			params: ActivationParams{
				Kernel:  tensor.New([]float64{-1, 0, 1}, []int{1, 1, 1, 3}),
				Padding: []int{0, 1},
				Stride:  []int{1, 1},
			},
			expected: tensor.New([]float64{2, 2, 2, 2, -4}, []int{1, 1, 1, 5}),
			wantErr:  false,
		},
		{
			name:  "2D convolution",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 1, 3, 3}),
			params: ActivationParams{
				Kernel:  tensor.New([]float64{1, 0, -1, 2, 0, -2, 1, 0, -1}, []int{1, 1, 3, 3}),
				Padding: []int{1, 1},
				Stride:  []int{1, 1},
			},
			expected: tensor.New([]float64{-9, -6, 9, -20, -8, 20, -21, -6, 21}, []int{1, 1, 3, 3}),
			wantErr:  false,
		},
		{
			name:  "2D convolution with batch size 2",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 1, 3, 3}),
			params: ActivationParams{
				Kernel:  tensor.New([]float64{1, 0, -1, 2, 0, -2, 1, 0, -1}, []int{1, 1, 3, 3}),
				Padding: []int{1, 1},
				Stride:  []int{1, 1},
			},
			expected: tensor.New([]float64{-9, -6, 9, -20, -8, 20, -21, -6, 21, -9, -6, 9, -20, -8, 20, -21, -6, 21}, []int{2, 1, 3, 3}),
			wantErr:  false,
		},
		{
			name:  "Invalid kernel dimensions",
			input: tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
			params: ActivationParams{
				Kernel:  tensor.New([]float64{1, 2, 3}, []int{3}),
				Padding: []int{0, 0},
				Stride:  []int{1, 1},
			},
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Convolution(tt.input, &tt.params)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestFlatten(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name:     "Flatten 2D tensor",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
			expected: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
			wantErr:  false,
		},
		{
			name:     "Flatten 3D tensor",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 3, 3}),
			expected: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 9}),
			wantErr:  false,
		},
		{
			name:     "Flatten 3D tensor with batch size 2",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 3, 3}),
			expected: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 9}),
			wantErr:  false,
		},
		{
			name:     "Flatten 4D tensor",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{1, 2, 2, 3}),
			expected: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{1, 12}),
			wantErr:  false,
		},
		{
			name:     "Invalid input tensor",
			input:    tensor.New([]float64{1, 2, 3, 4}, []int{4}),
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Flatten(tt.input, nil)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestMaskedAveragePool(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		mask     tensor.Tensor
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name: "3D mask averages active timesteps",
			input: tensor.New([]float64{
				1, 10,
				2, 20,
				3, 30,
				4, 40,
				5, 50,
				6, 60,
			}, []int{3, 2, 2}),
			mask: tensor.New([]float64{
				1, 1,
				1, 0,
				0, 1,
			}, []int{3, 2, 1}),
			expected: tensor.New([]float64{
				2, 20,
				4, 40,
			}, []int{2, 2}),
			wantErr: false,
		},
		{
			name: "2D mask averages active timesteps",
			input: tensor.New([]float64{
				1, 2,
				3, 4,
				5, 6,
			}, []int{3, 1, 2}),
			mask: tensor.New([]float64{
				1,
				0,
				1,
			}, []int{3, 1}),
			expected: tensor.New([]float64{
				3, 4,
			}, []int{1, 2}),
			wantErr: false,
		},
		{
			name:    "invalid input rank",
			input:   tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
			mask:    tensor.New([]float64{1, 1}, []int{2, 1}),
			wantErr: true,
		},
		{
			name: "invalid mask shape",
			input: tensor.New([]float64{
				1, 2,
				3, 4,
				5, 6,
			}, []int{3, 1, 2}),
			mask: tensor.New([]float64{
				1, 1,
				1, 1,
			}, []int{2, 2}),
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := MaskedAveragePool(tt.input, tt.mask)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}

			assert.NoError(t, err)
			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
		})
	}
}

func TestLayerNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    tensor.Tensor
		params   ActivationParams
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name:     "LayerNormalization 1D tensor",
			input:    tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			params:   ActivationParams{Gamma: 1.0, Beta: 0.0},
			expected: tensor.New([]float64{-1.414213558837561, -0.7071067794187805, 0, 0.7071067794187805, 1.414213558837561}, []int{5}),
			wantErr:  false,
		},
		{
			name:     "LayerNormalization with batch dimension",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
			params:   ActivationParams{Gamma: 1.0, Beta: 0.0},
			expected: tensor.New([]float64{-1.2247448622060026, 0, 1.2247448622060026, -1.2247448622060026, 0, 1.2247448622060026}, []int{2, 3}),
			wantErr:  false,
		},
		{
			name:     "LayerNormalization with batch dimension and 2 feature dimensions",
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, []int{2, 3, 3}),
			params:   ActivationParams{Gamma: 1.0, Beta: 0.0},
			expected: tensor.New([]float64{-1.2247448690951925, -0.6123724345475963, 0, 0.6123724345475963, 1.2247448690951925, 1.8371173036427888, -1.2247448690951925, -0.6123724345475963, 0, 0, 0.6123724345475963, 1.2247448690951925, -1.8371173036427888, -1.2247448690951925, -0.6123724345475963, 0, 0.6123724345475963, 1.2247448690951925}, []int{2, 3, 3}),
			wantErr:  false,
		},
		{
			name:     "LayerNormalization with Gamma and Beta",
			input:    tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			params:   ActivationParams{Gamma: 2.0, Beta: 1.0},
			expected: tensor.New([]float64{-1.8284271176751221, -0.41421355883756106, 1, 2.4142135588375613, 3.828427117675122}, []int{5}),
			wantErr:  false,
		},
		{
			name:     "Invalid input tensor",
			input:    tensor.Tensor{},
			params:   ActivationParams{Gamma: 1.0, Beta: 0.0},
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := LayerNormalization(tt.input, &tt.params)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestHiddenLstm(t *testing.T) {
	tests := []struct {
		name        string
		input       tensor.Tensor
		params      *ActivationParams
		expected    tensor.Tensor
		expectError bool
	}{
		{
			name:  "Basic LSTM test",
			input: tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{2, 1, 2}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8, 4}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8}),
			},
			expected:    tensor.New([]float64{0.17841275138284804, 0.26466225453039505, 0.36510584553571956, 0.5928499573683099}, []int{2, 1, 2}),
			expectError: false,
		},
		{
			name:  "LSTM with batch size > 1",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, []int{2, 2, 3}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
					0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4,
				}, []int{12, 6}),
				Bias: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4}, []int{12}),
			},
			expected:    tensor.New([]float64{0.6224867942774628, 0.6218805158367184, 0.6136560902960748, 0.7455224785510732, 0.7390660791413309, 0.7475399940738602, 0.9458645603047922, 0.9575698952065055, 0.9428744091593303, 0.9613414948217822, 0.9637161561672566, 0.9616790621847386}, []int{2, 2, 3}),
			expectError: false,
		},
		{
			name: "LSTM with padding values on seq_len",
			input: tensor.New([]float64{
				0.1, 0.2,
				0.3, 0.4,
				0, 0,
				0, 0,
			}, []int{4, 1, 2}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8, 4}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8}),
			},
			expected:    tensor.New([]float64{0.17841275138284804, 0.26466225453039505, 0.36510584553571956, 0.5928499573683099, 0.4847970737304926, 0.7108124331818988, 0.572027519415736, 0.7933815094396169}, []int{4, 1, 2}),
			expectError: false,
		},
		{
			name:        "LSTM with nil params",
			input:       tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 1, 3}),
			params:      nil,
			expectError: true,
		},
		{
			name:  "LSTM with empty weights",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 1, 3}),
			params: &ActivationParams{
				Weights: tensor.Tensor{},
				Bias:    tensor.Tensor{},
			},
			expectError: true,
		},
		{
			name:        "LSTM with invalid input shape",
			input:       tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 1, 3, 4}),
			params:      &ActivationParams{},
			expectError: true,
		},
		{
			name:  "LSTM with invalid input features",
			input: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 1, 3}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8, 4}),
				Bias:    tensor.New([]float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, []int{8}),
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := HiddenLstm(tt.input, tt.params)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-4), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestEmbedding(t *testing.T) {
	tests := []struct {
		name        string
		input       tensor.Tensor
		params      *ActivationParams
		expected    tensor.Tensor
		expectError bool
	}{
		{
			name:  "Basic embedding",
			input: tensor.New([]float64{1, 2, 0}, []int{3, 1, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2,
					0.3, 0.4,
					0.5, 0.6,
				}, []int{3, 2}),
			},
			expected:    tensor.New([]float64{0.3, 0.4, 0.5, 0.6, 0.1, 0.2}, []int{3, 1, 2}),
			expectError: false,
		},
		{
			name:  "Embedding with batch size > 1",
			input: tensor.New([]float64{1, 2, 0, 2}, []int{2, 2, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2, 0.3,
					0.4, 0.5, 0.6,
					0.7, 0.8, 0.9,
				}, []int{3, 3}),
			},
			expected:    tensor.New([]float64{0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9}, []int{2, 2, 3}),
			expectError: false,
		},
		{
			name:  "Embedding with seq_len > 1 and batch_size > 1",
			input: tensor.New([]float64{1, 2, 0, 2, 1, 0}, []int{3, 2, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2, 0.3,
					0.4, 0.5, 0.6,
					0.7, 0.8, 0.9,
				}, []int{3, 3}),
			},
			expected: tensor.New([]float64{
				0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
				0.1, 0.2, 0.3, 0.7, 0.8, 0.9,
				0.4, 0.5, 0.6, 0.1, 0.2, 0.3,
			}, []int{3, 2, 3}),
			expectError: false,
		},
		{
			name:  "Embedding with -1 index input",
			input: tensor.New([]float64{1, -1, 0}, []int{3, 1, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2,
					0.3, 0.4,
					0.5, 0.6,
				}, []int{3, 2}),
			},
			expected:    tensor.New([]float64{0.3, 0.4, 0.0, 0.0, 0.1, 0.2}, []int{3, 1, 2}),
			expectError: false,
		},
		{
			name:  "Batch embeddings with -1 index input",
			input: tensor.New([]float64{1, -1, 0, 2, 0, -1}, []int{3, 2, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{
					0.1, 0.2,
					0.3, 0.4,
					0.5, 0.6,
				}, []int{3, 2}),
			},
			expected: tensor.New([]float64{
				0.3, 0.4, 0.0, 0.0, 0.1, 0.2,
				0.5, 0.6, 0.1, 0.2, 0.0, 0.0,
			}, []int{3, 2, 2}),
			expectError: false,
		},
		{
			name:  "Invalid input shape",
			input: tensor.New([]float64{1, 2, 0}, []int{3}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{2, 2}),
			},
			expectError: true,
		},
		{
			name:  "Invalid input feature dimension",
			input: tensor.New([]float64{1, 2, 0, 1, 2, 0}, []int{3, 1, 2}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{2, 2}),
			},
			expectError: true,
		},
		{
			name:  "Out of range index",
			input: tensor.New([]float64{1, 2, 3}, []int{3, 1, 1}),
			params: &ActivationParams{
				Weights: tensor.New([]float64{0.1, 0.2, 0.3, 0.4}, []int{2, 2}),
			},
			expectError: true,
		},
		{
			name:        "Nil params",
			input:       tensor.New([]float64{1, 2, 0}, []int{3, 1, 1}),
			params:      nil,
			expectError: true,
		},
		{
			name:  "Empty weights",
			input: tensor.New([]float64{1, 2, 0}, []int{3, 1, 1}),
			params: &ActivationParams{
				Weights: tensor.Tensor{},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Embedding(tt.input, tt.params)
			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestApplyActivationFunction(t *testing.T) {
	tests := []struct {
		name           string
		input          tensor.Tensor
		activationType ActivationType
		params         ActivationParams
		expected       tensor.Tensor
	}{
		{
			name:           "ReLU activation",
			input:          tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}),
			activationType: ActivationReLU,
			params: ActivationParams{
				Alpha:      0.1,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.2, -0.1, 0, 1, 2}, []int{5}),
		},
		{
			name:           "Sigmoid activation",
			input:          tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}),
			activationType: ActivationSigmoid,
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.119203, 0.268941, 0.5, 0.731059, 0.880797}, []int{5}),
		},
		{
			name:           "TanH activation",
			input:          tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}),
			activationType: ActivationTanH,
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{-0.964028, -0.761594, 0, 0.761594, 0.964028}, []int{5}),
		},
		{
			name:           "Softmax activation",
			input:          tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			activationType: ActivationSoftmax,
			params: ActivationParams{
				Alpha:      1.0,
				SignalGain: 1.0,
			},
			expected: tensor.New([]float64{0.011656230956039609, 0.03168492079612427, 0.0861285444362687, 0.23412165725273662, 0.6364086465588308}, []int{5}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			activationFn, err := GetActivationFunction(tt.activationType)
			if err != nil {
				assert.Fail(t, "Failed to get activation function: %v", err)
			}

			result, err := ActivationForward(tt.input, activationFn, &tt.params)
			assert.NoError(t, err, "ActivationForward() should not return an error")

			assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result.Values())
		})
	}
}
