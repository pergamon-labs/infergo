package module

import (
	"os"
	"testing"

	"github.com/minervaai/infergo/backends/bionet/runtime/functional"
	"github.com/minervaai/infergo/backends/bionet/runtime/initializer"
	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
	"github.com/stretchr/testify/assert"
)

func TestCopy(t *testing.T) {
	// Create a test module with submodules and activation function
	originalModule := New(functional.ActivationReLU, functional.ActivationParams{
		Alpha:      0.1,
		SignalGain: 1.0,
		Weights:    tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
		Bias:       tensor.New([]float64{5, 6}, []int{2}),
		Padding:    []int{1, 2},
		Stride:     []int{3, 4},
	},
		New(functional.ActivationSigmoid, functional.ActivationParams{}),
		New(functional.ActivationTanH, functional.ActivationParams{}),
	)

	// Set the mode to train
	originalModule.Train()
	originalModule.ModuleType = ModuleTypeActivation

	// Create a copy of the module
	copiedModule := Copy(originalModule)

	// Test that the copied module is not the same instance as the original
	assert.NotSame(t, copiedModule, originalModule, "Copied module should not be the same instance as the original")

	// Test that the mode is copied correctly
	assert.Equal(t, originalModule.Mode, copiedModule.Mode, "Copied module mode should match original module mode")
	assert.Equal(t, originalModule.ModuleType, copiedModule.ModuleType, "Copied module type should match original module type")
	assert.Equal(t, originalModule.ActivationType, copiedModule.ActivationType, "Copied activation type should match")

	// Test that the activation function is copied correctly
	assert.NotNil(t, copiedModule.Activation, "Copied module should have an activation function")

	// Test that the activation parameters are copied correctly
	assert.Equal(t, originalModule.Params, copiedModule.Params, "Activation parameters should be copied correctly")
	assert.NotNil(t, copiedModule.Params.Padding, "Padding should be present on copy")
	assert.NotNil(t, copiedModule.Params.Stride, "Stride should be present on copy")
	assert.NotEmpty(t, copiedModule.Params.Weights.Values(), "Weights should be present on copy")
	assert.NotEmpty(t, copiedModule.Params.Bias.Values(), "Bias should be present on copy")
	assert.False(t, &originalModule.Params.Padding[0] == &copiedModule.Params.Padding[0], "Padding should be deep copied")
	assert.False(t, &originalModule.Params.Stride[0] == &copiedModule.Params.Stride[0], "Stride should be deep copied")
	assert.False(t, &originalModule.Params.Weights.Values()[0] == &copiedModule.Params.Weights.Values()[0], "Weights should be deep copied")
	assert.False(t, &originalModule.Params.Bias.Values()[0] == &copiedModule.Params.Bias.Values()[0], "Bias should be deep copied")

	originalModule.Params.Padding[0] = 99
	originalModule.Params.Stride[0] = 88
	originalModule.Params.Weights.SetFlatValue(0, 77)
	originalModule.Params.Bias.SetFlatValue(0, 66)

	assert.Equal(t, []int{1, 2}, copiedModule.Params.Padding, "Copied padding should not change when original is mutated")
	assert.Equal(t, []int{3, 4}, copiedModule.Params.Stride, "Copied stride should not change when original is mutated")
	assert.Equal(t, float64(1), copiedModule.Params.Weights.GetFlatValue(0), "Copied weights should not change when original is mutated")
	assert.Equal(t, float64(5), copiedModule.Params.Bias.GetFlatValue(0), "Copied bias should not change when original is mutated")

	// Test that the number of submodules is correct
	assert.Equal(t, len(originalModule.ModuleList), len(copiedModule.ModuleList), "Number of submodules should match")

	// Test that each submodule is copied correctly
	for i := range originalModule.ModuleList {
		assert.NotNil(t, copiedModule.ModuleList[i].Activation, "Copied submodule should have an activation function")
		assert.Equal(t, originalModule.ModuleList[i].Params, copiedModule.ModuleList[i].Params, "Submodule activation parameters should be copied correctly")
	}
}

func TestForward(t *testing.T) {
	tests := []struct {
		name     string
		module   *Module
		input    tensor.Tensor
		expected tensor.Tensor
		wantErr  bool
	}{
		{
			name: "Single module with ReLU activation",
			module: New(functional.ActivationReLU, functional.ActivationParams{
				Alpha:      0.0,
				SignalGain: 1.0,
			}),
			input:    tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}),
			expected: tensor.New([]float64{0, 0, 0, 1, 2}, []int{5}),
			wantErr:  false,
		},
		{
			name: "Multiple modules with different activations",
			module: New(functional.ActivationReLU, functional.ActivationParams{
				Alpha:      0.0,
				SignalGain: 1.0,
			},
				New(functional.ActivationSigmoid, functional.ActivationParams{
					Alpha: 1.0,
				}),
				New(functional.ActivationTanH, functional.ActivationParams{
					Alpha: 1.0,
				}),
			),
			input:    tensor.New([]float64{-2, -1, 0, 1, 2}, []int{5}),
			expected: tensor.New([]float64{0.11864151454914097, 0.2626395514040159, 0.46211715726000974, 0.6237125498258758, 0.7068184091418055}, []int{5}),
			wantErr:  false,
		},
		{
			name: "Linear module with bias",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5}, []int{1, 5}),
					Bias:    tensor.New([]float64{0.1}, []int{1}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			expected: tensor.New([]float64{55.1}, []int{1}),
			wantErr:  false,
		},
		{
			name: "Linear module with batch dimension",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5}, []int{1, 5}),
					Bias:    tensor.New([]float64{0.1}, []int{1}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 5}),
			expected: tensor.New([]float64{55.1, 55.1, 55.1}, []int{3, 1}),
			wantErr:  false,
		},
		{
			name: "Sequential linear modules",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 5}),
					Bias:    tensor.New([]float64{0.1, 0.1, 0.1}, []int{3}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5}, []int{1, 3}),
					Bias:    tensor.New([]float64{0.5}, []int{1}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			expected: tensor.New([]float64{551.5}, []int{1}),
			wantErr:  false,
		},
		{
			name: "Sequential linear modules, softmax activation",
			module: New(functional.ActivationSoftmax, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 5}),
					Bias:    tensor.New([]float64{0.1, 0.1, 0.1}, []int{3}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
					Bias:    tensor.New([]float64{0.5, 0.5}, []int{2}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 4, 5}, []int{5}),
			expected: tensor.New([]float64{0, 1}, []int{2}),
			wantErr:  false,
		},
		{
			name: "Sequential linear modules, batch dimension",
			module: New(functional.ActivationSoftmax, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 5}),
					Bias:    tensor.New([]float64{0.1, 0.1, 0.1}, []int{3}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 3}),
					Bias:    tensor.New([]float64{0.5, 0.5}, []int{2}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5}, []int{3, 5}),
			expected: tensor.New([]float64{0, 1, 0, 1, 0, 1}, []int{3, 2}),
			wantErr:  false,
		},
		{
			name: "Sequential conv and linear modules",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationConvolution, functional.ActivationParams{
					Kernel:  tensor.New([]float64{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, []int{4, 2, 3, 3}),
					Stride:  []int{1, 1},
					Padding: []int{1, 1},
				}),
				New(functional.ActivationFlatten, functional.ActivationParams{}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{2, 36}),
					Bias:    tensor.New([]float64{0.5, 0.5}, []int{2}),
				}),
			),
			input:    tensor.New([]float64{1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3}, []int{2, 2, 3, 3}),
			expected: tensor.New([]float64{8400.5, 8400.5, 8400.5, 8400.5}, []int{2, 2}),
			wantErr:  false,
		},
		{
			name: "Sequential linear and LSTM modules",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
					}, []int{3, 4}),
					Bias: tensor.New([]float64{0.2, 0.2, 0.2}, []int{3}),
				}),
				New(functional.ActivationHiddenLstm, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
					}, []int{12, 6}),
					Bias: tensor.New([]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, []int{12}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2,
						0.1, 0.2,
						0.1, 0.2,
					}, []int{2, 3}),
					Bias: tensor.New([]float64{0.5, 0.5}, []int{2}),
				}),
			),
			input: tensor.New([]float64{
				0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
				0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
				0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4,
			}, []int{3, 2, 4}),
			expected: tensor.New([]float64{
				0.5902174448695583, 0.5879217610082863, 0.5902174448695583, 0.5879217610082863, 0.6914238326222516, 0.7202860237417231, 0.6914238326222516, 0.7202860237417231, 0.7860104541100632, 0.8629330991352929, 0.7860104541100632, 0.8629330991352929,
			}, []int{3, 2, 2}),
			wantErr: false,
		},
		{
			name: "Sequential embedding, linear, and LSTM modules",
			module: New(functional.ActivationNone, functional.ActivationParams{},
				New(functional.ActivationEmbedding, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2, 0.3, 0.4,
						0.2, 0.3, 0.4, 0.5,
						0.3, 0.4, 0.5, 0.6,
						0.4, 0.5, 0.6, 0.7,
						0.5, 0.6, 0.7, 0.8,
						0.6, 0.7, 0.8, 0.9,
					}, []int{6, 4}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
						0.1, 0.2, 0.3,
					}, []int{3, 4}),
					Bias: tensor.New([]float64{0.2, 0.2, 0.2}, []int{3}),
				}),
				New(functional.ActivationHiddenLstm, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
						0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
					}, []int{12, 6}),
					Bias: tensor.New([]float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, []int{12}),
				}),
				New(functional.ActivationLinear, functional.ActivationParams{
					Weights: tensor.New([]float64{
						0.1, 0.2,
						0.1, 0.2,
						0.1, 0.2,
					}, []int{2, 3}),
					Bias: tensor.New([]float64{0.5, 0.5}, []int{2}),
				}),
			),
			input: tensor.New([]float64{
				0, 1, 2, 3, 4, 5,
			}, []int{3, 2, 1}),
			expected: tensor.New([]float64{
				0.5902174448695583, 0.5879217610082863, 0.6037100729488161, 0.6034365709062814, 0.7111681408071815, 0.7492561300056588, 0.7317312858850722, 0.7795484667382031, 0.8171080404068214, 0.9069621659608651, 0.8304130573101779, 0.9253515096486655,
			}, []int{3, 2, 2}),
			wantErr: false,
		},
		{
			name: "Incompatible dimensions in input",
			module: New(functional.ActivationLinear, functional.ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3}, []int{1, 3}),
				Bias:    tensor.New([]float64{0.1}, []int{1}),
			}),
			input:    tensor.New([]float64{1, 2, 3, 4}, []int{4}),
			expected: tensor.Tensor{},
			wantErr:  true,
		},
		{
			name: "Incompatible linear dimensions in weights",
			module: New(functional.ActivationLinear, functional.ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}),
				Bias:    tensor.New([]float64{0.1, 0.1}, []int{2}),
			}),
			input:    tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
			expected: tensor.Tensor{},
			wantErr:  true,
		},
		{
			name: "Incompatible linear dimensions in bias",
			module: New(functional.ActivationLinear, functional.ActivationParams{
				Weights: tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{2, 2}),
				Bias:    tensor.New([]float64{0.1, 0.1, 0.1}, []int{3}),
			}),
			input:    tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}),
			expected: tensor.Tensor{},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.module.Forward(tt.input)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.True(t, tensor.InDelta(tt.expected, result, 1e-6), "Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestEval(t *testing.T) {
	module := New(functional.ActivationLinear, functional.ActivationParams{})
	subModule1 := New(functional.ActivationSigmoid, functional.ActivationParams{})
	subModule2 := New(functional.ActivationReLU, functional.ActivationParams{})
	module.ModuleList = append(module.ModuleList, subModule1, subModule2)

	module.Train() // Set to train mode initially
	module.Eval()

	assert.Equal(t, ModuleModeEval, module.Mode, "Main module should be in eval mode")
	assert.Equal(t, ModuleModeEval, subModule1.Mode, "Submodule 1 should be in eval mode")
	assert.Equal(t, ModuleModeEval, subModule2.Mode, "Submodule 2 should be in eval mode")
}

func TestTrain(t *testing.T) {
	module := New(functional.ActivationLinear, functional.ActivationParams{})
	subModule1 := New(functional.ActivationSigmoid, functional.ActivationParams{})
	subModule2 := New(functional.ActivationReLU, functional.ActivationParams{})
	module.ModuleList = append(module.ModuleList, subModule1, subModule2)

	module.Eval() // Set to eval mode initially
	module.Train()

	assert.Equal(t, ModuleModeTrain, module.Mode, "Main module should be in train mode")
	assert.Equal(t, ModuleModeTrain, subModule1.Mode, "Submodule 1 should be in train mode")
	assert.Equal(t, ModuleModeTrain, subModule2.Mode, "Submodule 2 should be in train mode")
}

func TestSaveAndLoadFromFile(t *testing.T) {
	// Create a temporary file
	tempFile, err := os.CreateTemp("", "module_test")
	assert.NoError(t, err)
	defer os.Remove(tempFile.Name())

	// Create a test module with submodules
	module := New(functional.ActivationNone, functional.ActivationParams{},
		FullyConnected(3, 4, initializer.KaimingUniform),
		Activation(functional.ActivationReLU, functional.ActivationParams{Alpha: 1.0}),
		HiddenLstm(4, initializer.KaimingUniform),
		FullyConnected(4, 2, initializer.KaimingUniform),
		Activation(functional.ActivationSoftmax, functional.ActivationParams{Alpha: 1.0}),
	)

	// Create input tensor
	input := tensor.New([]float64{
		1, 2, 3, 4, 1, 2, 3, 4,
		1, 2, 3, 4, 1, 2, 3, 4,
		1, 2, 3, 4, 1, 2, 3, 4,
	}, []int{3, 2, 3})

	// Perform forward pass before saving
	outputBefore, err := module.Forward(input)
	assert.NoError(t, err)

	// Save the module to file
	err = SaveToFile(module, tempFile.Name())
	assert.NoError(t, err)

	// Load the module from file
	loadedModule, err := LoadFromFile(tempFile.Name())
	assert.NoError(t, err)

	// Perform forward pass after loading
	outputAfter, err := loadedModule.Forward(input)
	assert.NoError(t, err)

	// Compare the outputs
	assert.True(t, tensor.InDelta(outputBefore, outputAfter, 1e-8), "Output before and after should be the same. Got %v and %v", outputBefore, outputAfter)

	// Compare the loaded module with the original
	assert.Equal(t, module.ActivationType, loadedModule.ActivationType)
	assert.NotNil(t, loadedModule.ActivationType)
	assert.NotNil(t, loadedModule.Activation)
	assert.Equal(t, module.ModuleType, loadedModule.ModuleType)
	assert.Equal(t, module.Mode, loadedModule.Mode)
	assert.True(t, tensor.Equal(module.Params.Weights, loadedModule.Params.Weights))
	assert.True(t, tensor.Equal(module.Params.Bias, loadedModule.Params.Bias))
	assert.Equal(t, len(module.ModuleList), len(loadedModule.ModuleList))

	for i := range module.ModuleList {
		assert.Equal(t, module.ModuleList[i].ActivationType, loadedModule.ModuleList[i].ActivationType)
		assert.NotNil(t, loadedModule.ModuleList[i].Activation)
		assert.Equal(t, module.ModuleList[i].ModuleType, loadedModule.ModuleList[i].ModuleType)
		assert.Equal(t, module.ModuleList[i].Mode, loadedModule.ModuleList[i].Mode)
		assert.True(t, tensor.Equal(module.ModuleList[i].Params.Weights, loadedModule.ModuleList[i].Params.Weights))
		assert.True(t, tensor.Equal(module.ModuleList[i].Params.Bias, loadedModule.ModuleList[i].Params.Bias))
	}

}

func TestInitialize(t *testing.T) {
	mockInitializerCalled := 0
	mockInitializer := func(t *tensor.Tensor) {
		mockInitializerCalled++
	}

	t.Run("Initialize module with no submodules", func(t *testing.T) {
		module := New(functional.ActivationLinear, functional.ActivationParams{
			Weights: tensor.Zeros([]int{2, 2}),
			Bias:    tensor.Zeros([]int{2}),
		})

		mockInitializerCalled = 0
		module.Initialize(mockInitializer)

		assert.Equal(t, 3, mockInitializerCalled, "Initializer should be called twice (for weights and bias)")
	})

	t.Run("Initialize module with submodules", func(t *testing.T) {
		module := New(functional.ActivationLinear, functional.ActivationParams{
			Weights: tensor.Zeros([]int{2, 2}),
			Bias:    tensor.Zeros([]int{2}),
		})
		subModule1 := New(functional.ActivationSigmoid, functional.ActivationParams{
			Weights: tensor.Zeros([]int{3, 3}),
			Bias:    tensor.Zeros([]int{3}),
		})
		subModule2 := New(functional.ActivationReLU, functional.ActivationParams{
			Weights: tensor.Zeros([]int{4, 4}),
			Bias:    tensor.Zeros([]int{4}),
		})
		module.ModuleList = append(module.ModuleList, subModule1, subModule2)

		mockInitializerCalled = 0
		module.Initialize(mockInitializer)

		assert.Equal(t, 9, mockInitializerCalled, "Initializer should be called 9 times (3 for each module)")
	})

	t.Run("Initialize with nil initializer", func(t *testing.T) {
		module := New(functional.ActivationLinear, functional.ActivationParams{
			Weights: tensor.Zeros([]int{2, 2}),
			Bias:    tensor.Zeros([]int{2}),
		})

		mockInitializerCalled = 0
		module.Initialize(nil)

		assert.Equal(t, 0, mockInitializerCalled, "Initializer should not be called when it's nil")
	})
}

func TestFullyConnected(t *testing.T) {
	t.Run("Create fully connected module", func(t *testing.T) {
		inputSize := 10
		outputSize := 5
		initializer := initializer.KaimingUniform

		module := FullyConnected(inputSize, outputSize, initializer)

		assert.NotNil(t, module, "Module should not be nil")
		assert.Equal(t, []int{outputSize, inputSize}, module.Params.Weights.Shape(), "Weights shape should match input and output sizes")
		assert.Equal(t, []int{outputSize}, module.Params.Bias.Shape(), "Bias shape should match output size")
	})

	t.Run("Initialize weights and biases", func(t *testing.T) {
		inputSize := 8
		outputSize := 4
		initializer := initializer.KaimingUniform

		module := FullyConnected(inputSize, outputSize, initializer)

		// Check if weights are initialized (not all zeros)
		allZeros := true
		for _, v := range module.Params.Weights.Values() {
			if v != 0 {
				allZeros = false
				break
			}
		}
		assert.False(t, allZeros, "Weights should be initialized and not all zeros")

		// Check if biases are initialized (not all zeros)
		allZeros = true
		for _, v := range module.Params.Bias.Values() {
			if v != 0 {
				allZeros = false
				break
			}
		}
		assert.False(t, allZeros, "Biases should be initialized and not all zeros")
	})

	t.Run("Forward pass", func(t *testing.T) {
		inputSize := 3
		outputSize := 2
		initializer := initializer.KaimingUniform

		module := FullyConnected(inputSize, outputSize, initializer)

		input := tensor.New([]float64{1, 2, 3}, []int{1, inputSize})
		output, err := module.Forward(input)

		assert.NoError(t, err, "Forward pass should not return an error")
		assert.Equal(t, []int{1, outputSize}, output.Shape(), "Output shape should match expected dimensions")
	})
}

func TestConv2D(t *testing.T) {
	t.Run("Create Conv2D module", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := []int{3, 3}
		padding := []int{1, 1}
		stride := []int{1, 1}
		initializer := initializer.KaimingUniform

		module := Conv2D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		assert.NotNil(t, module, "Module should not be nil")
		assert.Equal(t, []int{out_channels, in_channels, kernelSize[0], kernelSize[1]}, module.Params.Kernel.Shape(), "Kernel shape should match specified size")
		assert.Equal(t, padding, module.Params.Padding, "Padding should match specified values")
		assert.Equal(t, stride, module.Params.Stride, "Stride should match specified values")
	})

	t.Run("Initialize kernel", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := []int{3, 3}
		padding := []int{1, 1}
		stride := []int{1, 1}
		initializer := initializer.KaimingUniform

		module := Conv2D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		// Check if kernel is initialized (not all zeros)
		allZeros := true
		for _, v := range module.Params.Kernel.Values() {
			if v != 0 {
				allZeros = false
				break
			}
		}
		assert.False(t, allZeros, "Kernel should be initialized and not all zeros")
	})

	t.Run("Forward pass", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := []int{3, 3}
		padding := []int{1, 1}
		stride := []int{1, 1}
		initializer := initializer.KaimingUniform

		module := Conv2D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		input := tensor.Rand([]int{5, 1, 5, 5}, 0, 1)
		output, err := module.Forward(input)

		assert.NoError(t, err, "Forward pass should not return an error")
		assert.Equal(t, []int{5, 32, 5, 5}, output.Shape(), "Output shape should match expected dimensions")
	})
}

func TestConv1D(t *testing.T) {
	t.Run("Create Conv1D module", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := 3
		padding := 1
		stride := 1
		initializer := initializer.KaimingUniform

		module := Conv1D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		assert.NotNil(t, module, "Module should not be nil")
		assert.Equal(t, []int{out_channels, in_channels, 1, kernelSize}, module.Params.Kernel.Shape(), "Kernel shape should match specified size")
		assert.Equal(t, []int{0, padding}, module.Params.Padding, "Padding should match specified values")
		assert.Equal(t, []int{1, stride}, module.Params.Stride, "Stride should match specified values")
	})

	t.Run("Initialize kernel", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := 3
		padding := 1
		stride := 1
		initializer := initializer.KaimingUniform

		module := Conv1D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		// Check if kernel is initialized (not all zeros)
		allZeros := true
		for _, v := range module.Params.Kernel.Values() {
			if v != 0 {
				allZeros = false
				break
			}
		}
		assert.False(t, allZeros, "Kernel should be initialized and not all zeros")
	})

	t.Run("Forward pass", func(t *testing.T) {
		in_channels := 1
		out_channels := 32
		kernelSize := 3
		padding := 1
		stride := 1
		initializer := initializer.KaimingUniform

		module := Conv1D(in_channels, out_channels, kernelSize, padding, stride, initializer)

		input := tensor.Rand([]int{5, 1, 1, 10}, 0, 1)
		output, err := module.Forward(input)

		assert.NoError(t, err, "Forward pass should not return an error")
		assert.Equal(t, []int{5, 32, 1, 10}, output.Shape(), "Output shape should match expected dimensions")
	})
}

func TestHiddenLstm(t *testing.T) {
	t.Run("Create hidden LSTM module", func(t *testing.T) {
		hiddenSize := 64
		initializer := initializer.KaimingUniform

		module := HiddenLstm(hiddenSize, initializer)

		assert.NotNil(t, module, "Module should not be nil")
		assert.Equal(t, ModuleTypeHiddenLstm, module.ModuleType, "Module type should be HiddenLstm")
		assert.Equal(t, []int{4 * hiddenSize, 2 * hiddenSize}, module.Params.Weights.Shape(), "Weights shape should match specified size")
		assert.Equal(t, []int{4 * hiddenSize}, module.Params.Bias.Shape(), "Bias shape should match specified size")
	})

	t.Run("Initialize weights and bias", func(t *testing.T) {
		hiddenSize := 32
		initializer := initializer.KaimingUniform

		module := HiddenLstm(hiddenSize, initializer)

		// Check if weights are initialized (not all zeros)
		allZerosWeights := true
		for _, v := range module.Params.Weights.Values() {
			if v != 0 {
				allZerosWeights = false
				break
			}
		}
		assert.False(t, allZerosWeights, "Weights should be initialized and not all zeros")

		// Check if bias is initialized (not all zeros)
		allZerosBias := true
		for _, v := range module.Params.Bias.Values() {
			if v != 0 {
				allZerosBias = false
				break
			}
		}
		assert.False(t, allZerosBias, "Bias should be initialized and not all zeros")
	})

	t.Run("Forward pass", func(t *testing.T) {
		hiddenSize := 16
		initializer := initializer.KaimingUniform

		module := HiddenLstm(hiddenSize, initializer)

		// Create a sample input tensor
		// Shape: [seq_len, batch, hidden_size]
		input := tensor.Rand([]int{5, 2, hiddenSize}, 0, 1)
		output, err := module.Forward(input)

		assert.NoError(t, err, "Forward pass should not return an error")
		assert.Equal(t, []int{5, 2, hiddenSize}, output.Shape(), "Output shape should match input shape")
	})
}

func TestEmbedding(t *testing.T) {
	t.Run("Create Embedding module", func(t *testing.T) {
		vocabSize := 1000
		embeddingDim := 50
		embeddingMatrix := tensor.Rand([]int{vocabSize, embeddingDim}, 0, 1)
		initializer := initializer.KaimingUniform

		module := Embedding(embeddingMatrix, initializer)

		assert.NotNil(t, module, "Module should not be nil")
		assert.Equal(t, ModuleTypeEmbedding, module.ModuleType, "Module type should be Embedding")
		assert.Equal(t, []int{vocabSize, embeddingDim}, module.Params.Weights.Shape(), "Weights shape should match embedding matrix")
	})

	t.Run("Initialize embedding weights", func(t *testing.T) {
		vocabSize := 500
		embeddingDim := 30
		embeddingMatrix := tensor.Zeros([]int{vocabSize, embeddingDim})
		initializer := initializer.KaimingUniform

		module := Embedding(embeddingMatrix, initializer)

		// Check if weights are initialized (not all zeros)
		allZeros := true
		for _, v := range module.Params.Weights.Values() {
			if v != 0 {
				allZeros = false
				break
			}
		}
		assert.False(t, allZeros, "Embedding weights should be initialized and not all zeros")
	})

	t.Run("Forward pass", func(t *testing.T) {
		vocabSize := 100
		embeddingDim := 20
		embeddingMatrix := tensor.Rand([]int{vocabSize, embeddingDim}, 0, 1)
		initializer := initializer.KaimingUniform

		module := Embedding(embeddingMatrix, initializer)

		// Create a sample input tensor (indices)
		input := tensor.New([]float64{1, 5, 10, 20, 11, 12}, []int{3, 2, 1})
		output, err := module.Forward(input)

		assert.NoError(t, err, "Forward pass should not return an error")
		assert.Equal(t, []int{3, 2, embeddingDim}, output.Shape(), "Output shape should be [2, 2, embeddingDim]")
	})
}
