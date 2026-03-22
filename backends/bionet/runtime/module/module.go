package module

import (
	"encoding/gob"
	"fmt"
	"os"
	"slices"
	"sync"

	"github.com/minervaai/infergo/backends/bionet/runtime/functional"
	"github.com/minervaai/infergo/backends/bionet/runtime/initializer"
	"github.com/minervaai/infergo/backends/bionet/runtime/tensor"
)

var _ ModuleInterface = (*Module)(nil)

// ModuleMode is the mode of a module.
type ModuleMode string

const (
	ModuleModeEval  ModuleMode = "eval"
	ModuleModeTrain ModuleMode = "train"
)

// ModuleType represents the type of a module.
type ModuleType int

const (
	ModuleTypeNone ModuleType = iota
	ModuleTypeFullyConnected
	ModuleTypeLayerNormalization
	ModuleTypeConv2D
	ModuleTypeConv1D
	ModuleTypePooling
	ModuleTypeRecurrent
	ModuleTypeHiddenLstm
	ModuleTypeActivation
	ModuleTypeEmbedding
)

// String returns the string representation of the ModuleType.
func (mt ModuleType) String() string {
	switch mt {
	case ModuleTypeNone:
		return "None"
	case ModuleTypeFullyConnected:
		return "FullyConnected"
	case ModuleTypeLayerNormalization:
		return "LayerNormalization"
	case ModuleTypeConv2D:
		return "Conv2D"
	case ModuleTypeConv1D:
		return "Conv1D"
	case ModuleTypePooling:
		return "Pooling"
	case ModuleTypeRecurrent:
		return "Recurrent"
	case ModuleTypeHiddenLstm:
		return "HiddenLstm"
	case ModuleTypeActivation:
		return "Activation"
	case ModuleTypeEmbedding:
		return "Embedding"
	default:
		return "Unknown"
	}
}

// Module is a collection of modules that can be used to build a model.  Modules
// persist parameters between forward passes which can be updated on backward passes
// during training.
type Module struct {
	Mode           ModuleMode
	ModuleList     []*Module
	ModuleType     ModuleType
	ActivationType functional.ActivationType
	Activation     functional.ActivationFn
	Params         functional.ActivationParams

	// mu is a mutex that protects the module's parameters during the backward pass.
	mu sync.RWMutex
}

// New creates a new module with the given activation function and parameters.
// The module is initialized in evaluation mode.
func New(activationType functional.ActivationType, activationParams functional.ActivationParams, modules ...*Module) *Module {

	activationFn, err := functional.GetActivationFunction(activationType)
	if err != nil {
		panic(err)
	}

	return &Module{
		Mode:           ModuleModeEval,
		ActivationType: activationType,
		Activation:     activationFn,
		Params:         activationParams,
		ModuleList:     modules,
	}
}

// Copy creates a copy of the module and its submodules.
func Copy(m *Module) *Module {
	m.mu.RLock()
	defer m.mu.RUnlock()

	newModule := &Module{
		Mode:           m.Mode,
		ModuleType:     m.ModuleType,
		ActivationType: m.ActivationType,
		Activation:     m.Activation,
		Params:         copyActivationParams(m.Params),
	}

	for i := range m.ModuleList {
		newModule.ModuleList = append(newModule.ModuleList, Copy(m.ModuleList[i]))
	}

	return newModule
}

func copyActivationParams(params functional.ActivationParams) functional.ActivationParams {
	out := params
	out.Weights = params.Weights.Copy()
	out.Bias = params.Bias.Copy()
	out.Kernel = params.Kernel.Copy()

	if params.Padding != nil {
		out.Padding = slices.Clone(params.Padding)
	}

	if params.Stride != nil {
		out.Stride = slices.Clone(params.Stride)
	}

	return out
}

// Forward performs a forward pass through the module and its submodules.
// It applies the activation function if one is specified.
// It returns the output tensor and any error encountered during the forward pass.
// The input is copied at the beginning of the forward pass so that the tensor
// values are not modified by the forward pass.
func (m *Module) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	output := input.Copy()

	// Use a submodule forward to apply the activation function to the output of the submodules
	// and minimize unnecessary copying of the input tensor.
	output, err := m.subModuleForward(output)
	if err != nil {
		return tensor.Tensor{}, err
	}

	return output, nil
}

// subModuleForward performs a forward pass through the module and its submodules.
// It applies the activation function if one is specified.
// It returns the output tensor and any error encountered during the forward pass.
// The input is not copied, so the tensor values are modified by the forward pass.
func (m *Module) subModuleForward(input tensor.Tensor) (tensor.Tensor, error) {
	var err error

	for i := range m.ModuleList {
		input, err = m.ModuleList[i].subModuleForward(input)
		if err != nil {
			return tensor.Tensor{}, err
		}
	}

	// Apply activation function to input if specified and activation is not first
	if m.Activation != nil {
		input, err = functional.ActivationForward(input, m.Activation, &m.Params)
		if err != nil {
			return tensor.Tensor{}, err
		}
	}

	return input, nil
}

// Eval sets the module and all its submodules to evaluation mode.
func (m *Module) Eval() {
	m.Mode = ModuleModeEval

	for i := range m.ModuleList {
		m.ModuleList[i].Eval()
	}
}

// Train sets the module and all its submodules to training mode.
func (m *Module) Train() {
	m.Mode = ModuleModeTrain

	for i := range m.ModuleList {
		m.ModuleList[i].Train()
	}
}

// PrintSummary prints a summary of the module and its submodules.
func (m *Module) PrintSummary() {
	fmt.Println("Model Summary:")
	fmt.Println("==============")
	fmt.Printf("%-20s %-20s %-15s %-15s\n", "Layer (type)", "Parameter Shape", "Param #", "Activation")
	fmt.Println("=========================================================================")

	totalParams := 0

	for i, module := range m.ModuleList {
		var moduleType string
		var paramShape string
		var paramCount int
		var activationType string

		switch module.ModuleType {
		case ModuleTypeActivation:
			paramShape = "None"
			paramCount = 0
		case ModuleTypeEmbedding:
			paramShape = fmt.Sprintf("(%d, %d)", module.Params.Weights.Shape()[0], module.Params.Weights.Shape()[1])
			paramCount = len(module.Params.Weights.Values())
		case ModuleTypeFullyConnected:
			paramShape = fmt.Sprintf("(%d, %d)", module.Params.Weights.Shape()[0], module.Params.Weights.Shape()[1])
			paramCount = len(module.Params.Weights.Values()) + len(module.Params.Bias.Values())
		case ModuleTypeConv2D:
			paramShape = fmt.Sprintf("(%d, %d, %d, %d)", module.Params.Kernel.Shape()[0], module.Params.Kernel.Shape()[1], module.Params.Kernel.Shape()[2], module.Params.Kernel.Shape()[3])
			paramCount = len(module.Params.Kernel.Values())
		case ModuleTypeConv1D:
			paramShape = fmt.Sprintf("(%d, %d, %d)", module.Params.Kernel.Shape()[0], module.Params.Kernel.Shape()[1], module.Params.Kernel.Shape()[2])
			paramCount = len(module.Params.Kernel.Values())
		case ModuleTypeHiddenLstm:
			paramShape = fmt.Sprintf("(%d, %d)", module.Params.Weights.Shape()[0], module.Params.Weights.Shape()[1])
			paramCount = len(module.Params.Weights.Values()) + len(module.Params.Bias.Values())
		default:
			paramShape = "Unknown"
			paramCount = 0
		}

		moduleType = module.ModuleType.String()
		activationType = module.ActivationType.String()

		fmt.Printf("%-20s %-20s %-15d %-15s\n", fmt.Sprintf("%d %s", i+1, moduleType), paramShape, paramCount, activationType)
		totalParams += paramCount
	}

	fmt.Println("=========================================================================")
	fmt.Printf("Total params: %d\n", totalParams)
}

// SaveToFile saves the module to a file at the given path as a gob binary file.
// It returns an error if the file cannot be created or if the module cannot be encoded.
func SaveToFile(m *Module, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(m)
	if err != nil {
		return fmt.Errorf("error encoding module: %v", err)
	}

	return nil
}

// LoadFromFile loads the module from a file at the given path as a gob binary file.
// It returns an error if the file cannot be opened or if the module cannot be decoded.
func LoadFromFile(path string) (*Module, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %v", err)
	}
	defer file.Close()

	m := &Module{}

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(m)
	if err != nil {
		return nil, fmt.Errorf("error decoding module: %v", err)
	}

	// Perform runtime setup such as setting the activation function.
	err = m.onLoad()
	if err != nil {
		return nil, fmt.Errorf("error on load: %v", err)
	}

	return m, nil
}

// onLoad is a helper function that is called after the module is loaded from a file.
// It recursively calls onLoad for all submodules and performs necessary runtime setup
// such as setting the activation function.
func (m *Module) onLoad() error {
	var err error

	for i := range m.ModuleList {
		err = m.ModuleList[i].onLoad()
		if err != nil {
			return err
		}
	}

	m.Activation, err = functional.GetActivationFunction(m.ActivationType)
	if err != nil {
		return err
	}

	return nil
}

// Initialize applies the given initializer function to the module's weights and biases,
// as well as recursively initializing all submodules. If the initializer is nil, this
// function does nothing.
func (m *Module) Initialize(initializer initializer.Initializer) {
	if initializer == nil {
		return
	}

	for i := range m.ModuleList {
		m.ModuleList[i].Initialize(initializer)
	}

	initializer(&m.Params.Weights)
	initializer(&m.Params.Bias)
	initializer(&m.Params.Kernel)
}

// Activation creates a new activation module with the specified activation function.
// It is used to apply sequential activation functions before or after other modules.
// For example, it can be used to apply a ReLU activation function to the output of
// a fully connected layer.
func Activation(activationType functional.ActivationType, activationParams functional.ActivationParams) *Module {

	output := New(activationType, activationParams)
	output.ModuleType = ModuleTypeActivation

	return output
}

// FullyConnected creates a new fully connected (linear) module with the specified
// input and output sizes. It initializes the weights and biases using the provided
// initializer function. The returned module uses a linear activation function.
func FullyConnected(inputSize, outputSize int, initializer initializer.Initializer) *Module {

	weights := tensor.Zeros([]int{outputSize, inputSize})
	bias := tensor.Zeros([]int{outputSize})

	output := New(functional.ActivationLinear, functional.ActivationParams{
		Weights: weights,
		Bias:    bias,
	})

	output.ModuleType = ModuleTypeFullyConnected

	output.Initialize(initializer)

	return output
}

// Conv2D creates a new 2D convolution module with the specified input and output sizes.
// It initializes the weights and biases using the provided initializer function.
// The returned module uses a convolution activation function.
// The expected input shape is [batch, in_channels, height, width]
func Conv2D(in_channels, out_channels int, kernelSize []int, padding, stride []int, initializer initializer.Initializer) *Module {
	kernel := tensor.Zeros([]int{out_channels, in_channels, kernelSize[0], kernelSize[1]})

	output := New(functional.ActivationConvolution, functional.ActivationParams{
		Kernel:  kernel,
		Padding: padding,
		Stride:  stride,
	})

	output.ModuleType = ModuleTypeConv2D

	output.Initialize(initializer)

	return output
}

// Conv1D creates a new 1D convolution module with the specified input and output sizes.
// It initializes the weights and biases using the provided initializer function.
// The returned module uses a linear activation function.
// The expected input shape is [batch, in_channels, length]
// The padding is applied to the left and right sides of the input with dimensions
// {0, padding}.  The stride is applied to the left and right sides of the input with
// dimensions {1, stride}.
func Conv1D(in_channels, out_channels int, kernelSize int, padding, stride int, initializer initializer.Initializer) *Module {

	kernel := tensor.Zeros([]int{out_channels, in_channels, 1, kernelSize})

	output := New(functional.ActivationConvolution, functional.ActivationParams{
		Kernel:  kernel,
		Padding: []int{0, padding},
		Stride:  []int{1, stride},
	})

	output.ModuleType = ModuleTypeConv1D

	output.Initialize(initializer)

	return output

}

// HiddenLstm creates a new LSTM module with the specified hidden layer size.
// It initializes the weights and biases using the provided initializer function.
// The input must be of shape [seq_len, batch, hidden_size].  A preceding fully
// connected layer may be required to transform the input to the correct shape.
func HiddenLstm(hiddenSize int, initializer initializer.Initializer) *Module {
	weights := tensor.Zeros([]int{4 * hiddenSize, 2 * hiddenSize})
	bias := tensor.Zeros([]int{4 * hiddenSize})

	output := New(functional.ActivationHiddenLstm, functional.ActivationParams{
		Weights: weights,
		Bias:    bias,
	})

	output.ModuleType = ModuleTypeHiddenLstm

	output.Initialize(initializer)

	return output
}

// Embedding creates a new embedding module with the specified embedding dimension.
// The weights can be supplied from a pre-trained embedding matrix or initialized randomly.
// The embedding matrix is of shape [vocab_size, embedding_dim] where vocab_size is the
// number of unique tokens in the vocabulary and embedding_dim is the number of dimensions
// in the embedding space.
func Embedding(embeddingMatrix tensor.Tensor, initializer initializer.Initializer) *Module {
	output := New(functional.ActivationEmbedding, functional.ActivationParams{
		Weights: embeddingMatrix,
	})

	output.ModuleType = ModuleTypeEmbedding

	output.Initialize(initializer)

	return output
}
