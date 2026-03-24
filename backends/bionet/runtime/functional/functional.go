package functional

import (
	"fmt"
	"math"

	"github.com/pergamon-labs/infergo/backends/bionet/runtime/tensor"
)

// ActivationType represents different activation functions.
type ActivationType int

// ActivationParams is a struct that contains the parameters for an activation function.
// It is used to control the behaviour of an activation function.  It also
// contains learnable parameters for activation functions that can be trained.
type ActivationParams struct {
	// Alpha is the slope of the negative part of the activation function.  It
	// is used to control the behaviour of activation functions that treat the
	// negative part of the input differently than the positive part, such as
	// ReLU.
	Alpha float64
	// SignalGain is the gain of the activation function.  It is used to control
	// the behaviour of activation functions that scale the input differently
	// depending on variance, such as RangeSigmoid.
	SignalGain float64
	// Weights is the tensor for the linear activation parameters for the activation
	// function.
	Weights tensor.Tensor
	// Bias is the tensor for the bias of the activation function.
	Bias tensor.Tensor
	// Kernel is the tensor for the kernel of the activation function, used for
	// convolution activation functions.  It should have shape [out_channels,
	// in_channels, kernel_height, kernel_width]
	Kernel tensor.Tensor
	// Padding is the padding for the activation function.  It is used for
	// convolution activation functions.
	Padding []int
	// Stride is the stride for the activation function.  It is used for
	// convolution activation functions.
	Stride []int
	// Gamma is the scale multiplier for activation functions that make use of
	// it, such as LayerNormalization.
	Gamma float64
	// Beta is the bias addition for activation functions that make use of
	// it, such as LayerNormalization.
	Beta float64
	// Epsilon is the epsilon for the activation function, which is applied for
	// activation functions that divide by a value with a potential min of zero.
	Epsilon float64
}

// ActivationFn is a function that applies an activation function to a slice of values.
// It takes a slice of values, and returns a new slice of values after the activation function has been applied.
type ActivationFn func(tensor tensor.Tensor, params *ActivationParams) (tensor.Tensor, error)

const (
	ActivationNone ActivationType = iota
	ActivationReLU
	ActivationPReLU
	ActivationELU
	ActivationSigmoid
	ActivationRangeSigmoid
	ActivationTanH
	ActivationSwish
	ActivationSoftmax
	ActivationLinear
	ActivationConvolution
	ActivationHiddenLstm
	ActivationEmbedding
	ActivationFlatten
	ActivationLayerNormalization
)

// String returns the string representation of the ActivationType.
func (a ActivationType) String() string {
	switch a {
	case ActivationNone:
		return "None"
	case ActivationReLU:
		return "ReLU"
	case ActivationPReLU:
		return "PReLU"
	case ActivationELU:
		return "ELU"
	case ActivationSigmoid:
		return "Sigmoid"
	case ActivationRangeSigmoid:
		return "RangeSigmoid"
	case ActivationTanH:
		return "TanH"
	case ActivationSwish:
		return "Swish"
	case ActivationSoftmax:
		return "Softmax"
	case ActivationLinear:
		return "Linear"
	case ActivationConvolution:
		return "Convolution"
	case ActivationHiddenLstm:
		return "HiddenLstm"
	case ActivationEmbedding:
		return "Embedding"
	case ActivationFlatten:
		return "Flatten"
	case ActivationLayerNormalization:
		return "LayerNormalization"
	default:
		return "Unknown"
	}
}

// GetActivationFunction returns the activation function.
// It returns an error if the activation function is not supported.
func GetActivationFunction(activationType ActivationType) (ActivationFn, error) {
	switch activationType {
	case ActivationNone:
		return Identity, nil
	case ActivationSigmoid:
		return Sigmoid, nil
	case ActivationRangeSigmoid:
		return RangeSigmoid, nil
	case ActivationReLU:
		return ReLU, nil
	case ActivationTanH:
		return Tanh, nil
	case ActivationSwish:
		return Swish, nil
	case ActivationSoftmax:
		return Softmax, nil
	case ActivationELU:
		return nil, fmt.Errorf("ELU activation fn is not implemented")
	case ActivationPReLU:
		return nil, fmt.Errorf("PReLU activation fn is not implemented")
	case ActivationLinear:
		return Linear, nil
	case ActivationConvolution:
		return Convolution, nil
	case ActivationHiddenLstm:
		return HiddenLstm, nil
	case ActivationEmbedding:
		return Embedding, nil
	case ActivationFlatten:
		return Flatten, nil
	case ActivationLayerNormalization:
		return LayerNormalization, nil
	default:
		return nil, fmt.Errorf("unknown activation function: %v", activationType)
	}
}

// GetActivationFunctionPrime returns the derivative of the activation function.
// It returns an error if the activation function is not supported.
func GetActivationFunctionPrime(activationType ActivationType) (ActivationFn, error) {
	switch activationType {
	case ActivationNone:
		return IdentityPrime, nil
	case ActivationSigmoid:
		return SigmoidPrime, nil
	case ActivationRangeSigmoid:
		return RangeSigmoidPrime, nil
	case ActivationReLU:
		return ReLUPrime, nil
	case ActivationTanH:
		return TanhPrime, nil
	case ActivationSwish:
		return SwishPrime, nil
	case ActivationSoftmax:
		return SoftmaxPrimePartial, nil
	case ActivationELU:
		return nil, fmt.Errorf("ELU gradient fn is not implemented")
	case ActivationPReLU:
		return nil, fmt.Errorf("PReLU gradient fn is not implemented")
	case ActivationLayerNormalization:
		return nil, fmt.Errorf("LayerNormalization gradient fn is not implemented")
	default:
		return nil, fmt.Errorf("unknown activation function: %v", activationType)
	}
}

// Identity applies the identity activation function to the input values.
// It returns the input values as is.
func Identity(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	return t, nil
}

// IdentityPrime computes the derivative of the identity function.
// It returns a slice of ones with the same length as the input values.
func IdentityPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	return tensor.Ones(t.Shape()), nil
}

// Sigmoid applies the sigmoid activation function to the input x.
// It uses different formulas for positive and negative inputs to improve numerical stability.
// For x < 0: f(x) = 1 / (1 + exp(-x * alpha))
// For x >= 0: f(x) = 1 / (1 + exp(-x))
// The alpha parameter allows for adjusting the slope of the negative side of the function.
func Sigmoid(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()

	for i, value := range output.Values() {
		if value < 0 {
			output.SetFlatValue(i, 1/(1+math.Exp(-value*params.Alpha)))
		} else {
			output.SetFlatValue(i, 1/(1+math.Exp(-value)))
		}
	}

	return output, nil
}

// SigmoidPrime computes the derivative of the sigmoid function.
// It uses different formulas for positive and negative inputs to match the Sigmoid function.
// For x < 0: f'(x) = (alpha * exp(-alpha * x)) / (exp(-alpha * x) + 1)^2
// For x >= 0: f'(x) = exp(-x) / (exp(-x) + 1)^2
func SigmoidPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value < 0.0 {
			en := math.Exp(-params.Alpha * value)
			output.SetFlatValue(i, (params.Alpha*en)/math.Pow(en+1, 2))
		} else {
			en := math.Exp(-value)
			output.SetFlatValue(i, en/math.Pow(en+1, 2))
		}
	}
	return output, nil
}

// RangeSigmoid applies a modified sigmoid function that outputs values in the range [-1, 1].
// It uses different formulas for positive and negative inputs to allow for asymmetry.
// For x < 0: f(x) = 2 / (1 + exp(-x * signalGain * alpha)) - 1
// For x >= 0: f(x) = 2 / (1 + exp(-x * signalGain)) - 1
// The alpha and signalGain parameters allow for adjusting the slope and range of the function.
func RangeSigmoid(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value < 0.0 {
			output.SetFlatValue(i, 2.0/(1.0+math.Exp(-value*params.SignalGain*params.Alpha))-1.0)
		} else {
			output.SetFlatValue(i, 2.0/(1.0+math.Exp(-value*params.SignalGain))-1.0)
		}
	}
	return output, nil
}

// RangeSigmoidPrime computes the derivative of the RangeSigmoid function.
// It uses different formulas for positive and negative inputs to match the RangeSigmoid function.
// For x < 0: f'(x) = 2 * alpha * signalGain * exp(-alpha * signalGain * x) / (exp(-alpha * signalGain * x) + 1)^2
// For x >= 0: f'(x) = 2 * signalGain * exp(-signalGain * x) / (exp(-signalGain * x) + 1)^2
func RangeSigmoidPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value < 0.0 {
			expTerm := math.Exp(-params.Alpha * params.SignalGain * value)
			output.SetFlatValue(i, 2.0*params.Alpha*params.SignalGain*expTerm/math.Pow(expTerm+1.0, 2))
		} else {
			expTerm := math.Exp(-params.SignalGain * value)
			output.SetFlatValue(i, 2.0*params.SignalGain*expTerm/math.Pow(expTerm+1.0, 2))
		}
	}
	return output, nil
}

// ReLU applies the Rectified Linear Unit activation function to the input x.
// For x < 0: f(x) = alpha * x
// For x >= 0: f(x) = x
// The alpha parameter allows for a non-zero slope for negative inputs (Leaky ReLU).
func ReLU(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value < 0 {
			output.SetFlatValue(i, value*params.Alpha)
		} else {
			output.SetFlatValue(i, value)
		}
	}
	return output, nil
}

// ReLUPrime computes the derivative of the ReLU function.
// For x <= 0: f'(x) = alpha
// For x > 0: f'(x) = 1
func ReLUPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value <= 0 {
			output.SetFlatValue(i, params.Alpha)
		} else {
			output.SetFlatValue(i, 1)
		}
	}
	return output, nil
}

// Tanh applies the hyperbolic tangent activation function to the input x.
// It uses different formulas for positive and negative inputs to allow for asymmetry.
// For x < 0: f(x) = tanh(x * alpha)
// For x >= 0: f(x) = tanh(x)
// The alpha parameter allows for adjusting the slope of the negative side of the function.
func Tanh(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()

	for i, value := range output.Values() {
		if value < 0 {
			output.SetFlatValue(i, math.Tanh(value*params.Alpha))
		} else {
			output.SetFlatValue(i, math.Tanh(value))
		}
	}
	return output, nil
}

// TanhPrime computes the derivative of the Tanh function.
// It uses different formulas for positive and negative inputs to match the Tanh function.
// For x < 0: f'(x) = alpha * (1 - tanh^2(x))
// For x >= 0: f'(x) = 1 - tanh^2(x)
func TanhPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		if value < 0 {
			th := math.Tanh(value * params.Alpha)
			output.SetFlatValue(i, params.Alpha*(1-th*th))
		} else {
			th := math.Tanh(value)
			output.SetFlatValue(i, 1-th*th)
		}
	}
	return output, nil
}

// Swish applies the Swish activation function to the input x.
// f(x) = x / (1 + exp(-x))
// The Swish function is a smooth, non-monotonic function that consistently matches
// or outperforms ReLU on deep networks.
func Swish(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		output.SetFlatValue(i, value/(1.0+math.Exp(-value)))
	}
	return output, nil
}

// SwishPrime computes the derivative of the Swish function.
// f'(x) = (exp(x) * (exp(x) + x + 1)) / (exp(x) + 1)^2
func SwishPrime(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		e := math.Exp(value)
		output.SetFlatValue(i, e*(e+value+1)/((e+1)*(e+1)))
	}
	return output, nil
}

// Softmax applies the softmax function to the input tensor t.
// f(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// The softmax function normalizes the inputs into a probability distribution.
// It uses the max trick for numerical stability.
// The input tensor can have any number of dimensions, with the last dimension being the feature dimension.
func Softmax(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	shape := output.Shape()
	if len(shape) < 1 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must have at least one dimension, got shape %v", shape)
	}

	featureDim := shape[len(shape)-1]
	outerDims := 1
	for i := 0; i < len(shape)-1; i++ {
		outerDims *= shape[i]
	}

	for b := 0; b < outerDims; b++ {
		pMax := math.NaN()
		pSum := 0.0

		// Find the maximum value for this slice
		for i := 0; i < featureDim; i++ {
			value := output.Values()[b*featureDim+i]
			if math.IsNaN(pMax) || value > pMax {
				pMax = value
			}
		}

		// Calculate the sum of exponentials for this slice
		for i := 0; i < featureDim; i++ {
			pSum += math.Exp(output.Values()[b*featureDim+i] - pMax)
		}

		pSum = pMax + math.Log(pSum)

		// Calculate the softmax probabilities for this slice
		for i := 0; i < featureDim; i++ {
			output.SetFlatValue(b*featureDim+i, math.Exp(output.Values()[b*featureDim+i]-pSum))
		}
	}

	return output, nil
}

// SoftmaxPrimePartial computes partial derivatives of the Softmax function.
// For the i-th element: f'(x_i) = f(x_i) * (1 - f(x_i))
// Note: This is a simplification and is only correct when considering
// the derivative with respect to a single input while holding others constant.
func SoftmaxPrimePartial(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	output := t.Copy()
	for i, value := range output.Values() {
		output.SetFlatValue(i, value*(1-value))
	}
	return output, nil
}

// Linear performs a linear transformation on the input values with matrix multiplication
// of weights and bias.
// The tensor input can be of shape [..., in_features] and the output will be [..., out_features]
// where all dimensions except the last are preserved.
func Linear(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to Linear")
	}

	if params.Weights.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("weights are empty in Linear")
	}

	if params.Bias.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("bias is empty in Linear")
	}

	inputShape := t.Shape()
	if len(inputShape) < 1 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must have at least one dimension")
	}

	inFeatures := inputShape[len(inputShape)-1]
	outFeatures := params.Weights.Shape()[0]

	if inFeatures != params.Weights.Shape()[1] {
		return tensor.Tensor{}, fmt.Errorf("input features (%d) must match weight matrix input dimension (%d)", inFeatures, params.Weights.Shape()[1])
	}

	// Reshape input to 2D: [batch_size, in_features]
	// Wrap all other dimensions into the batch dimension
	batchSize := 1
	for i := 0; i < len(inputShape)-1; i++ {
		batchSize *= inputShape[i]
	}

	reshaped := t.Copy()
	reshaped.Reshape([]int{batchSize, inFeatures})

	// Perform matrix multiplication
	out, err := tensor.Transpose(params.Weights, 0, 1)
	if err != nil {
		return tensor.Tensor{}, err
	}

	out, err = tensor.MatMul(reshaped, out)
	if err != nil {
		return tensor.Tensor{}, err
	}

	// Add bias
	out, err = tensor.ScalarDimAdd(out, params.Bias, 1)
	if err != nil {
		return tensor.Tensor{}, err
	}

	// Unwrap batch dimension into original dimensions
	outputShape := append(inputShape[:len(inputShape)-1], outFeatures)
	out.Reshape(outputShape)

	return out, nil
}

// LinearGrad computes the gradient of the linear function with respect to the input.
// It returns the gradient of the input with respect to the output.
// The tensor input should be of shape [batch, inner_dim]
func LinearGrad(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to LinearGrad")
	}

	if params.Weights.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("weights are empty in LinearGrad")
	}

	out, err := tensor.MatMul(t, params.Weights)
	if err != nil {
		return tensor.Tensor{}, err
	}

	return out, nil
}

// Convolution applies a convolution operation to the input tensor.
// The tensor input should be of shape [batch, in_channels, height, width]
func Convolution(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to Convolution")
	}

	if params.Kernel.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("kernel is empty in Convolution")
	}

	out, err := tensor.Convolve(t, params.Kernel, params.Padding, params.Stride)
	if err != nil {
		return tensor.Tensor{}, err
	}

	return out, nil
}

// ConvolutionGrad computes the gradient of the convolution function with respect to the input.
// It returns the gradient of the input with respect to the output.
// The tensor input should be of shape [batch, in_channels, height, width]
func ConvolutionGrad(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to ConvolutionGrad")
	}

	if params.Kernel.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("kernel is empty in ConvolutionGrad")
	}

	// The gradient is the convolution of the input with the flipped kernel
	// along the height and width dimensions
	rotatedKernel, err := tensor.Rotate180(params.Kernel, 2, 3)
	if err != nil {
		return tensor.Tensor{}, err
	}

	out, err := tensor.Convolve(t, rotatedKernel, params.Padding, params.Stride)
	if err != nil {
		return tensor.Tensor{}, err
	}

	return out, nil
}

// Flatten flattens the inner dimensions of the input tensor to a single dimension.
// The tensor input should be of shape [batch, inner_dims...].
// The output tensor will have shape [batch, inner_dims[0]*inner_dims[1]*...*inner_dims[n]].
func Flatten(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	shape := t.Shape()
	if len(shape) < 2 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must be at least 2D, got shape %v", shape)
	}

	innerDims := 1
	for _, dim := range shape[1:] {
		innerDims *= dim
	}

	t.Reshape([]int{shape[0], innerDims})

	return t, nil
}

// MaskedAveragePool averages a [seq_len, batch_size, feature_dim] tensor across
// the sequence dimension using a binary mask. The mask may be shaped
// [seq_len, batch_size] or [seq_len, batch_size, 1]. The result has shape
// [batch_size, feature_dim].
func MaskedAveragePool(t tensor.Tensor, mask tensor.Tensor) (tensor.Tensor, error) {
	inputShape := t.Shape()
	if len(inputShape) != 3 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must be 3D, got shape %v", inputShape)
	}

	maskShape := mask.Shape()
	switch len(maskShape) {
	case 2:
		if maskShape[0] != inputShape[0] || maskShape[1] != inputShape[1] {
			return tensor.Tensor{}, fmt.Errorf("mask shape %v must match [seq_len, batch_size]", maskShape)
		}
	case 3:
		if maskShape[0] != inputShape[0] || maskShape[1] != inputShape[1] || maskShape[2] != 1 {
			return tensor.Tensor{}, fmt.Errorf("mask shape %v must match [seq_len, batch_size, 1]", maskShape)
		}
	default:
		return tensor.Tensor{}, fmt.Errorf("mask must be 2D or 3D, got shape %v", maskShape)
	}

	seqLen := inputShape[0]
	batchSize := inputShape[1]
	featureDim := inputShape[2]
	output := tensor.Zeros([]int{batchSize, featureDim})
	counts := make([]float64, batchSize)

	values := t.Values()
	maskValues := mask.Values()
	for seqIdx := 0; seqIdx < seqLen; seqIdx++ {
		for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
			maskIndex := seqIdx*batchSize + batchIdx
			if len(maskShape) == 3 {
				maskIndex = (seqIdx*batchSize + batchIdx) * maskShape[2]
			}

			if maskValues[maskIndex] <= 0 {
				continue
			}

			counts[batchIdx]++
			for featureIdx := 0; featureIdx < featureDim; featureIdx++ {
				inputIndex := (seqIdx*batchSize+batchIdx)*featureDim + featureIdx
				outputIndex := batchIdx*featureDim + featureIdx
				output.Values()[outputIndex] += values[inputIndex]
			}
		}
	}

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		if counts[batchIdx] == 0 {
			continue
		}

		for featureIdx := 0; featureIdx < featureDim; featureIdx++ {
			outputIndex := batchIdx*featureDim + featureIdx
			output.Values()[outputIndex] /= counts[batchIdx]
		}
	}

	return output, nil
}

// LayerNormalization normalizes the input tensor along the batch dimension.
// It applies the batch normalization formula: y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
// where mean and variance are computed per batch, and gamma and beta are learnable parameters.
// The tensor input should be at least 1D. If the input tensor has more than 1 dimension, the first dimension is considered the batch dimension.
// The remaining dimensions are considered the inner dimensions.
func LayerNormalization(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to LayerNormalization")
	}

	shape := t.Shape()
	if len(shape) < 1 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must be at least 1D, got shape %v", shape)
	}

	batchDim := 1
	otherDims := shape
	if len(shape) > 1 {
		batchDim = shape[0]
		otherDims = shape[1:]
	}
	innerDim := 1
	for _, dim := range otherDims {
		innerDim *= dim
	}

	newValues := make([]float64, len(t.Values()))

	for b := 0; b < batchDim; b++ {
		mean := 0.0
		for i := 0; i < innerDim; i++ {
			mean += t.Values()[b*innerDim+i]
		}
		mean /= float64(innerDim)

		variance := 0.0
		for i := 0; i < innerDim; i++ {
			variance += math.Pow(t.Values()[b*innerDim+i]-mean, 2)
		}
		variance /= float64(innerDim)

		for i := 0; i < innerDim; i++ {
			newValues[b*innerDim+i] = (t.Values()[b*innerDim+i] - mean) / math.Sqrt(variance+params.Epsilon)
			newValues[b*innerDim+i] = newValues[b*innerDim+i]*params.Gamma + params.Beta
		}
	}

	t.SetValues(newValues)

	return t, nil
}

// HiddenLstm applies the Long Short-Term Memory (LSTM) activation function to the input tensor.
// It takes an input tensor of shape [input_len, batch_dim, in_features] and activation parameters,
// and returns the output tensor after the LSTM activation has been applied.
// The function initializes the hidden state and cell state, performs LSTM operations,
// and updates the cell and hidden states at each time step.
// It returns an error if the activation parameters are nil or if the weights or bias are empty.
// It assumes that the input tensor has already been transformed to the correct shape with an input
// fully connected layer.
func HiddenLstm(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to Lstm")
	}

	if params.Weights.IsEmpty() || params.Bias.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("weights or bias are empty in Lstm")
	}

	if len(t.Shape()) != 3 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must be 3D, got shape %v", t.Shape())
	}

	inputLen, batchDim, inFeatures := t.Shape()[0], t.Shape()[1], t.Shape()[2]
	hiddenSize := params.Weights.Shape()[0] / 4

	if inFeatures != hiddenSize {
		return tensor.Tensor{}, fmt.Errorf("input features must match hidden size, got %d and %d", inFeatures, hiddenSize)
	}

	// Initialize hidden state and cell state
	hiddenState := tensor.Zeros([]int{batchDim, hiddenSize})
	cellState := tensor.Zeros([]int{batchDim, hiddenSize})

	// Prepare output tensor
	outputs := tensor.Zeros([]int{inputLen, batchDim, hiddenSize})

	// Perform LSTM operations
	for tIndex := 0; tIndex < inputLen; tIndex++ {
		// Extract the current input slice
		inputSlice, err := tensor.Slice(t, []int{tIndex, 0, 0}, []int{tIndex + 1, batchDim, inFeatures})
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Reshape the input slice to [batch_dim, in_features]
		inputSlice.Reshape([]int{batchDim, inFeatures})

		// Concatenate input and hidden state
		concatInputHidden, err := tensor.ConcatTensors(1, inputSlice, hiddenState)
		if err != nil {
			return tensor.Tensor{}, err
		}

		gates, err := Linear(concatInputHidden, &ActivationParams{Weights: params.Weights, Bias: params.Bias})
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Split gates into input, forget, cell, and output gates
		gateSize := gates.Shape()[1] / 4
		inputGate, err := tensor.Slice(gates, []int{0, 0}, []int{batchDim, gateSize})
		if err != nil {
			return tensor.Tensor{}, err
		}
		forgetGate, err := tensor.Slice(gates, []int{0, gateSize}, []int{batchDim, 2 * gateSize})
		if err != nil {
			return tensor.Tensor{}, err
		}
		cellGate, err := tensor.Slice(gates, []int{0, 2 * gateSize}, []int{batchDim, 3 * gateSize})
		if err != nil {
			return tensor.Tensor{}, err
		}
		outputGate, err := tensor.Slice(gates, []int{0, 3 * gateSize}, []int{batchDim, 4 * gateSize})
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Apply activation functions
		inputGate, err = Sigmoid(inputGate, &ActivationParams{Alpha: 1.0})
		if err != nil {
			return tensor.Tensor{}, err
		}
		forgetGate, err = Sigmoid(forgetGate, &ActivationParams{Alpha: 1.0})
		if err != nil {
			return tensor.Tensor{}, err
		}
		cellGate, err = Tanh(cellGate, &ActivationParams{Alpha: 1.0})
		if err != nil {
			return tensor.Tensor{}, err
		}
		outputGate, err = Sigmoid(outputGate, &ActivationParams{Alpha: 1.0})
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Update cell state
		cellState, err = tensor.ElementMultiply(forgetGate, cellState)
		if err != nil {
			return tensor.Tensor{}, err
		}
		gateMult, err := tensor.ElementMultiply(inputGate, cellGate)
		if err != nil {
			return tensor.Tensor{}, err
		}
		cellState, err = tensor.ElementAdd(cellState, gateMult)
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Update hidden state
		tanhCellState, err := Tanh(cellState, params)
		if err != nil {
			return tensor.Tensor{}, err
		}
		hiddenState, err = tensor.ElementMultiply(outputGate, tanhCellState)
		if err != nil {
			return tensor.Tensor{}, err
		}

		// Store the output for this time step
		err = outputs.SetSlice([]int{tIndex, 0, 0}, []int{tIndex + 1, batchDim, hiddenSize}, hiddenState.Values())
		if err != nil {
			return tensor.Tensor{}, err
		}
	}

	return outputs, nil
}

// Embedding maps the input tensor indices to an embedding map.  The input tensor
// should be of shape [seq_len, batch_size, 1] with whole number values in the range [0, vocab_size-1].
// The embedding map should be of shape [vocab_size, embedding_dim].  The output tensor
// will be of shape [seq_len, batch_size, embedding_dim].
func Embedding(t tensor.Tensor, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to Embedding")
	}

	if params.Weights.IsEmpty() {
		return tensor.Tensor{}, fmt.Errorf("weights are empty in Embedding")
	}

	if len(t.Shape()) != 3 || t.Shape()[2] != 1 {
		return tensor.Tensor{}, fmt.Errorf("input tensor must be 3D with shape [seq_len, batch_size, 1], got shape %v", t.Shape())
	}

	seqLen, batchSize, _ := t.Shape()[0], t.Shape()[1], t.Shape()[2]
	vocabSize, embeddingDim := params.Weights.Shape()[0], params.Weights.Shape()[1]

	// Create output tensor with shape [seq_len, batch_size, embedding_dim]
	output := tensor.Zeros([]int{seqLen, batchSize, embeddingDim})

	// Perform embedding lookup
	for i := 0; i < seqLen; i++ {
		for j := 0; j < batchSize; j++ {
			index, err := t.GetValue([]int{i, j, 0})
			indexInt := int(index)

			if err != nil {
				return tensor.Tensor{}, fmt.Errorf("failed to get index from input tensor: %v", err)
			}

			if indexInt >= vocabSize {
				return tensor.Tensor{}, fmt.Errorf("index %d out of range [0, %d]", indexInt, vocabSize)
			}

			if indexInt == -1 {
				continue
			}

			embedding, err := tensor.Slice(params.Weights, []int{indexInt, 0}, []int{indexInt + 1, embeddingDim})
			if err != nil {
				return tensor.Tensor{}, fmt.Errorf("failed to get embedding from weights: %v", err)
			}

			err = output.SetSlice([]int{i, j, 0}, []int{i + 1, j + 1, embeddingDim}, embedding.Values())
			if err != nil {
				return tensor.Tensor{}, fmt.Errorf("failed to set embedding in output tensor: %v", err)
			}
		}
	}

	return output, nil
}

// ApplyActivationFunction applies an activation function to a tensor.
// It returns an error if the activation function is not supported or if the length
// of the values does not match the length of the tensor values.
func ActivationForward(t tensor.Tensor, activationFn ActivationFn, params *ActivationParams) (tensor.Tensor, error) {
	if params == nil {
		return tensor.Tensor{}, fmt.Errorf("nil params passed to activation")
	}

	t, err := activationFn(t, params)
	if err != nil {
		return tensor.Tensor{}, err
	}

	return t, nil
}
