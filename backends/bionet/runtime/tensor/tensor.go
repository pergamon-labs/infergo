package tensor

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"

	bionet "github.com/pergamon-labs/infergo/backends/bionet/runtime"
)

// Tensor represents a multi-dimensional array of float64 values.
type Tensor struct {
	values     []float64 // Flattened array of tensor values
	dimensions []int     // Dimensions of the tensor
}

// New creates a new tensor with the specified values and dimensions.
func New(values []float64, dimensions []int) Tensor {
	return Tensor{values: values, dimensions: dimensions}
}

// Values returns the flattened array of tensor values.
func (t *Tensor) Values() []float64 {
	return t.values
}

// SetValues sets the flattened array of tensor values.
// It returns an error if the length of the values does not match the length of the tensor values.
func (t *Tensor) SetValues(values []float64) error {
	if len(values) != len(t.values) {
		return fmt.Errorf("tensor: cannot set values with incorrect length")
	}
	copy(t.values, values)
	return nil
}

// Copy creates a copy of the tensor.  It is a deep copy, so the values for the tensor are copied
// to the new tensor in addition to the struct headers.
func (t *Tensor) Copy() Tensor {
	return Tensor{
		values:     slices.Clone(t.values),
		dimensions: slices.Clone(t.dimensions),
	}
}

// Shape returns the dimensions of the tensor.
func (t *Tensor) Shape() []int {
	return t.dimensions
}

// String returns a string representation of the tensor.
func (t *Tensor) String() string {
	if len(t.dimensions) == 0 {
		return "[]"
	}

	var result string
	var printTensor func([]float64, []int, int)

	printTensor = func(values []float64, dims []int, depth int) {
		if len(dims) == 1 {
			result += "["
			for i, v := range values {
				if i > 0 {
					result += ", "
				}
				result += fmt.Sprintf("%.4f", v)
			}
			result += "]"
			return
		}

		result += "[\n"
		subSize := bionet.CumProd(dims[1:])
		for i := 0; i < dims[0]; i++ {
			result += strings.Repeat("  ", depth+1)
			start := i * subSize
			end := start + subSize
			printTensor(values[start:end], dims[1:], depth+1)
			if i < dims[0]-1 {
				result += ",\n"
			}
		}
		result += "\n" + strings.Repeat("  ", depth) + "]"
	}

	printTensor(t.values, t.dimensions, 0)
	return result
}

// GetValue retrieves the value at the specified indices in the tensor.
// It returns the value and an error if the indices are invalid.  It works
// for an arbitrary number of dimensions, however the number of dimensions
// must match the number of indices or an error is returned.
func (t *Tensor) GetValue(indices []int) (float64, error) {
	if len(indices) != len(t.dimensions) {
		return 0, fmt.Errorf("number of indices must match number of dimensions (%d != %d)", len(indices), len(t.dimensions))
	}

	index := 0
	stride := 1

	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.dimensions[i] {
			return 0, fmt.Errorf("index out of bounds (%d >= %d)", indices[i], t.dimensions[i])
		}
		index += indices[i] * stride
		stride *= t.dimensions[i]
	}

	return t.values[index], nil
}

// SetValue sets the value at the specified indices in the tensor.
// It returns an error if the indices are invalid.  It works for an
// arbitrary number of dimensions, however the number of dimensions
// must match the number of indices or an error is returned.
func (t *Tensor) SetValue(indices []int, value float64) error {
	if len(indices) != len(t.dimensions) {
		return fmt.Errorf("number of indices must match number of dimensions (%d != %d)", len(indices), len(t.dimensions))
	}

	index := 0
	stride := 1

	for i := len(indices) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.dimensions[i] {
			return fmt.Errorf("index out of bounds (%d >= %d)", indices[i], t.dimensions[i])
		}
		index += indices[i] * stride
		stride *= t.dimensions[i]
	}

	t.values[index] = value
	return nil
}

// SetFlatValue sets the value at the specified index in the flattened tensor.
func (t *Tensor) SetFlatValue(index int, value float64) {
	t.values[index] = value
}

// GetFlatValue retrieves the value at the specified index in the flattened tensor.
func (t *Tensor) GetFlatValue(index int) float64 {
	return t.values[index]
}

// IsEmpty returns true if the tensor has no values, false otherwise.
func (t *Tensor) IsEmpty() bool {
	return len(t.values) == 0
}

// IsScalar returns true if the tensor is a scalar (has one value and no dimensions), false otherwise.
func (t *Tensor) IsScalar() bool {
	return len(t.values) == 1 && len(t.dimensions) == 0
}

// Scalar returns the scalar value of the tensor if it is a scalar, and an error otherwise.
func (t *Tensor) Scalar() (float64, error) {
	if !t.IsScalar() {
		return 0, fmt.Errorf("tensor is not a scalar")
	}
	return t.values[0], nil
}

// Fill sets all values in the tensor to the specified value.
func (t *Tensor) Fill(value float64) *Tensor {
	for i := range t.values {
		t.values[i] = value
	}

	return t
}

// Reshape changes the dimensions of the tensor.  If the new values are longer than the old
// values, the extra values are padded with 0s.  If the new values are shorter
// than the old values, the overflowing values are discarded.
func (t *Tensor) Reshape(dimensions []int) *Tensor {
	t.dimensions = dimensions

	newValueLength := bionet.CumProd(dimensions)

	oldValues := t.values
	t.values = make([]float64, newValueLength)

	// Copy the old values to the new values, leaving blanks if the new
	// values are longer than the old values.
	minLength := min(len(oldValues), newValueLength)
	for i := 0; i < minLength; i++ {
		t.values[i] = oldValues[i]
	}

	return t
}

// Squeeze removes a dimension of size 1 from the tensor.  The dimension to remove is specified by the
// index of the dimension in the dimensions array.  If a dimension of -1 is specified, all dimensions of
// size 1 are removed.
func (t *Tensor) Squeeze(dim int) *Tensor {
	if dim == -1 {
		// Remove all dimensions of size 1
		newDims := make([]int, 0)
		for _, d := range t.dimensions {
			if d != 1 {
				newDims = append(newDims, d)
			}
		}
		t.Reshape(newDims)
	} else if dim >= 0 && dim < len(t.dimensions) && t.dimensions[dim] == 1 {
		// Remove the specified dimension if it's of size 1
		newDims := make([]int, 0, len(t.dimensions)-1)
		for i, d := range t.dimensions {
			if i != dim {
				newDims = append(newDims, d)
			}
		}
		t.Reshape(newDims)
	}

	return t
}

// Unsqueeze adds a dimension of size 1 to the tensor at the specified index.
// If the dimension is invalid, the tensor is returned unchanged.
func (t *Tensor) Unsqueeze(dim int) *Tensor {
	if dim < 0 || dim > len(t.dimensions) {
		return t
	}

	newDims := make([]int, len(t.dimensions)+1)
	copy(newDims, t.dimensions[:dim])
	newDims[dim] = 1
	copy(newDims[dim+1:], t.dimensions[dim:])
	t.Reshape(newDims)
	return t
}

// ElementMultiply multiplies the values of two tensors together.  The tensors must have the same
// number of dimensions and the same number of values.
func ElementMultiply(t1, t2 Tensor) (Tensor, error) {
	err := validateDimensionMatch(&t1, &t2)
	if err != nil {
		return Tensor{}, err
	}

	output := t1.Copy()

	for i := range output.values {
		output.values[i] *= t2.values[i]
	}

	return output, nil
}

// ElementAdd performs element-wise addition of two tensors.
// The tensors must have the same dimensions and number of values.
// It returns the result tensor (which is t1 modified) and an error if the dimensions don't match.
func ElementAdd(t1, t2 Tensor) (Tensor, error) {
	err := validateDimensionMatch(&t1, &t2)
	if err != nil {
		return Tensor{}, err
	}

	output := t1.Copy()

	for i := range output.values {
		output.values[i] += t2.values[i]
	}

	return output, nil
}

// ElementSubtract performs element-wise subtraction of two tensors.
// The tensors must have the same dimensions and number of values.
// It returns the result tensor (which is t1 modified) and an error if the dimensions don't match.
func ElementSubtract(t1, t2 Tensor) (Tensor, error) {
	err := validateDimensionMatch(&t1, &t2)
	if err != nil {
		return Tensor{}, err
	}

	output := t1.Copy()

	for i := range output.values {
		output.values[i] -= t2.values[i]
	}

	return output, nil
}

// ElementDivide performs element-wise division of two tensors.
// The tensors must have the same dimensions and number of values.
// It returns the result tensor (which is t1 modified) and an error if the dimensions don't match.
// Note: This function does not check for division by zero.
func ElementDivide(t1, t2 Tensor) (Tensor, error) {
	err := validateDimensionMatch(&t1, &t2)

	if err != nil {
		return Tensor{}, err
	}

	output := t1.Copy()

	for i := range output.values {
		output.values[i] /= t2.values[i]
	}

	return output, nil
}

// ScalarDimAdd adds a scalar value to all elements of a tensor along a given dimension.
// It returns the modified tensor and an error if the dimensions don't match.
// It only requires the right hand tensor to match along the requested dimension.
func ScalarDimAdd(t1, t2 Tensor, dim int) (Tensor, error) {
	if dim < 0 || dim >= len(t1.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimension: %d", dim)
	}

	if len(t2.dimensions) != 1 || t2.dimensions[0] != t1.dimensions[dim] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for scalar addition along dimension %d (%v + %v)", dim, t2.dimensions, t1.dimensions)
	}

	// Calculate the stride for the given dimension
	stride := 1
	for i := dim + 1; i < len(t1.dimensions); i++ {
		stride *= t1.dimensions[i]
	}

	output := t1.Copy()

	// Perform the addition
	for i := 0; i < len(t1.values); i++ {
		dimIndex := (i / stride) % t1.dimensions[dim]
		output.values[i] += t2.values[dimIndex]
	}

	return output, nil
}

// ScalarDimSubtract subtracts a scalar value from all elements of a tensor along a given dimension.
// It returns the modified tensor and an error if the dimensions don't match.
// It only requires the right hand tensor to match along the requested dimension.
func ScalarDimSubtract(t1, t2 Tensor, dim int) (Tensor, error) {
	if dim < 0 || dim >= len(t1.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimension: %d", dim)
	}

	if len(t2.dimensions) != 1 || t2.dimensions[0] != t1.dimensions[dim] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for scalar subtraction along dimension %d (%v - %v)", dim, t2.dimensions, t1.dimensions)
	}

	// Calculate the stride for the given dimension
	stride := 1
	for i := dim + 1; i < len(t1.dimensions); i++ {
		stride *= t1.dimensions[i]
	}

	output := t1.Copy()

	// Perform the subtraction
	for i := 0; i < len(output.values); i++ {
		dimIndex := (i / stride) % t1.dimensions[dim]
		output.values[i] -= t2.values[dimIndex]
	}

	return output, nil
}

// ScalarDimMultiply multiplies a scalar value with all elements of a tensor along a given dimension.
// It returns the modified tensor and an error if the dimensions don't match.
// It only requires the right hand tensor to match along the requested dimension.
func ScalarDimMultiply(t1, t2 Tensor, dim int) (Tensor, error) {
	if dim < 0 || dim >= len(t1.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimension: %d", dim)
	}

	if len(t2.dimensions) != 1 || t2.dimensions[0] != t1.dimensions[dim] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for scalar multiplication along dimension %d (%v * %v)", dim, t2.dimensions, t1.dimensions)
	}

	// Calculate the stride for the given dimension
	stride := 1
	for i := dim + 1; i < len(t1.dimensions); i++ {
		stride *= t1.dimensions[i]
	}

	output := t1.Copy()

	// Perform the multiplication
	for i := 0; i < len(t1.values); i++ {
		dimIndex := (i / stride) % t1.dimensions[dim]
		output.values[i] *= t2.values[dimIndex]
	}

	return output, nil
}

// ScalarDimDivide divides all elements of a tensor by a scalar value along a given dimension.
// It returns the modified tensor and an error if the dimensions don't match.
// It only requires the right hand tensor to match along the requested dimension.
// Note: This function does not check for division by zero.
func ScalarDimDivide(t1, t2 Tensor, dim int) (Tensor, error) {
	if dim < 0 || dim >= len(t1.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimension: %d", dim)
	}

	if len(t2.dimensions) != 1 || t2.dimensions[0] != t1.dimensions[dim] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for scalar division along dimension %d (%v / %v)", dim, t2.dimensions, t1.dimensions)
	}

	// Calculate the stride for the given dimension
	stride := 1
	for i := dim + 1; i < len(t1.dimensions); i++ {
		stride *= t1.dimensions[i]
	}

	output := t1.Copy()

	// Perform the division
	for i := 0; i < len(output.values); i++ {
		dimIndex := (i / stride) % t1.dimensions[dim]
		output.values[i] /= t2.values[dimIndex]
	}

	return output, nil
}

// validateDimensionMatch checks if two tensors have the same dimensions and number of values.
// It returns an error if the dimensions or number of values don't match.
func validateDimensionMatch(t1, t2 *Tensor) error {
	if len(t1.dimensions) != len(t2.dimensions) {
		return fmt.Errorf("tensors must have the same number of dimensions.  t1 has %d dimensions and t2 has %d dimensions", len(t1.dimensions), len(t2.dimensions))
	}

	if !slices.Equal(t1.dimensions, t2.dimensions) {
		return fmt.Errorf("tensor shapes must match exactly. t1 has shape %v and t2 has shape %v", t1.dimensions, t2.dimensions)
	}

	if len(t1.values) != len(t2.values) {
		return fmt.Errorf("tensors dimensions must be the same.  t1 has %d values and t2 has %d values", len(t1.values), len(t2.values))
	}

	return nil
}

// ScalarAdd adds a scalar value to all elements of a tensor.
// It returns the modified tensor.
func ScalarAdd(t Tensor, scalar float64) Tensor {
	for i := range t.values {
		t.values[i] += scalar
	}
	return t
}

// ScalarSubtract subtracts a scalar value from all elements of a tensor.
// It returns the modified tensor.
func ScalarSubtract(t Tensor, scalar float64) Tensor {
	for i := range t.values {
		t.values[i] -= scalar
	}
	return t
}

// ScalarMultiply multiplies all elements of a tensor by a scalar value.
// It returns the modified tensor.
func ScalarMultiply(t Tensor, scalar float64) Tensor {
	for i := range t.values {
		t.values[i] *= scalar
	}
	return t
}

// ScalarDivide divides all elements of a tensor by a scalar value.
// It returns the modified tensor.
// Note: This function does not check for division by zero.
func ScalarDivide(t Tensor, scalar float64) Tensor {
	for i := range t.values {
		t.values[i] /= scalar
	}
	return t
}

// Abs applies the absolute value function to all elements of the tensor.
// It modifies the tensor in place.
func (t *Tensor) Abs() {
	for i := range t.values {
		t.values[i] = math.Abs(t.values[i])
	}
}

// Sqrt applies the square root function to all elements of the tensor.
// It modifies the tensor in place.
// Note: This function does not check for negative values.
func (t *Tensor) Sqrt() {
	for i := range t.values {
		t.values[i] = math.Sqrt(t.values[i])
	}
}

// Clip constrains all elements of the tensor to be between min and max values.
// It modifies the tensor in place.
func (t *Tensor) Clip(min, max float64) {
	for i := range t.values {
		t.values[i] = math.Max(min, math.Min(max, t.values[i]))
	}
}

// Zero sets all elements of the tensor to 0.
func (t *Tensor) Zero() {
	for i := range t.values {
		t.values[i] = 0
	}
}

// ConcatTensors concatenates tensors along a given dimension.
// It returns the concatenated tensor and an error if the tensors have incompatible dimensions.
func ConcatTensors(dim int, tensors ...Tensor) (Tensor, error) {
	if len(tensors) == 0 {
		return Tensor{}, fmt.Errorf("no tensors provided")
	}

	// Validate dimensions
	baseDims := tensors[0].dimensions
	for i, t := range tensors {
		if i > 0 && !dimensionsMatch(baseDims, t.dimensions, dim) {
			return Tensor{}, fmt.Errorf("tensor %d has incompatible dimensions", i)
		}
	}

	// Calculate new dimensions
	newDims := make([]int, len(baseDims))
	copy(newDims, baseDims)
	for _, t := range tensors[1:] {
		newDims[dim] += t.dimensions[dim]
	}

	// Create new tensor
	newTensor := Tensor{
		dimensions: newDims,
		values:     make([]float64, 0, bionet.CumProd(newDims)),
	}

	// Calculate strides
	stride := 1
	for i := len(baseDims) - 1; i > dim; i-- {
		stride *= baseDims[i]
	}

	// Concatenate tensors
	for i := 0; i < bionet.CumProd(baseDims[:dim]); i++ {
		for _, t := range tensors {
			start := i * stride * t.dimensions[dim]
			end := start + stride*t.dimensions[dim]
			newTensor.values = append(newTensor.values, t.values[start:end]...)
		}
	}

	return newTensor, nil
}

// dimensionsMatch checks if two sets of dimensions match along a given dimension.
// It returns true if the dimensions match for a concat operation along the specified dimension,
// false otherwise.
func dimensionsMatch(dims1, dims2 []int, dim int) bool {
	for i := 0; i < len(dims1); i++ {
		if i != dim && dims1[i] != dims2[i] {
			return false
		}
	}
	return true
}

// Dot performs a dot product between two tensors.
// It returns the resulting tensor and an error if the tensors have incompatible dimensions.
func Dot(t1, t2 Tensor) (Tensor, error) {
	// Check if the tensors can be multiplied
	if len(t1.dimensions) == 0 || len(t2.dimensions) == 0 {
		return Tensor{}, fmt.Errorf("cannot perform dot product on empty tensors")
	}
	if t1.dimensions[len(t1.dimensions)-1] != t2.dimensions[0] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for dot product: %v and %v", t1.dimensions, t2.dimensions)
	}

	// Calculate the dimensions of the resulting tensor
	newDims := append(t1.dimensions[:len(t1.dimensions)-1], t2.dimensions[1:]...)

	// Create the result tensor
	result := Tensor{
		dimensions: newDims,
		values:     make([]float64, bionet.CumProd(newDims)),
	}

	// Perform the dot product
	t1Stride := bionet.CumProd(t1.dimensions[len(t1.dimensions)-1:])
	t2Stride := bionet.CumProd(t2.dimensions[1:])
	resultStride := bionet.CumProd(newDims[len(newDims)-len(t2.dimensions)+1:])

	for i := 0; i < len(result.values); i++ {
		t1Start := (i / resultStride) * t1Stride
		t2Start := (i % resultStride)
		sum := 0.0
		for j := 0; j < t1.dimensions[len(t1.dimensions)-1]; j++ {
			sum += t1.values[t1Start+j] * t2.values[t2Start+j*t2Stride]
		}
		result.values[i] = sum
	}

	return result, nil
}

// Transpose performs a transpose operation on a tensor along two given dimensions.
// It returns the resulting tensor and an error if the dimensions are invalid.
func Transpose(t Tensor, dim1, dim2 int) (Tensor, error) {
	if dim1 < 0 || dim2 < 0 || dim1 >= len(t.dimensions) || dim2 >= len(t.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimensions: dim1=%d, dim2=%d", dim1, dim2)
	}

	if dim1 == dim2 {
		return t, nil // No change needed if dimensions are the same
	}

	newDims := make([]int, len(t.dimensions))
	copy(newDims, t.dimensions)
	newDims[dim1], newDims[dim2] = newDims[dim2], newDims[dim1]

	result := Tensor{
		dimensions: newDims,
		values:     make([]float64, len(t.values)),
	}

	// Calculate strides for the original and new tensor
	oldStrides := make([]int, len(t.dimensions))
	newStrides := make([]int, len(t.dimensions))
	oldStride, newStride := 1, 1

	for i := len(t.dimensions) - 1; i >= 0; i-- {
		oldStrides[i] = oldStride
		newStrides[i] = newStride
		oldStride *= t.dimensions[i]
		newStride *= newDims[i]
	}

	// Perform the transpose
	for i := range t.values {
		oldIndex := 0
		newIndex := 0
		remaining := i

		for j := 0; j < len(t.dimensions); j++ {
			dim := remaining / oldStrides[j]
			remaining %= oldStrides[j]

			if j == dim1 {
				newIndex += dim * newStrides[dim2]
			} else if j == dim2 {
				newIndex += dim * newStrides[dim1]
			} else {
				newIndex += dim * newStrides[j]
			}

			oldIndex += dim * oldStrides[j]
		}

		result.values[newIndex] = t.values[oldIndex]
	}

	return result, nil
}

// MatMul performs a matrix multiplication between two tensors.
// It returns the resulting tensor and an error if the tensors have incompatible dimensions.
// The tensors can be 1D, 2D, or 3D. For 1D tensors, it performs vector-matrix multiplication.
// leftTensor: [batch_size, rows, inner_dim] or [rows, inner_dim] or [inner_dim]
// rightTensor: [batch_size, inner_dim, cols] or [inner_dim, cols]
// result: [batch_size, rows, cols] or [rows, cols] or [cols]
func MatMul(leftTensor, rightTensor Tensor) (Tensor, error) {
	// Check if dimensions are compatible for matrix multiplication
	if len(leftTensor.dimensions) < 1 || len(leftTensor.dimensions) > 3 ||
		len(rightTensor.dimensions) < 2 || len(rightTensor.dimensions) > 3 ||
		leftTensor.dimensions[len(leftTensor.dimensions)-1] != rightTensor.dimensions[len(rightTensor.dimensions)-2] {
		return Tensor{}, fmt.Errorf("incompatible dimensions for matrix multiplication (%v, %v)", leftTensor.dimensions, rightTensor.dimensions)
	}

	// Calculate output dimensions
	var outputDimensions []int
	if len(leftTensor.dimensions) == 1 {
		outputDimensions = []int{rightTensor.dimensions[len(rightTensor.dimensions)-1]}
	} else {
		outputDimensions = make([]int, len(leftTensor.dimensions))
		copy(outputDimensions, leftTensor.dimensions)
		outputDimensions[len(outputDimensions)-1] = rightTensor.dimensions[len(rightTensor.dimensions)-1]
	}

	// Initialize output tensor
	resultTensor := Tensor{
		values:     make([]float64, bionet.CumProd(outputDimensions)),
		dimensions: outputDimensions,
	}

	// Perform matrix multiplication
	batchSize := 1
	leftRows := 1
	if len(leftTensor.dimensions) == 3 {
		batchSize = leftTensor.dimensions[0]
		leftRows = leftTensor.dimensions[1]
	} else if len(leftTensor.dimensions) == 2 {
		leftRows = leftTensor.dimensions[0]
	}
	innerDim := leftTensor.dimensions[len(leftTensor.dimensions)-1]
	rightCols := rightTensor.dimensions[len(rightTensor.dimensions)-1]

	for batch := 0; batch < batchSize; batch++ {
		for row := 0; row < leftRows; row++ {
			for col := 0; col < rightCols; col++ {
				dotProduct := 0.0
				for k := 0; k < innerDim; k++ {
					leftIndex := batch*leftRows*innerDim + row*innerDim + k
					if len(leftTensor.dimensions) == 1 {
						leftIndex = k
					}
					rightIndex := batch*innerDim*rightCols + k*rightCols + col
					if len(rightTensor.dimensions) == 2 {
						rightIndex = k*rightCols + col
					}
					dotProduct += leftTensor.values[leftIndex] * rightTensor.values[rightIndex]
				}
				resultIndex := batch*leftRows*rightCols + row*rightCols + col
				if len(outputDimensions) == 1 {
					resultIndex = col
				}
				resultTensor.values[resultIndex] = dotProduct
			}
		}
	}

	return resultTensor, nil
}

// Slice slices a tensor along a given dimension with start and end indices.
// It returns the resulting tensor and an error if the indices are invalid.
func Slice(t Tensor, start, end []int) (Tensor, error) {
	if len(start) != len(t.dimensions) || len(end) != len(t.dimensions) {
		return Tensor{}, fmt.Errorf("start and end indices must have the same number of dimensions as the tensor")
	}

	for i, dim := range t.dimensions {
		if start[i] < 0 || start[i] >= dim || end[i] <= start[i] || end[i] > dim {
			return Tensor{}, fmt.Errorf("invalid start or end index for dimension %d (%d, %d)", i, start[i], end[i])
		}
	}

	newDims := make([]int, len(t.dimensions))
	for i := range newDims {
		newDims[i] = end[i] - start[i]
	}

	newSize := 1
	for _, dim := range newDims {
		newSize *= dim
	}

	newValues := make([]float64, newSize)

	oldStrides := make([]int, len(t.dimensions))
	newStrides := make([]int, len(t.dimensions))
	oldStride, newStride := 1, 1

	for i := len(t.dimensions) - 1; i >= 0; i-- {
		oldStrides[i] = oldStride
		newStrides[i] = newStride
		oldStride *= t.dimensions[i]
		newStride *= newDims[i]
	}

	for i := range newValues {
		oldIndex := 0
		remaining := i

		for j := 0; j < len(newDims); j++ {
			dim := remaining / newStrides[j]
			remaining %= newStrides[j]
			oldIndex += (start[j] + dim) * oldStrides[j]
		}

		newValues[i] = t.values[oldIndex]
	}

	return Tensor{
		values:     newValues,
		dimensions: newDims,
	}, nil
}

// SetSlice sets a slice of the tensor to the given values.
// It takes start and end indices for each dimension, and a slice of values to set.
// The function returns an error if the indices are invalid or if the length of values
// does not match the size of the specified slice.
func (t *Tensor) SetSlice(start, end []int, values []float64) error {
	if len(start) != len(end) || len(start) != len(t.dimensions) {
		return fmt.Errorf("start and end indices must have the same number of dimensions as the tensor")
	}

	for i, dim := range t.dimensions {
		if start[i] < 0 || start[i] >= dim || end[i] <= start[i] || end[i] > dim {
			return fmt.Errorf("invalid start or end index for dimension %d (%d, %d)", i, start[i], end[i])
		}
	}

	newDims := make([]int, len(t.dimensions))
	for i := range newDims {
		newDims[i] = end[i] - start[i]
	}

	newSize := 1
	for _, dim := range newDims {
		newSize *= dim
	}

	if len(values) != newSize {
		return fmt.Errorf("length of values does not match the size of the slice")
	}

	oldStrides := make([]int, len(t.dimensions))
	newStrides := make([]int, len(t.dimensions))
	oldStride, newStride := 1, 1

	for i := len(t.dimensions) - 1; i >= 0; i-- {
		oldStrides[i] = oldStride
		newStrides[i] = newStride
		oldStride *= t.dimensions[i]
		newStride *= newDims[i]
	}

	for i := range values {
		oldIndex := 0
		remaining := i

		for j := 0; j < len(newDims); j++ {
			dim := remaining / newStrides[j]
			remaining %= newStrides[j]
			oldIndex += (start[j] + dim) * oldStrides[j]
		}

		t.values[oldIndex] = values[i]
	}

	return nil
}

// Rotate180 rotates a tensor 180 degrees along two given dimensions.
// It returns the resulting tensor and an error if the dimensions are invalid.
func Rotate180(t Tensor, dim1, dim2 int) (Tensor, error) {
	if dim1 < 0 || dim2 < 0 || dim1 >= len(t.dimensions) || dim2 >= len(t.dimensions) || dim1 == dim2 {
		return Tensor{}, fmt.Errorf("invalid dimensions for rotation")
	}

	newTensor := Tensor{
		values:     make([]float64, len(t.values)),
		dimensions: make([]int, len(t.dimensions)),
	}
	copy(newTensor.dimensions, t.dimensions)

	strides := make([]int, len(t.dimensions))
	stride := 1
	for i := len(t.dimensions) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= t.dimensions[i]
	}

	for i := range t.values {
		indices := make([]int, len(t.dimensions))
		remaining := i
		for j := 0; j < len(t.dimensions); j++ {
			indices[j] = remaining / strides[j]
			remaining %= strides[j]
		}

		indices[dim1] = t.dimensions[dim1] - 1 - indices[dim1]
		indices[dim2] = t.dimensions[dim2] - 1 - indices[dim2]

		newIndex := 0
		for j, idx := range indices {
			newIndex += idx * strides[j]
		}

		newTensor.values[newIndex] = t.values[i]
	}

	return newTensor, nil
}

// Flip flips a tensor along a given dimension.
// It returns the resulting tensor and an error if the dimension is invalid.
func Flip(t Tensor, dim int) (Tensor, error) {
	if dim < 0 || dim >= len(t.dimensions) {
		return Tensor{}, fmt.Errorf("invalid dimension for flipping")
	}

	newTensor := Tensor{
		values:     make([]float64, len(t.values)),
		dimensions: make([]int, len(t.dimensions)),
	}
	copy(newTensor.dimensions, t.dimensions)

	strides := make([]int, len(t.dimensions))
	stride := 1
	for i := len(t.dimensions) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= t.dimensions[i]
	}

	for i := range t.values {
		indices := make([]int, len(t.dimensions))
		remaining := i
		for j := 0; j < len(t.dimensions); j++ {
			indices[j] = remaining / strides[j]
			remaining %= strides[j]
		}

		indices[dim] = t.dimensions[dim] - 1 - indices[dim]

		newIndex := 0
		for j, idx := range indices {
			newIndex += idx * strides[j]
		}

		newTensor.values[newIndex] = t.values[i]
	}

	return newTensor, nil
}

// Convolve performs a convolution operation on a tensor with a kernel.
// It returns the resulting tensor and an error if the operation is not possible.
// The tensor should have shape [batch_size, in_channels, height, width].
// The kernel should have shape [out_channels, in_channels, kernel_height, kernel_width].
// Padding and stride should be provided for height and width dimensions.
// For 1D convolution, height, kernel height, padding height, and stride height
// should all be 1.
func Convolve(t Tensor, kernel Tensor, padding, stride []int) (Tensor, error) {
	if len(t.dimensions) != 4 || len(kernel.dimensions) != 4 {
		return Tensor{}, fmt.Errorf("tensor must have shape [batch_size, in_channels, height, width] and kernel must have shape [out_channels, in_channels, kernel_height, kernel_width]")
	}

	if len(padding) != 2 || len(stride) != 2 {
		return Tensor{}, fmt.Errorf("padding and stride must have 2 dimensions for height and width")
	}

	batchSize, inChannels, inHeight, inWidth := t.dimensions[0], t.dimensions[1], t.dimensions[2], t.dimensions[3]
	outChannels, _, kernelHeight, kernelWidth := kernel.dimensions[0], kernel.dimensions[1], kernel.dimensions[2], kernel.dimensions[3]

	if inChannels != kernel.dimensions[1] {
		return Tensor{}, fmt.Errorf("input channels must match kernel's in_channels")
	}

	// Calculate output dimensions
	outHeight := (inHeight+2*padding[0]-kernelHeight)/stride[0] + 1
	outWidth := (inWidth+2*padding[1]-kernelWidth)/stride[1] + 1

	if outHeight <= 0 || outWidth <= 0 {
		return Tensor{}, fmt.Errorf("invalid dimensions for convolution")
	}

	// Create output tensor
	output := Tensor{
		values:     make([]float64, batchSize*outChannels*outHeight*outWidth),
		dimensions: []int{batchSize, outChannels, outHeight, outWidth},
	}

	// Perform convolution
	for b := 0; b < batchSize; b++ {
		for oc := 0; oc < outChannels; oc++ {
			for oh := 0; oh < outHeight; oh++ {
				for ow := 0; ow < outWidth; ow++ {
					sum := 0.0
					for ic := 0; ic < inChannels; ic++ {
						for kh := 0; kh < kernelHeight; kh++ {
							for kw := 0; kw < kernelWidth; kw++ {
								ih := oh*stride[0] + kh - padding[0]
								iw := ow*stride[1] + kw - padding[1]
								if ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth {
									inputIdx := ((b*inChannels+ic)*inHeight+ih)*inWidth + iw
									kernelIdx := ((oc*inChannels+ic)*kernelHeight+kh)*kernelWidth + kw
									sum += t.values[inputIdx] * kernel.values[kernelIdx]
								}
							}
						}
					}
					outputIdx := ((b*outChannels+oc)*outHeight+oh)*outWidth + ow
					output.values[outputIdx] = sum
				}
			}
		}
	}

	return output, nil
}

// MaxPool performs a max pooling operation on a tensor with a kernel.
// It returns the resulting tensor and an error if the operation is not possible.
// The tensor, kernelSize, and stride must have the same number of dimensions.
// Zero values can be provided for padding.  If stride is 0, then the kernel size is
// used as the stride.
func MaxPool(t Tensor, kernelSize, stride []int) (Tensor, error) {
	if len(kernelSize) != len(t.dimensions) || len(stride) != len(t.dimensions) {
		return Tensor{}, fmt.Errorf("kernelSize and stride must have the same number of dimensions as the tensor")
	}

	// If stride is 0, then set to kernel size
	for i := range stride {
		if stride[i] == 0 {
			stride[i] = kernelSize[i]
		}
	}

	// Calculate output dimensions
	outputDims := make([]int, len(t.dimensions))
	for i := range t.dimensions {
		outputDims[i] = (t.dimensions[i]-kernelSize[i])/stride[i] + 1
		if outputDims[i] <= 0 {
			return Tensor{}, fmt.Errorf("invalid dimensions for max pooling")
		}
	}

	// Create output tensor
	output := Tensor{
		values:     make([]float64, bionet.CumProd(outputDims)),
		dimensions: outputDims,
	}

	// Perform max pooling
	for i := range output.values {
		outputIndices := bionet.GetIndices(i, output.dimensions)
		maxVal := math.Inf(-1)

		// Iterate over the kernel
		for j := 0; j < bionet.CumProd(kernelSize); j++ {
			kernelIndices := bionet.GetIndices(j, kernelSize)
			valid := true

			inputIndices := make([]int, len(t.dimensions))
			for k := range inputIndices {
				inputIndices[k] = outputIndices[k]*stride[k] + kernelIndices[k]
				if inputIndices[k] < 0 || inputIndices[k] >= t.dimensions[k] {
					valid = false
					break
				}
			}

			if valid {
				tIndex := bionet.GetFlatIndex(inputIndices, t.dimensions)
				maxVal = math.Max(maxVal, t.values[tIndex])
			}
		}

		output.values[i] = maxVal
	}

	return output, nil
}

// AvgPool performs an average pooling operation on a tensor with a kernel.
// It returns the resulting tensor and an error if the operation is not possible.
// The tensor, kernelSize, and stride must have the same number of dimensions.
// Zero values can be provided for padding. If stride is 0, then the kernel size is
// used as the stride.
func AvgPool(t Tensor, kernelSize, stride []int) (Tensor, error) {
	if len(kernelSize) != len(t.dimensions) || len(stride) != len(t.dimensions) {
		return Tensor{}, fmt.Errorf("kernelSize and stride must have the same number of dimensions as the tensor")
	}

	// If stride is 0, then set to kernel size
	for i := range stride {
		if stride[i] == 0 {
			stride[i] = kernelSize[i]
		}
	}

	// Calculate output dimensions
	outputDims := make([]int, len(t.dimensions))
	for i := range t.dimensions {
		outputDims[i] = (t.dimensions[i]-kernelSize[i])/stride[i] + 1
		if outputDims[i] <= 0 {
			return Tensor{}, fmt.Errorf("invalid dimensions for average pooling")
		}
	}

	// Create output tensor
	output := Tensor{
		values:     make([]float64, bionet.CumProd(outputDims)),
		dimensions: outputDims,
	}

	// Perform average pooling
	for i := range output.values {
		outputIndices := bionet.GetIndices(i, output.dimensions)
		sum := 0.0
		count := 0

		// Iterate over the kernel
		for j := 0; j < bionet.CumProd(kernelSize); j++ {
			kernelIndices := bionet.GetIndices(j, kernelSize)
			valid := true

			inputIndices := make([]int, len(t.dimensions))
			for k := range inputIndices {
				inputIndices[k] = outputIndices[k]*stride[k] + kernelIndices[k]
				if inputIndices[k] < 0 || inputIndices[k] >= t.dimensions[k] {
					valid = false
					break
				}
			}

			if valid {
				tIndex := bionet.GetFlatIndex(inputIndices, t.dimensions)
				sum += t.values[tIndex]
				count++
			}
		}

		if count > 0 {
			output.values[i] = sum / float64(count)
		} else {
			output.values[i] = 0
		}
	}

	return output, nil
}

// Ones creates a tensor of ones with the given dimensions.
func Ones(dims []int) Tensor {
	if len(dims) == 0 {
		return New(nil, nil)
	}

	out := New(make([]float64, bionet.CumProd(dims)), dims)
	out.Fill(1)
	return out
}

// Zeros creates a tensor of zeros with the given dimensions.
func Zeros(dims []int) Tensor {
	if len(dims) == 0 {
		return New(nil, nil)
	}

	return New(make([]float64, bionet.CumProd(dims)), dims)
}

// Rand creates a tensor of random values with the given dimensions and range.
func Rand(dims []int, min, max float64) Tensor {
	out := New(make([]float64, bionet.CumProd(dims)), dims)
	for i := range out.values {
		out.values[i] = min + rand.Float64()*(max-min)
	}
	return out
}

// Equal checks if two tensors are equal.  This comparison checks if the tensors
// have the same dimensions and the same values.
func Equal(t1, t2 Tensor) bool {
	return slices.Equal(t1.dimensions, t2.dimensions) && slices.Equal(t1.values, t2.values)
}

// EqualValues checks if the values of two tensors are equal.  This comparison
// does not check if the tensors have the same dimensions.
func EqualValues(t1, t2 Tensor) bool {
	return slices.Equal(t1.values, t2.values)
}

// InDelta checks if two tensors are within a delta of each other.  This comparison
// expects the tensors to have the same dimensions.
func InDelta(t1, t2 Tensor, delta float64) bool {
	return slices.Equal(t1.dimensions, t2.dimensions) && InDeltaValues(t1, t2, delta)
}

// InDeltaValues checks if the values of two tensors are within a delta of each other.
// This comparison does not check if the tensors have the same dimensions.
func InDeltaValues(t1, t2 Tensor, delta float64) bool {
	for i := range t1.values {
		if math.Abs(t1.values[i]-t2.values[i]) > delta {
			return false
		}
	}
	return true
}

// GobEncode encodes a tensor to a byte slice using gob.  This implements the
// necessary gob encoder interface for the non-exported struct fields.
func (t Tensor) GobEncode() ([]byte, error) {
	w := new(bytes.Buffer)
	enc := gob.NewEncoder(w)
	err := enc.Encode(struct {
		Values     []float64
		Dimensions []int
	}{
		Values:     t.values,
		Dimensions: t.dimensions,
	})
	if err != nil {
		return nil, err
	}
	return w.Bytes(), nil
}

// GobDecode decodes a byte slice to a tensor using gob.  This implements the
// necessary gob decoder interface for the non-exported struct fields.
func (t *Tensor) GobDecode(data []byte) error {
	r := bytes.NewBuffer(data)
	dec := gob.NewDecoder(r)
	var s struct {
		Values     []float64
		Dimensions []int
	}
	err := dec.Decode(&s)
	if err != nil {
		return err
	}
	t.values = s.Values
	t.dimensions = s.Dimensions
	return nil
}
