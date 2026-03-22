package runtime

// CumProd calculates the cumulative product of the slice values
func CumProd(values []int) int {
	result := 1
	for _, value := range values {
		result *= value
	}
	return result
}

// Helper function to get the indices of a flat index in a multi-dimensional tensor.
// This is used to map between an index on the values array and the corresponding
// index on the dimensions array.
func GetIndices(flatIndex int, dimensions []int) []int {
	indices := make([]int, len(dimensions))
	for i := len(dimensions) - 1; i >= 0; i-- {
		indices[i] = flatIndex % dimensions[i]
		flatIndex /= dimensions[i]
	}
	return indices
}

// Helper function to get the flat index from indices in a multi-dimensional tensor.
// This is used to map between an index on the dimensions array and the corresponding
// index on the flat values array.
func GetFlatIndex(indices []int, dimensions []int) int {
	flatIndex := 0
	multiplier := 1
	for i := len(dimensions) - 1; i >= 0; i-- {
		flatIndex += indices[i] * multiplier
		multiplier *= dimensions[i]
	}
	return flatIndex
}

// Helper function to add two slices of indices
func AddIndices(a, b []int) []int {
	result := make([]int, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}
