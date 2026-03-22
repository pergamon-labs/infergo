package runtime

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCumprod(t *testing.T) {
	testCases := []struct {
		name       string
		dimensions []int
		expected   int
	}{
		{
			name:       "Empty slice",
			dimensions: []int{},
			expected:   1,
		},
		{
			name:       "Single dimension",
			dimensions: []int{5},
			expected:   5,
		},
		{
			name:       "Two dimensions",
			dimensions: []int{3, 4},
			expected:   12,
		},
		{
			name:       "Three dimensions",
			dimensions: []int{2, 3, 4},
			expected:   24,
		},
		{
			name:       "Dimensions with 1",
			dimensions: []int{1, 5, 1, 3},
			expected:   15,
		},
		{
			name:       "Large dimensions",
			dimensions: []int{10, 20, 30},
			expected:   6000,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := CumProd(tc.dimensions)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestGetIndices(t *testing.T) {
	testCases := []struct {
		name       string
		flatIndex  int
		dimensions []int
		expected   []int
	}{
		{
			name:       "1D array",
			flatIndex:  3,
			dimensions: []int{5},
			expected:   []int{3},
		},
		{
			name:       "2D array",
			flatIndex:  5,
			dimensions: []int{3, 4},
			expected:   []int{1, 1},
		},
		{
			name:       "3D array",
			flatIndex:  11,
			dimensions: []int{2, 3, 4},
			expected:   []int{0, 2, 3},
		},
		{
			name:       "4D array",
			flatIndex:  23,
			dimensions: []int{2, 2, 3, 4},
			expected:   []int{0, 1, 2, 3},
		},
		{
			name:       "Zero flat index",
			flatIndex:  0,
			dimensions: []int{3, 3, 3},
			expected:   []int{0, 0, 0},
		},
		{
			name:       "Large flat index",
			flatIndex:  999,
			dimensions: []int{10, 10, 10},
			expected:   []int{9, 9, 9},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := GetIndices(tc.flatIndex, tc.dimensions)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestGetFlatIndex(t *testing.T) {
	testCases := []struct {
		name       string
		indices    []int
		dimensions []int
		expected   int
	}{
		{
			name:       "1D array",
			indices:    []int{3},
			dimensions: []int{5},
			expected:   3,
		},
		{
			name:       "2D array",
			indices:    []int{1, 1},
			dimensions: []int{3, 4},
			expected:   5,
		},
		{
			name:       "3D array",
			indices:    []int{0, 2, 3},
			dimensions: []int{2, 3, 4},
			expected:   11,
		},
		{
			name:       "4D array",
			indices:    []int{0, 1, 2, 3},
			dimensions: []int{2, 2, 3, 4},
			expected:   23,
		},
		{
			name:       "Zero indices",
			indices:    []int{0, 0, 0},
			dimensions: []int{3, 3, 3},
			expected:   0,
		},
		{
			name:       "Large indices",
			indices:    []int{9, 9, 9},
			dimensions: []int{10, 10, 10},
			expected:   999,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := GetFlatIndex(tc.indices, tc.dimensions)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestAddIndices(t *testing.T) {
	testCases := []struct {
		name     string
		a        []int
		b        []int
		expected []int
	}{
		{
			name:     "Empty slices",
			a:        []int{},
			b:        []int{},
			expected: []int{},
		},
		{
			name:     "Single element slices",
			a:        []int{1},
			b:        []int{2},
			expected: []int{3},
		},
		{
			name:     "Multiple element slices",
			a:        []int{1, 2, 3},
			b:        []int{4, 5, 6},
			expected: []int{5, 7, 9},
		},
		{
			name:     "Slices with negative numbers",
			a:        []int{-1, 0, 1},
			b:        []int{1, 0, -1},
			expected: []int{0, 0, 0},
		},
		{
			name:     "Slices with large numbers",
			a:        []int{1000000, 2000000},
			b:        []int{3000000, 4000000},
			expected: []int{4000000, 6000000},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := AddIndices(tc.a, tc.b)
			assert.Equal(t, tc.expected, result)
		})
	}
}
