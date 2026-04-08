//go:build torchscript_native && cgo

package bindingnative

/*
#cgo CXXFLAGS: -std=c++17
#cgo LDFLAGS: -ltorch -lc10 -ltorch_cpu
#include <stdlib.h>
#include "torchscript.hpp"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// Module wraps a loaded native TorchScript module.
type Module struct {
	ptr C.TorchScriptModule
}

// LoadModule loads a TorchScript model file into a native libtorch module.
func LoadModule(path string) (*Module, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cErr C.TorchError
	modulePtr := C.TorchJitLoadModel(cPath, &cErr)
	if cErr.message != nil {
		defer C.TorchFreeCString(cErr.message)
		return nil, fmt.Errorf("%s", C.GoString(cErr.message))
	}

	return &Module{ptr: modulePtr}, nil
}

// ForwardTextClassification runs a batch of token id and attention mask inputs.
func (m *Module) ForwardTextClassification(inputIDs, attentionMasks [][]int64) ([][]float64, error) {
	if m == nil || m.ptr == nil {
		return nil, fmt.Errorf("torchscript module is nil")
	}

	if len(inputIDs) != len(attentionMasks) {
		return nil, fmt.Errorf("input ids batch and attention mask batch must have same size")
	}

	if len(inputIDs) == 0 {
		return [][]float64{}, nil
	}

	seqLen := len(inputIDs[0])
	if seqLen == 0 {
		return nil, fmt.Errorf("sequence length must be greater than zero")
	}

	flatInputIDs := make([]C.longlong, 0, len(inputIDs)*seqLen)
	flatAttentionMasks := make([]C.longlong, 0, len(attentionMasks)*seqLen)

	for i := range inputIDs {
		if len(inputIDs[i]) != seqLen || len(attentionMasks[i]) != seqLen {
			return nil, fmt.Errorf("all sequences must share the same padded length")
		}

		for j := 0; j < seqLen; j++ {
			flatInputIDs = append(flatInputIDs, C.longlong(inputIDs[i][j]))
			flatAttentionMasks = append(flatAttentionMasks, C.longlong(attentionMasks[i][j]))
		}
	}

	var cErr C.TorchError
	result := C.TorchJitForwardTextClassification(
		m.ptr,
		(*C.longlong)(unsafe.Pointer(&flatInputIDs[0])),
		(*C.longlong)(unsafe.Pointer(&flatAttentionMasks[0])),
		C.int(len(inputIDs)),
		C.int(seqLen),
		&cErr,
	)
	if cErr.message != nil {
		defer C.TorchFreeCString(cErr.message)
		return nil, fmt.Errorf("%s", C.GoString(cErr.message))
	}
	defer C.TorchFreeFloatArray(result)

	if int(result.cols) == 0 {
		return nil, fmt.Errorf("torchscript forward returned zero columns")
	}

	flat := unsafe.Slice((*C.float)(result.data), int(result.size))
	output := make([][]float64, int(result.rows))
	for row := 0; row < int(result.rows); row++ {
		start := row * int(result.cols)
		output[row] = make([]float64, int(result.cols))
		for col := 0; col < int(result.cols); col++ {
			output[row][col] = float64(flat[start+col])
		}
	}

	return output, nil
}

// ForwardFeatureScoring runs engineered numeric feature vectors plus a shared
// message vector through the native TorchScript module.
func (m *Module) ForwardFeatureScoring(vectors [][]float64, message []float64) ([]float64, error) {
	if m == nil || m.ptr == nil {
		return nil, fmt.Errorf("torchscript module is nil")
	}

	if len(vectors) == 0 {
		return nil, fmt.Errorf("vectors batch must not be empty")
	}
	if len(message) == 0 {
		return nil, fmt.Errorf("message vector must not be empty")
	}

	featureSize := len(vectors[0])
	if featureSize == 0 {
		return nil, fmt.Errorf("feature vector size must be greater than zero")
	}

	flatVectors := make([]C.float, 0, len(vectors)*featureSize)
	for i := range vectors {
		if len(vectors[i]) != featureSize {
			return nil, fmt.Errorf("all feature vectors must share the same size")
		}
		for _, value := range vectors[i] {
			flatVectors = append(flatVectors, C.float(value))
		}
	}

	flatMessage := make([]C.float, len(message))
	for i, value := range message {
		flatMessage[i] = C.float(value)
	}

	var cErr C.TorchError
	result := C.TorchJitForwardFeatureScoring(
		m.ptr,
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		(*C.float)(unsafe.Pointer(&flatMessage[0])),
		C.int(len(vectors)),
		C.int(featureSize),
		C.int(len(message)),
		&cErr,
	)
	if cErr.message != nil {
		defer C.TorchFreeCString(cErr.message)
		return nil, fmt.Errorf("%s", C.GoString(cErr.message))
	}
	defer C.TorchFreeFloatArray(result)

	if int(result.rows) == 0 {
		return nil, fmt.Errorf("torchscript forward returned zero rows")
	}
	if int(result.cols) != 1 {
		return nil, fmt.Errorf("torchscript feature scoring expected one score per row, got %d columns", int(result.cols))
	}

	flat := unsafe.Slice((*C.float)(result.data), int(result.size))
	output := make([]float64, int(result.rows))
	for row := 0; row < int(result.rows); row++ {
		output[row] = float64(flat[row])
	}

	return output, nil
}

// Close releases the native libtorch module.
func (m *Module) Close() error {
	if m == nil || m.ptr == nil {
		return nil
	}

	C.TorchJitFreeModel(m.ptr)
	m.ptr = nil
	return nil
}
