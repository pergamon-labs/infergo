#pragma once
#ifndef INFERGO_TORCHSCRIPT_H
#define INFERGO_TORCHSCRIPT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* TorchScriptModule;

typedef struct TorchError {
    char* message;
} TorchError;

typedef struct TorchFloatArray {
    float* data;
    int size;
    int rows;
    int cols;
} TorchFloatArray;

TorchScriptModule TorchJitLoadModel(const char* path, TorchError* error);
void TorchJitFreeModel(TorchScriptModule module);
TorchFloatArray TorchJitForwardTextClassification(
    TorchScriptModule module,
    long long* input_ids,
    long long* attention_mask,
    int batch_dim,
    int seq_dim,
    TorchError* error
);
TorchFloatArray TorchJitForwardFeatureScoring(
    TorchScriptModule module,
    float* vectors,
    float* message,
    int batch_dim,
    int feature_dim,
    int message_dim,
    TorchError* error
);
void TorchFreeFloatArray(TorchFloatArray array);
void TorchFreeCString(char* value);

#ifdef __cplusplus
}
#endif

#endif
