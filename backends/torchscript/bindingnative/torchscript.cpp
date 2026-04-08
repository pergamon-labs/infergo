#include "torchscript.hpp"

#include <cstring>
#include <stdexcept>
#include <torch/script.h>

extern "C" {

TorchScriptModule TorchJitLoadModel(const char* path, TorchError* error) {
    if (error == nullptr) {
        throw std::invalid_argument("error pointer is null");
    }

    try {
        auto* model = new torch::jit::script::Module(torch::jit::load(path));
        return static_cast<TorchScriptModule>(model);
    } catch (const std::exception& e) {
        error->message = strdup(e.what());
        return nullptr;
    }
}

void TorchJitFreeModel(TorchScriptModule module) {
    delete static_cast<torch::jit::script::Module*>(module);
}

TorchFloatArray TorchJitForwardTextClassification(
    TorchScriptModule module,
    long long* input_ids,
    long long* attention_mask,
    int batch_dim,
    int seq_dim,
    TorchError* error
) {
    TorchFloatArray result{nullptr, 0, 0, 0};

    if (error == nullptr) {
        throw std::invalid_argument("error pointer is null");
    }

    if (module == nullptr) {
        error->message = strdup("module pointer is null");
        return result;
    }

    if (input_ids == nullptr || attention_mask == nullptr) {
        error->message = strdup("input_ids or attention_mask pointer is null");
        return result;
    }

    if (batch_dim == 0 || seq_dim == 0) {
        error->message = strdup("batch_dim and seq_dim must be greater than zero");
        return result;
    }

    try {
        auto& model_ref = *static_cast<torch::jit::script::Module*>(module);
        torch::NoGradGuard no_grad;

        auto input_tensor = torch::from_blob(input_ids, {batch_dim, seq_dim}, torch::kInt64).clone();
        auto mask_tensor = torch::from_blob(attention_mask, {batch_dim, seq_dim}, torch::kInt64).clone();

        std::vector<torch::jit::IValue> inputs = {input_tensor, mask_tensor};
        torch::Tensor output_tensor = model_ref.forward(inputs).toTensor().detach().cpu();

        int rows = 1;
        int cols = 0;
        if (output_tensor.dim() == 1) {
            cols = static_cast<int>(output_tensor.size(0));
        } else {
            rows = static_cast<int>(output_tensor.size(0));
            cols = static_cast<int>(output_tensor.size(1));
        }

        int size = rows * cols;
        float* output = static_cast<float*>(malloc(sizeof(float) * size));
        if (output == nullptr) {
            error->message = strdup("failed to allocate output buffer");
            return result;
        }

        std::memcpy(output, output_tensor.data_ptr(), sizeof(float) * size);
        result.data = output;
        result.size = size;
        result.rows = rows;
        result.cols = cols;
        return result;
    } catch (const std::exception& e) {
        error->message = strdup(e.what());
        return result;
    }
}

TorchFloatArray TorchJitForwardFeatureScoring(
    TorchScriptModule module,
    float* vectors,
    float* message,
    int batch_dim,
    int feature_dim,
    int message_dim,
    TorchError* error
) {
    TorchFloatArray result{nullptr, 0, 0, 0};

    if (error == nullptr) {
        throw std::invalid_argument("error pointer is null");
    }

    if (module == nullptr) {
        error->message = strdup("module pointer is null");
        return result;
    }

    if (vectors == nullptr || message == nullptr) {
        error->message = strdup("vectors or message pointer is null");
        return result;
    }

    if (batch_dim == 0 || feature_dim == 0 || message_dim == 0) {
        error->message = strdup("batch_dim, feature_dim, and message_dim must be greater than zero");
        return result;
    }

    try {
        auto& model_ref = *static_cast<torch::jit::script::Module*>(module);
        torch::NoGradGuard no_grad;

        auto vectors_tensor = torch::from_blob(vectors, {batch_dim, feature_dim}, torch::kFloat32).clone();
        auto message_tensor = torch::from_blob(message, {message_dim}, torch::kFloat32).clone();

        std::vector<torch::jit::IValue> inputs = {vectors_tensor, message_tensor};
        torch::Tensor output_tensor = model_ref.forward(inputs).toTensor().detach().cpu();

        if (output_tensor.dim() == 1) {
            output_tensor = output_tensor.unsqueeze(1);
        }

        if (output_tensor.dim() != 2 || output_tensor.size(1) != 1) {
            error->message = strdup("feature scoring forward must return shape [batch] or [batch, 1]");
            return result;
        }

        int rows = static_cast<int>(output_tensor.size(0));
        int cols = static_cast<int>(output_tensor.size(1));
        int size = rows * cols;
        float* output = static_cast<float*>(malloc(sizeof(float) * size));
        if (output == nullptr) {
            error->message = strdup("failed to allocate output buffer");
            return result;
        }

        std::memcpy(output, output_tensor.data_ptr(), sizeof(float) * size);
        result.data = output;
        result.size = size;
        result.rows = rows;
        result.cols = cols;
        return result;
    } catch (const std::exception& e) {
        error->message = strdup(e.what());
        return result;
    }
}

void TorchFreeFloatArray(TorchFloatArray array) {
    if (array.data != nullptr) {
        free(array.data);
    }
}

void TorchFreeCString(char* value) {
    if (value != nullptr) {
        free(value);
    }
}

}
