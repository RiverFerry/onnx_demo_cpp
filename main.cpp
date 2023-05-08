#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <memory>
#include <cstdio>
using namespace std;

class onnxWrapper {
private:
    void readModelShape() {
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        input_node_names.reserve(num_input_nodes);
        cout << "num_output_nodes = " << num_input_nodes << endl;

        for (int i = 0; i < num_input_nodes; i++){
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            printf("Input %d : name = %s\n", i, input_name.get());
            inputNodeNameAllocatedStrings.push_back(std::move(input_name));
            input_node_names.push_back(inputNodeNameAllocatedStrings.back().get());

            Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            ONNXTensorElementDataType type = tensor_info.GetElementType();
            printf("Input %d : type = %d\n", i, type);
            input_node_dims = tensor_info.GetShape();
            printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
            input_tensor_size = 1;
            for (auto& it : input_node_dims)
            {
                if (it == -1)
                    it = 1;

                input_tensor_size *= it;
            }
        }
        
        size_t num_output_nodes = session_->GetOutputCount();
        output_node_names.reserve(num_output_nodes);
        
        cout << "num_output_nodes = " << num_output_nodes << endl;
        for (int i = 0; i < num_output_nodes; i++){
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            printf("Output %d : name = %s\n", i, output_name.get());
            outputNodeNameAllocatedStrings.push_back(std::move(output_name));
            output_node_names.push_back(outputNodeNameAllocatedStrings.back().get());
            Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            printf("Output %d : type = %d\n", i, tensor_info.GetElementType());
            
            auto output_node_dims = tensor_info.GetShape();
            printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());

            cout << "dim: ";
            for (auto& it : output_node_dims)
            {
                if (it == -1)
                    it = 1;

                cout << it << " ";
            }

            cout << endl;
        }
}

public:
void init(string model_path) {
        std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_ERROR,"test"));
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_  = make_unique<Ort::Session>(*environment, model_path.c_str(), session_options);
        cout << "=========  create session done   ===========" << endl;
        readModelShape();
}

void predict(vector<int>& input) {
    cout << "size of input = " << input.size() << endl;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<int>(memory_info, input.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    cout << "output size is " << output_tensors.size() << endl;
    // int * probilities0 = output_tensors[0].GetTensorMutableData<int>();
    // float * probilities1 = output_tensors[1].GetTensorMutableData<float>();
}

private:
    int input_tensor_size = 0;
    std::vector<int64_t> input_node_dims;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::unique_ptr<Ort::Session> session_;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings;
};



int main() {
    vector<int> input1 = {1452, 1008,  388,  777, 1277,  689,  777, 1208,  860,  583, 1230,
            351, 1352, 1287, 1519,  886, 1396,  777, 1208, 1139, 1139,  162,
            108,  562,  172,  567, 1139,  736, 1352,  442, 1394,  499, 1139,
            562, 1352, 1139,  736, 1511,  480, 1121, 1416, 1321, 1394,  499,
        1238,  373, 1326, 1121, 1416, 1321, 1139,  777, 1208,  120,  396,
            736,  736,  442,  736, 1427,   72,  190,  135,  204,   72,  860,
            583, 1427, 1204,  699, 1004,  853,  132,  329, 1186,  781, 1126,
            162,  781,  562,  172,  567, 1139,  736, 1352,  736,  562,  499,
            108,  781, 1352,  108,  442,  956,  283,  148,  137,  227,  791,
            526, 1535, 1393,  789,  814,  639, 1171, 1427, 1220,  905,  246,
            401,  891, 1008,  388,  777, 1277,   40,  180,  180,  814,  841,
        1199, 1113,  643,  182,  829,  829,  981, 1352,  509,  509,  829,
            176,  854,  829,  896,  718,  981,  176,  854,   28,  509,  720,
            727, 1154, 1054, 1273,  268,  956,    0};

    vector<int> input2 = {1452, 1008,  388,  777, 1277,  689,  777, 1208,  860,  583, 1230,
            351, 1352, 1287, 1519,  886, 1396,  777, 1208, 1139, 1139,  162,
            108,  562,  172,  567, 1139,  736, 1352,  442, 1394,  499, 1139,
            562, 1352, 1139,  736, 1511,  480, 1121, 1416, 1321, 1394,  499,
        1238,  373, 1326, 1121, 1416, 1321, 1139,  777, 1208,  120,  396,
            736,  736,  442,  736, 1427,   72,  190,  135,  204,   72,  860,
            583, 1427, 1204,  699, 1004,  853,  132,  329, 1186,  781, 1126,
            162,  781,  562,  172,  567, 1139,  736, 1352,  736,  562,  499,
            108,  781, 1352,  108,  442,  956,  283,  148,  137,  227,  791,
            526, 1535, 1393,  789,  814,  639, 1171, 1427, 1220,  905,  246,
            401,  891, 1008,  388,  777, 1277,   40,  180,  180,  814,  841,
        1199, 1113,  643,  182,  829,  829,  981, 1352,  509,  509,  829,
            176,  854,  829,  896,  718,  981,  176,  854,   28,  509,  720,
            727, 1154, 1054, 1273,  268,  956,    0};

    onnxWrapper onnx;
    onnx.init("./test.onnx");
    onnx.predict(input1);
    onnx.predict(input2);

    return 0;
}