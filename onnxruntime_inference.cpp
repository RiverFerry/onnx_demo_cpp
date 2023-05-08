
#include "onnxruntime_inference.h"

Inference::Inference(std::unique_ptr<Ort::Env>& env, const char* modelpath, int label_len) : env_(std::move(env)), modelpath_(modelpath), label_len_(label_len) {
    //LOGD("model path  %s ", modelpath_);
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_  = std::make_unique<Ort::Session>(*env_.get(), modelpath_, session_options);
    printNodes();
//    createInputBuffer();
    createInputBufferInt();
}

void Inference::createInputBuffer()
{
    //LOGD("creating onnxReq data buffer of size =  %lu", input_tensor_size);
    ort_inputs_data  = std::make_unique<float[]>(input_tensor_size);
    
    ouput_tensor_size = output_node_dims[output_node_dims.size()-1];
    //LOGD("creating ouput data buffer of size =  %lu", ouput_tensor_size);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory_info, ort_inputs_data.get(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());
    //LOGD("Inference::createInputBuffer() ok.");

}

void Inference::createInputBufferInt()
{
    //LOGD("creating onnxReq data buffer of size =  %lu", input_tensor_size);
    ort_inputs_data_int  = std::make_unique<int[]>(input_tensor_size);

    ouput_tensor_size = output_node_dims[output_node_dims.size()-1];
    //LOGD("creating ouput data buffer of size =  %lu", ouput_tensor_size);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<int>(memory_info, ort_inputs_data_int.get(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());


}

void Inference::printNodes() {
     Ort::AllocatorWithDefaultOptions allocator;
        
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names.reserve(num_input_nodes);
    
    
    //LOGD("Number of onnxReq =  %zu",num_input_nodes);
    
    for (int i = 0; i < num_input_nodes; i++){
        char* input_name = session_->GetInputName(i, allocator);
        //LOGD("Input %d : name = %s", i, input_name);
        input_node_names[i] = input_name;
        
        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        //LOGD("Input %d : type = %d", i, type);
        
        input_node_dims = tensor_info.GetShape();
        //LOGD("Input %d : num_dims=%zu", i, input_node_dims.size());
        
        input_tensor_size = 1;
        for (int j = 0;j < input_node_dims.size(); j++)
        {
            // by river
            input_node_dims[j] = input_node_dims[j] == -1 ? 1 : input_node_dims[j];
            //LOGD("Input %d : dim %d = %ld",i,j,input_node_dims[j]);
            input_tensor_size *= input_node_dims[j];
        }
    }
    
    
    
    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names.reserve(num_output_nodes);
    
    
    for (int i = 0; i < num_output_nodes; i++){
        char* output_name = session_->GetOutputName(i, allocator);
        //LOGD("Output %d : name = %s", i, output_name);
        output_node_names[i] = output_name;
        
        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        //LOGD("Output %d : type = %d", i, type);
        
        output_node_dims = tensor_info.GetShape();
        //LOGD("Output %d : num_dims=%zu", i, output_node_dims.size());

        for (int j = 0;j < output_node_dims.size(); j++)
        {
            output_node_dims[j] = output_node_dims[j] == -1 ? 1 : output_node_dims[j];
            //LOGD("Output %d : dim %d = %ld ",i,j,output_node_dims[j]);
        }
    }
}


float* Inference::run(float* values){
    float *input_p = ort_inputs_data.get();
    memcpy(input_p, values, input_tensor_size * sizeof(float));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    float* probilities = output_tensors[1].GetTensorMutableData<float>();
    //LOGD("run\n");
    return probilities;



}

int * Inference::run(int * values){
    float *input_p = ort_inputs_data.get();
    memcpy(input_p, values, input_tensor_size * sizeof(int ));

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    int * probilities = output_tensors[1].GetTensorMutableData<int>();
    //LOGD("run\n");
    return probilities;



}

int Inference::getLabelLen() {
    return label_len_;
}

Inference::~Inference(){
    delete modelpath_;
}
