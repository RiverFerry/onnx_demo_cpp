

#ifndef onnxruntime_inference_hpp
#define onnxruntime_inference_hpp

#include <stdio.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"



class Inference {
    
private:

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    const char* modelpath_;
    int label_len_;

    unsigned long input_tensor_size;
    unsigned long ouput_tensor_size;
    
    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    
    Ort::Value input_tensor{nullptr};
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    
    std::unique_ptr<float[]> ort_inputs_data;
    std::unique_ptr<int[]> ort_inputs_data_int;
    
    void createInputBuffer();
    void createInputBufferInt();
    void printNodes();



public:

    Inference(std::unique_ptr<Ort::Env>& env,  const char*  modelpath, int label_len);
    Inference(const Inference& ) = delete; //no copy
    Inference& operator = (const Inference &) = delete;//no copy
    float* run(float* values);
    int * run(int * values);
    int getLabelLen();
  
    ~Inference();
    
};

#endif /* onnxruntime_inference_hpp */
