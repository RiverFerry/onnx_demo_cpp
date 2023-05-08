g++ main.cpp -std=c++14 -I ./include/ -L ./lib  -lonnxruntime  -o onnx -g

LD_LIBRARY_PATH=./lib ./onnx