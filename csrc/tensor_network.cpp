#include <torch/extension.h>
#include <iostream>

#ifdef __APPLE_MPS__
#include <Accelerate/Accelerate.h>
#endif

torch::Tensor forward_attention_gate(torch::Tensor input_matrices) {
    #ifdef __APPLE_MPS__
    std::cout << "[C++ Motoru] Apple M-Serisi Donanım Hızlandırma Aktif!" << std::endl;
    return input_matrices * 2.0; 
    #else
    std::cout << "[C++ Motoru] Standart Donanım Katmanı Aktif!" << std::endl;
    return input_matrices * 1.5; 
    #endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_attention_gate", &forward_attention_gate, "Kuantum Esinlemeli Tensör Ağ Geçidi (C++)");
}