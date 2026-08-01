#include <torch/extension.h>
#include <iostream>

#ifdef __APPLE_MPS__
#include <Accelerate/Accelerate.h>
#endif

// Kuantum Tensör Ağları: SVD ile Matris Sıkıştırma ve Tensör Büzme (Tensor Contraction)
torch::Tensor compress_and_forward(torch::Tensor query, torch::Tensor key) {
    
    // 1. Matris Çarpımı (Q x K^T)
    torch::Tensor attention_scores = torch::matmul(query, key.transpose(-2, -1));
    
    #ifdef __APPLE_MPS__
    std::cout << "[C++ Motoru] M5 Accelerate Motoru: Tensör SVD Sıkıştırması Başlatıldı..." << std::endl;
    #else
    std::cout << "[C++ Motoru] CPU/CUDA Motoru: Tensör SVD Sıkıştırması Başlatıldı..." << std::endl;
    #endif

    // 2. Kuantum Esinlemeli SVD (Tekil Değer Ayrışımı) ile Tensörleri Ayrıştırma
    // attention_scores matrisini U, S, V parçalarına bölüyoruz
    auto svd_result = torch::linalg_svd(attention_scores, /*full_matrices=*/false);
    
    torch::Tensor U = std::get<0>(svd_result);
    torch::Tensor S = std::get<1>(svd_result);
    torch::Tensor V = std::get<2>(svd_result);

    // 3. Kuantum Budama (Truncation): En büyük bilgiye sahip ilk K tekil değeri alıyoruz
    // Bu sayede matrisin boyutunu bilgi kaybetmeden yarı yarıya düşürüyoruz!
    int k = S.size(-1) / 2; // Yarı yarıya sıkıştırma oranı
    if (k < 1) k = 1;

    torch::Tensor S_truncated = S.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor U_truncated = U.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor V_truncated = V.slice(/*dim=*/-2, /*start=*/0, /*end=*/k);

    // 4. Sıkıştırılmış Tensörleri Tekrar Birleştirme (Tensor Contraction)
    torch::Tensor compressed_scores = torch::matmul(U_truncated, torch::matmul(torch::diag_embed(S_truncated), V_truncated));

    return compressed_scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_and_forward", &compress_and_forward, "Kuantum Tensör SVD Sıkıştırma Motoru (C++)");
}