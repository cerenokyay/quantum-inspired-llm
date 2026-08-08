#include <torch/extension.h>
#include <iostream>

#ifdef __APPLE_MPS__
#include <Accelerate/Accelerate.h>
#endif

// Kuantum Tensör Ağları: Dinamik SVD Sıkıştırması ve Güvenli Büzme
torch::Tensor compress_and_forward(torch::Tensor query, torch::Tensor key, double compression_rate) {
    
    // 1. Tip ve Boyut Kontrolü (Production-Grade Safety Check)
    TORCH_CHECK(query.dim() == 3, "Query tensörü 3 boyutlu olmalıdır: [Batch, SeqLen, Dim]");
    TORCH_CHECK(key.dim() == 3, "Key tensörü 3 boyutlu olmalıdır: [Batch, SeqLen, Dim]");
    TORCH_CHECK(compression_rate > 0.0 && compression_rate <= 1.0, "Sıkıştırma oranı 0.0 ile 1.0 arasında olmalıdır.");

    // 2. Matris Çarpımı (Q x K^T)
    torch::Tensor attention_scores = torch::matmul(query, key.transpose(-2, -1));
    
    // 3. Kuantum Esinlemeli SVD (Tekil Değer Ayrışımı)
    auto svd_result = torch::linalg_svd(attention_scores, /*full_matrices=*/false);
    
    torch::Tensor U = std::get<0>(svd_result);
    torch::Tensor S = std::get<1>(svd_result);
    torch::Tensor V = std::get<2>(svd_result);

    // 4. Dinamik Kuantum Budama (Dynamic Truncation)
    int max_k = S.size(-1);
    int k = static_cast<int>(max_k * compression_rate);
    if (k < 1) k = 1; // En az 1 bileşen korunmalı

    torch::Tensor S_truncated = S.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor U_truncated = U.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor V_truncated = V.slice(/*dim=*/-2, /*start=*/0, /*end=*/k);

    // 5. Tensör Birleştirme (Tensor Contraction)
    torch::Tensor compressed_scores = torch::matmul(U_truncated, torch::matmul(torch::diag_embed(S_truncated), V_truncated));

    return compressed_scores;
}

// --- EKLENECEK YENİ FONKSİYON (Platform Statik Sıkıştırması İçin) ---
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd_compress_weight(torch::Tensor weight, double compression_rate) {
    // 1. Tip ve Boyut Kontrolü
    TORCH_CHECK(weight.dim() == 2, "Agirlik matrisi 2 boyutlu olmalidir.");
    TORCH_CHECK(compression_rate > 0.0 && compression_rate <= 1.0, "Sikistirma orani 0.0 ile 1.0 arasinda olmalidir.");

    // 2. Kuantum Esinlemeli SVD
    auto svd_result = torch::linalg_svd(weight, /*full_matrices=*/false);
    
    torch::Tensor U = std::get<0>(svd_result);
    torch::Tensor S = std::get<1>(svd_result);
    torch::Tensor V = std::get<2>(svd_result);

    // 3. Budama (Truncation)
    int max_k = S.size(-1);
    int k = static_cast<int>(max_k * compression_rate);
    if (k < 1) k = 1;

    torch::Tensor U_truncated = U.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor S_truncated = S.slice(/*dim=*/-1, /*start=*/0, /*end=*/k);
    torch::Tensor V_truncated = V.slice(/*dim=*/-2, /*start=*/0, /*end=*/k);

    return std::make_tuple(U_truncated, S_truncated, V_truncated);
}

// --- PYBIND11 BLOĞUNUN EN SON VE DOĞRU HALİ ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 1. Senin orijinal dinamik fonksiyonun (Hiçbir işlevini kaybetmedi)
    m.def("compress_and_forward", &compress_and_forward, 
          "Dinamik Oranli Kuantum Tensor SVD Sikistirma Motoru (C++)",
          py::arg("query"), py::arg("key"), py::arg("compression_rate") = 0.5);
          
    // 2. Yeni eklediğimiz statik model sıkıştırma fonksiyonu
    m.def("svd_compress", &svd_compress_weight, 
          "2D Sabit Agirlik Matrisleri Icin SVD (C++)",
          py::arg("weight"), py::arg("compression_rate") = 0.5);
}