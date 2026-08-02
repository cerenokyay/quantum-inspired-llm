import torch
import torch.nn as nn
import math
import qillm_cpp

class QuantumInspiredAttention(nn.Module):
    def __init__(self, d_model: int, compression_rate: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.compression_rate = compression_rate
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        # C++ Kuantum SVD motoruna dinamik sıkıştırma oranımızı gönderiyoruz
        compressed_attention_scores = qillm_cpp.compress_and_forward(Q, K, self.compression_rate)
        
        attention_weights = torch.softmax(compressed_attention_scores / math.sqrt(self.d_model), dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output