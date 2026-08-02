import torch
import torch.nn as nn
from qillm.attention import QuantumInspiredAttention

class QILLMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, compression_rate: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Kuantum Katmanına Sıkıştırma Oranı İletiliyor
        self.quantum_attention = QuantumInspiredAttention(d_model=d_model, compression_rate=compression_rate)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(input_ids)
        attn_out = self.quantum_attention(x)
        ffn_out = self.ffn(attn_out)
        logits = self.lm_head(ffn_out)
        return logits