import torch
import torch.nn as nn
from qillm.attention import QuantumInspiredAttention

class QILLMLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 1. Kelime Vektörleştirme (Token Embedding)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. Bizim C++ Destekli Kuantum Attention Katmanımız
        self.quantum_attention = QuantumInspiredAttention(d_model=d_model)
        
        # 3. İleri Beslemeli Sinir Ağı (Feed-Forward Network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # 4. Kelime Tahmin Katmanı (LM Head)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids):
        # input_ids: Kelime ID'lerinden oluşan matris [Batch_Size, Sequence_Length]
        
        # Kelimeleri vektöre dönüştür
        x = self.token_embedding(input_ids)
        
        # C++ Kuantum SVD katmanından geçir
        attn_out = self.quantum_attention(x)
        
        # Sinir ağından geçir
        ffn_out = self.ffn(attn_out)
        
        # Bir sonraki kelime olasılıklarını üret
        logits = self.lm_head(ffn_out)
        
        return logits