import torch
import torch.nn as nn
import torch.optim as optim
from qillm.model import QILLMLanguageModel

print("--- 🏋️ KUANTUM ESİNLEMELİ LLM EĞİTİM DÖNGÜSÜ ---")

# 1. Konfigürasyon
vocab_size = 50   # 50 kelimelik mini bir sözlük
d_model = 16      # Vektör boyutu
epochs = 20       # Eğitimin kaç tur döneceği
lr = 0.01         # Öğrenme oranı (Learning Rate)

# 2. Modeli ve Optimizatörü (AdamW) Tanımlayalım
model = QILLMLanguageModel(vocab_size=vocab_size, d_model=d_model)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 3. Eğiteceğimiz Örnek Veri Seti (Target, Input'un 1 adım kaydırılmış halidir)
# Örn: Input = [1, 2, 3, 4] -> Target = [2, 3, 4, 5]
input_ids = torch.tensor([[1, 5, 12, 23, 40, 8]])
target_ids = torch.tensor([[5, 12, 23, 40, 8, 15]])  # Gerçekte gelmesi gereken sonraki kelimeler

print(f"Girdi Cümlesi ID'leri : {input_ids.tolist()[0]}")
print(f"Hedef Cümle ID'leri   : {target_ids.tolist()[0]}\n")

print("🚀 Eğitim Başlatılıyor...\n")

# 4. Eğitim Döngüsü (Training Loop)
model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()  # Gradient sıfırlama
    
    # İleri Besleme (C++ Kuantum SVD Motoru Tetikleniyor)
    logits = model(input_ids)
    
    # Hata (Loss) Hesabı
    # logits: [Batch, Seq, Vocab] -> Reshape: [Batch * Seq, Vocab]
    loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
    
    # Geriye Yayılım ve Ağırlık Güncelleme
    loss.backward()
    optimizer.step()
    
    # Her 5 adımda bir durumu görelim
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ---> Kayıp (Loss): {loss.item():.4f}")

print("\n✅ Eğitim Başarıyla Tamamlandı!")