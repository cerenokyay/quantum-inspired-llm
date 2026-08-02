import torch
import torch.nn as nn
import torch.optim as optim
from qillm.model import QILLMLanguageModel
from qillm.tokenizer import SimpleTokenizer

print("--- 🏋️ PROFESYONEL KUANTUM LLM EĞİTİMİ (TÜRKÇE METİN) ---")

# 1. Örnek Eğitim Metni
sample_text = "kuantum yapay zeka donanim optimizasyonu ile gelecegin dil modellerini hizlandirir"

# 2. Tokenizer'ı Kur ve Metne Göre Eğit
tokenizer = SimpleTokenizer()
tokenizer.fit_on_text(sample_text)

print(f"Sözlük Oluşturuldu! Toplam Kelime Sayısı (Vocab Size): {tokenizer.vocab_size}")

# 3. Metni Token ID'lerine Çevir
tokens = tokenizer.encode(sample_text)

# Input: "kuantum yapay zeka donanim..." -> Target: "...yapay zeka donanim optimizasyonu..."
input_ids = torch.tensor([tokens[:-1]])
target_ids = torch.tensor([tokens[1:]])

# 4. Modeli Oluştur (Dinamik %50 Kuantum Sıkıştırma Oranı ile)
d_model = 32
model = QILLMLanguageModel(vocab_size=tokenizer.vocab_size, d_model=d_model, compression_rate=0.5)

optimizer = optim.AdamW(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 5. Eğitim Döngüsü
epochs = 40
model.train()
print("\n🚀 C++ Motoru ile Eğitim Başlatılıyor...\n")

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits.view(-1, tokenizer.vocab_size), target_ids.view(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ---> Kayıp (Loss): {loss.item():.4f}")

# 6. Modeli ve Tokenizer Sözlüğünü Saklama (Production Standard)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'vocab_size': tokenizer.vocab_size,
    'd_model': d_model,
    'word2id': tokenizer.word2id,
    'id2word': tokenizer.id2word
}
torch.save(checkpoint, "qillm_checkpoint.pt")
print("\n✅ Eğitilmiş Model ve Ağırlıklar 'qillm_checkpoint.pt' Olarak Kaydedildi!")