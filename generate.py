import torch
from qillm.model import QILLMLanguageModel

print("--- 🔮 KUANTUM ESİNLEMELİ LLM METİN ÜRETİMİ (INFERENCE) ---")

# 1. Konfigürasyon (Eğitimdeki boyutlarla aynı olmalı)
vocab_size = 50
d_model = 16

# 2. Modeli Oluşturalım
model = QILLMLanguageModel(vocab_size=vocab_size, d_model=d_model)

# Not: Gerçek projelerde eğitilmiş ağırlıklar (.pt dosyası) yüklenir. 
# Şimdilik modelin otoregresif döngüde C++ motorunu nasıl tetiklediğini simüle ediyoruz.
model.eval()

# 3. Başlangıç Kelimesi (Seed Token)
start_token = 1
generate_length = 8  # Üretilecek toplam kelime sayısı

# Başlangıç dizisi [1]
generated_sequence = [start_token]

print(f"\n🌱 Başlangıç Kelime ID'si (Prompt): [{start_token}]")
print("🔄 C++ Kuantum Motoru Otoregresif Döngüde Tetikleniyor...\n")

# 4. Metin Üretim Döngüsü
with torch.no_grad(): # Çıkarım yaparken gradyan hesaplamaya gerek yoktur (RAM tasarrufu)
    for step in range(generate_length):
        # Mevcut diziyi tensör formatına getir
        input_ids = torch.tensor([generated_sequence])
        
        # İleri Besleme (C++ SVD Sıkıştırması Tetiklenir)
        logits = model(input_ids)
        
        # En son üretilen kelimenin tahmin olasılıklarını al
        next_token_logits = logits[0, -1, :]
        
        # En yüksek olasılıklı kelimeyi seç (Greedy Search)
        next_token = torch.argmax(next_token_logits).item()
        
        # Yeni tahmin edilen kelimeyi dizinin sonuna ekle
        generated_sequence.append(next_token)
        
        print(f"Adım {step+1:02d}: Tahmin Edisi [{next_token}] ---> Güncel Cümle Dizisi: {generated_sequence}")

print("\n✨ Üretilen Tam Kelime Dizisi:")
print(generated_sequence)