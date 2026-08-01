import torch
from qillm.model import QILLMLanguageModel

print("--- 🚀 TAM KUANTUM ESİNLEMELİ LLM MİMARİSİ TESTİ ---")

# Sözlük Boyutu (Vocab Size): 100 kelimelik küçük bir kelime dağarcığı
# Vektör Boyutu (d_model): 16
vocab_size = 100
d_model = 16

# 1. Modeli Örnekleyelim
model = QILLMLanguageModel(vocab_size=vocab_size, d_model=d_model)

# 2. Temsili bir cümle girdisi oluşturalım (Örn: [12, 45, 89, 3] -> 4 kelimelik cümle ID'leri)
temsili_cumle = torch.tensor([[12, 45, 89, 3]])

print(f"\n1. Girdi Cümle Token ID'leri: {temsili_cumle}")

# 3. Modeli Çalıştıralım
logits = model(temsili_cumle)

print("\n2. Model Çıktısı (Logits) Başarıyla Üretildi!")
print(f"Çıktı Matrisi Boyutu: {logits.shape} -> [Batch, Cümle Uzunluğu, Sözlük Boyutu]")

# 4. En Yüksek Olasılıklı Kelime Tahminini Seçelim
tahmin_id = torch.argmax(logits[0, -1, :]).item()
print(f"\n3. Modelin C++ Kuantum Motoruyla Tahmin Ettiği Bir Sonraki Kelime ID'si: {tahmin_id}")