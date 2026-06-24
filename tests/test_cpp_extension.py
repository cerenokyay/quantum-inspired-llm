import torch
import qillm_cpp  # Bizim az önce derlediğimiz C++ motoru!

print("--- 🧠 KUANTUM ESİNLEMELİ LLM MOTORU TESTİ ---")

# 1. Adım: Yapay zekada verileri temsil eden rastgele bir matris üretiyoruz (3 satır, 3 sütun)
klasik_veri = torch.randn(3, 3)
print("\n1. Python Tarafında Oluşturulan İlk Veri (Matris):")
print(klasik_veri)

# 2. Adım: Bu matrisi Python'dan alıp, C++ dünyasına fırlatıyoruz
print("\n2. Veri C++ Motoruna Gönderiliyor...")
cpp_sonuc = qillm_cpp.forward_attention_gate(klasik_veri)

# 3. Adım: C++'tan dönen sonucu ekrana basıp doğruluğunu kontrol ediyoruz
print("\n3. C++ Motorundan Python'a Geri Dönen Sonuç:")
print(cpp_sonuc)