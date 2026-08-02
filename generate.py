import torch
from qillm.model import QILLMLanguageModel
from qillm.tokenizer import SimpleTokenizer

print("--- 🔮 PROFESYONEL METİN ÜRETİMİ (INFERENCE FROM CHECKPOINT) ---")

# 1. Kaydedilmiş Modeli ve Sözlüğü Yükle
checkpoint = torch.load("qillm_checkpoint.pt")

tokenizer = SimpleTokenizer()
tokenizer.word2id = checkpoint['word2id']
tokenizer.id2word = checkpoint['id2word']

model = QILLMLanguageModel(
    vocab_size=checkpoint['vocab_size'], 
    d_model=checkpoint['d_model'],
    compression_rate=0.5
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Başlangıç Kelimesi (Prompt)
prompt = "kuantum"
input_tokens = tokenizer.encode(prompt)
generated_tokens = list(input_tokens)

print(f"Girdi (Prompt): '{prompt}'")
print("🔄 C++ SVD Motoru ile Metin Tamamlanıyor...\n")

# 3. Otoregresif Metin Üretimi
generate_words = 6

with torch.no_grad():
    for _ in range(generate_words):
        input_tensor = torch.tensor([generated_tokens])
        logits = model(input_tensor)
        
        next_token_id = torch.argmax(logits[0, -1, :]).item()
        generated_tokens.append(next_token_id)

# 4. Sonucu Metne Çevirip Ekrana Bas
final_text = tokenizer.decode(generated_tokens)
print(f"✨ Üretilen Cümle: \"{final_text}\"")