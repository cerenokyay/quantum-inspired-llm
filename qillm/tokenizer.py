from typing import List, Dict

class SimpleTokenizer:
    """
    Kullanıcı dostu, kelime tabanlı basit bir Türkçe Tokenizer.
    Metinleri tam sayı dizilerine (ID) ve ID'leri tekrar metne çevirir.
    """
    def __init__(self):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        self.word2id: Dict[str, int] = {self.pad_token: 0, self.unk_token: 1}
        self.id2word: Dict[int, str] = {0: self.pad_token, 1: self.unk_token}
        
    def fit_on_text(self, text: str):
        """Metindeki tüm benzersiz kelimeleri tarar ve sözlüğü oluşturur."""
        words = text.lower().split()
        for word in words:
            if word not in self.word2id:
                new_id = len(self.word2id)
                self.word2id[word] = new_id
                self.id2word[new_id] = word
                
    def encode(self, text: str) -> List[int]:
        """Metni sayı dizisine (Token ID'leri) çevirir."""
        words = text.lower().split()
        return [self.word2id.get(w, self.word2id[self.unk_token]) for w in words]

    def decode(self, ids: List[int]) -> str:
        """Sayı dizisini tekrar anlaşılır Türkçe metne çevirir."""
        return " ".join([self.id2word.get(i, self.unk_token) for i in ids])

    @property
    def vocab_size(self) -> int:
        return len(self.word2id)