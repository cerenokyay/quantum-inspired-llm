import torch
import time
import math

class QILLMEvaluator:
    @staticmethod
    def get_model_size_mb(model: torch.nn.Module) -> float:
        """Modelin bellekte (RAM/VRAM) kapladığı gerçek alanı MB cinsinden hesaplar."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return (param_size + buffer_size) / (1024 ** 2)

    @staticmethod
    def measure_speed(model: torch.nn.Module, tokenizer, prompt: str, max_new_tokens: int = 30) -> float:
        """Modelin 1 saniyede kaç token (kelime parçası) üretebildiğini ölçer."""
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Donanım Isınma Turu (Warm-up)
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1)
            
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        time_taken = end_time - start_time
        
        return generated_tokens / time_taken

    @staticmethod
    def calculate_perplexity(model: torch.nn.Module, tokenizer, text: str) -> float:
        """
        Modelin dili anlama kalitesini ölçer (PPL). 
        Düşük PPL daha iyidir. Sıkıştırma sonrası bu değerin uçmaması gerekir.
        """
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        return math.exp(loss.item())