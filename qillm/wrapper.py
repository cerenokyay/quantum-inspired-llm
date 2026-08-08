import torch
import torch.nn as nn
from qillm.layers import QILLMSVDLinear
import qillm_cpp  # setup.py'de belirlediğin C++ modül adı (örn: qillm_ext)

def compress_model(
    model: nn.Module, 
    compression_ratio: float = 0.5, 
    target_layers: list = ["q_proj", "k_proj", "v_proj", "out_proj", "c_attn", "c_proj"]
) -> nn.Module:
    
    compressed_count = 0
    for name, module in list(model.named_modules()):
        if any(target in name for target in target_layers) and isinstance(module, nn.Linear):
            weight = module.weight.data.float()
            bias = module.bias.data if module.bias is not None else None
            
            # C++ fonksiyonunun adını kendi csrc/tensor_network.cpp dosyadakine göre uyarla
            U, S, V_T = qillm_cpp.svd_compress(weight, compression_ratio)
            
            S_sqrt = torch.sqrt(S).unsqueeze(0)
            U_eff = U * S_sqrt
            VT_eff = V_T * S_sqrt.t()
            
            new_layer = QILLMSVDLinear(U_eff, VT_eff, bias)
            
            parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent_module = dict(model.named_modules())[parent_name] if parent_name else model
            setattr(parent_module, attr_name, new_layer)
            
            compressed_count += 1
            print(f"⚡ [QILLM C++] Katman Sıkıştırıldı: {name} | Oran: {compression_ratio}")
            
    print(f"\n✅ Toplam {compressed_count} katman sıkıştırıldı!")
    return model