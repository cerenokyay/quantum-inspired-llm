import torch
import torch.nn as nn

class QILLMSVDLinear(nn.Module):
    """
    Orijinal ağırlık matrisi yerine SVD ile ayrıştırılmış 
    U ve V_T matrislerini tutan özel Kuantum-İlhamlı katman.
    """
    def __init__(self, U: torch.Tensor, V_T: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.V_T = nn.Parameter(V_T, requires_grad=False)
        
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.V_T.t())
        out = torch.matmul(out, self.U.t())
        if self.bias is not None:
            out += self.bias
        return out