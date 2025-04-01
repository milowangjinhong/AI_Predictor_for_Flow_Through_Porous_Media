import torch
import torch.nn as nn

def mape_loss(outputs, targets):
    epsilon = 1e-8  # To avoid division by zero
    return torch.mean(torch.abs((targets - outputs) / (targets + epsilon)))

def loss_reduction(tensor, reduction):
    
    if reduction == 'sum':
        out = torch.sum(tensor)
    elif reduction == 'mean' or reduction == 'avg':
        out = torch.mean(tensor)
    elif reduction == 'max':
        out = torch.max(tensor)
    else:
        raise TypeError("Unrecognised reduction type %s"%(reduction))
    
    return out

# example customised loss
class CombinedLoss(nn.Module):
    
    def __init__(self, reduction = 'sum', loss_param_dict = {'alpha': 0.5}):
        
        super(CombinedLoss, self).__init__()
        
        self.alpha      = loss_param_dict['alpha']  # Balance factor for the two losses
        self.reduction  = reduction
        self.mse_loss   = nn.MSELoss(reduction = reduction)

    def forward(self, outputs, targets):
        e1 = self.mse_loss(outputs, targets)
        
        e2 = torch.mean(torch.abs((targets - outputs) / (targets + 1e-8)))
        e2 = loss_reduction(e2, self.reduction)
            
        return self.alpha * e1 + (1 - self.alpha) * e2