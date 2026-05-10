import torch

def calculate_sdr_torch(ref, est, eps=1e-10):
    """
    Calculate SDR between reference and estimation using torch.
    Supports batch processing.
    
    Args:
        ref (torch.Tensor): reference signal, shape (batch, ...)
        est (torch.Tensor): estimated signal, shape (batch, ...)
    """
    # Flatten everything except batch dimension
    ref = ref.view(ref.size(0), -1)
    est = est.view(est.size(0), -1)
    
    noise = est - ref
    
    numerator = torch.mean(ref ** 2, dim=-1).clamp(min=eps)
    denominator = torch.mean(noise ** 2, dim=-1).clamp(min=eps)
    
    sdr = 10.0 * torch.log10(numerator / denominator)
    return sdr

def calculate_sisdr_torch(ref, est, eps=1e-10):
    """
    Calculate SI-SDR between reference and estimation using torch.
    Supports batch processing.
    
    Args:
        ref (torch.Tensor): reference signal, shape (batch, ...)
        est (torch.Tensor): estimated signal, shape (batch, ...)
    """
    # Flatten everything except batch dimension
    ref = ref.view(ref.size(0), -1)
    est = est.view(est.size(0), -1)
    
    # Dot products across the sample dimension
    # (batch,)
    dot_target_est = torch.sum(ref * est, dim=-1)
    dot_target_target = torch.sum(ref * ref, dim=-1)
    
    # Scaling factor
    # (batch,)
    a = (dot_target_est + eps) / (dot_target_target + eps)
    
    e_true = a.unsqueeze(-1) * ref
    e_res = est - e_true
    
    # (batch,)
    sss = torch.sum(e_true ** 2, dim=-1)
    snn = torch.sum(e_res ** 2, dim=-1)
    
    sisdr = 10.0 * torch.log10((sss + eps) / (snn + eps))
    return sisdr
