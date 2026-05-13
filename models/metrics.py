import torch


def _finite_flatten(audio):
    audio = torch.nan_to_num(audio.float(), nan=0.0, posinf=0.0, neginf=0.0)
    return audio.reshape(audio.size(0), -1)


def calculate_sdr_torch(ref, est, eps=1e-10):
    """
    Calculate SDR between reference and estimation using torch.
    Supports batch processing.
    
    Args:
        ref (torch.Tensor): reference signal, shape (batch, ...)
        est (torch.Tensor): estimated signal, shape (batch, ...)
    """
    # Flatten everything except batch dimension.
    ref = _finite_flatten(ref)
    est = _finite_flatten(est)
    
    noise = est - ref
    
    numerator = torch.mean(ref ** 2, dim=-1).clamp(min=eps)
    denominator = torch.mean(noise ** 2, dim=-1).clamp(min=eps)
    
    sdr = 10.0 * torch.log10((numerator / denominator).clamp(min=eps))
    return torch.nan_to_num(sdr, nan=0.0, posinf=100.0, neginf=-100.0)

def calculate_sisdr_torch(ref, est, eps=1e-10):
    """
    Calculate SI-SDR between reference and estimation using torch.
    Supports batch processing.
    
    Args:
        ref (torch.Tensor): reference signal, shape (batch, ...)
        est (torch.Tensor): estimated signal, shape (batch, ...)
    """
    # Flatten everything except batch dimension.
    ref = _finite_flatten(ref)
    est = _finite_flatten(est)
    
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
    
    sisdr = 10.0 * torch.log10(((sss + eps) / (snn + eps)).clamp(min=eps))
    return torch.nan_to_num(sisdr, nan=0.0, posinf=100.0, neginf=-100.0)
