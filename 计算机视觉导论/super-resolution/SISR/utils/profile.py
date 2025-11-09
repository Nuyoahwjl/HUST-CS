import torch

def count_params_m(model) -> float:
    return sum(p.numel() for p in model.parameters())

def profile_flops_g(model, inp_size):
    from thop import profile
    import warnings
    warnings.filterwarnings("ignore", message="This API is being deprecated", module="thop", category=UserWarning)
    device = next(model.parameters()).device
    dummy = torch.zeros(*inp_size, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return float(flops) / 1e9
