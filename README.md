# FAMOptimizer
Frequency-Adaptive Momentum optimizer with parameter-specific handlers. + FAMScheduler.

Beta . Working. I'm surprised too.
Proof of concept. 

An experimental approach specifically designed for speech recognition tasks, FAM adapts momentum based on the frequency characteristics of gradient updates.

### Frequency-Adaptive Momentum (FAM)

#### Core Concept

- Speech signals possess an inherent frequency structure, with different parts of the model responding to various frequency bands. This frequency structure remains preserved, albeit transformed, when converted to log-mel spectrograms, with model parameters adapting to capture this structure.
- The Chain of Frequency Information: Original Audio → Log-Mel Spectrogram → Encoder Parameters → Gradient Updates.
- Empirical observations reveal that transformer-based speech models develop:
  - Lower encoder layers with filters responsive to specific frequency bands in the mel spectrogram.
  - Attention heads tracking particular acoustic patterns over time.
  - A hierarchical representation from acoustic features to phonetic units to words.
- FAM aims to integrate a momentum scheme that adapts based on the "frequency signature" of gradient updates.

#### Why This Optimizer Makes Sense

FAM acknowledges the frequency structure within the optimization process itself, recognizing that:
- **Gradient Frequencies Matter:** The Fourier transform of gradient updates reveals patterns linked to the model's current learning phase.
- **Different Parameters Process Different Bands:** Similar to how our ears have frequency-specific receptors, different parts of the model specialize in various acoustic frequencies.
- **Temporal Structure in Learning:** Speech learning progresses through stages - from basic acoustics to phonetic patterns to linguistic structures.

By applying distinct momentum factors to different frequency bands in parameter space, FAM provides the optimizer with domain-specific audio information that it otherwise wouldn't have.




Usage example
```
param_groups = get_parameter_groups(model=model, lr=0.001, weight_decay=1e-6)

optimizer = FAMOptimizer(
    params=param_groups,
    beta=0.99,
    n_bands=10,
    fam_start_step=100,
    layer_boost=True,
    min_size=128,
    debug=True,
    weight_decay=0.0025,
    lr=0.001,
)

scheduler = FAMScheduler2(
    optimizer=optimizer,
    warmup_steps=100,
    total_steps=10000,
    decay_start_step=100
)


```

```python


import torch
import torch.nn as nn
import numpy as np
import json
import os
from datetime import datetime

class FrequencyHandler:
    """Base class for parameter-specific frequency analysis functions"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        """Default frequency analysis implementation"""
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        band_size = freq_power.shape[0] // n_bands
        if band_size <= 0:
            return [0.0] * n_bands
            
        band_powers = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i+1) * band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Default adaptive momentum calculation"""
        n_bands = len(band_values)
        high_freq_activity = sum(band_values[n_bands//2:])
        
        if high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha

class ConvFrequencyHandler(FrequencyHandler):
    """Specialized handler for convolutional layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        band_powers = []
        total_freqs = freq_power.shape[0]
        
        for i in range(n_bands):
            start_idx = int((total_freqs ** (i/n_bands)) - 1)
            end_idx = int((total_freqs ** ((i+1)/n_bands)) - 1)
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, total_freqs)
            
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Convolutional layers benefit from more smoothing in mid-frequencies"""
        n_bands = len(band_values)
        mid_freq_activity = sum(band_values[n_bands//4:(3*n_bands)//4])
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        if mid_freq_activity > 0.4:
            return min(0.97, base_alpha + 0.07)
        elif high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha

class AttentionFrequencyHandler(FrequencyHandler):
    """Specialized handler for attention layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        band_powers = []
        half_bands = n_bands // 2
        low_band_size = (freq_power.shape[0] // 2) // half_bands
        for i in range(half_bands):
            start_idx = i * low_band_size
            end_idx = min((i+1) * low_band_size, freq_power.shape[0] // 2)
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
        high_band_size = (freq_power.shape[0] - (freq_power.shape[0] // 2)) // (n_bands - half_bands)
        for i in range(half_bands, n_bands):
            start_idx = (freq_power.shape[0] // 2) + (i - half_bands) * high_band_size
            end_idx = min((freq_power.shape[0] // 2) + (i - half_bands + 1) * high_band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Custom adaptive momentum for attention layers"""
        n_bands = len(band_values)
        max_band_idx = np.argmax(band_values)
        if max_band_idx < n_bands // 4:
            return max(0.85, base_alpha - 0.05)
        elif max_band_idx > 3*n_bands // 4:
            return min(0.98, base_alpha + 0.08)
        return base_alpha

class EmbeddingFrequencyHandler(FrequencyHandler):
    """Specialized handler for embedding layers"""
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Embeddings often benefit from very stable updates"""
        n_bands = len(band_values)
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        if high_freq_activity > 0.2:
            return min(0.98, base_alpha + 0.08)
        return base_alpha

class FAMOptimizer(torch.optim.Optimizer):
    """
    Frequency-Adaptive Momentum optimizer with parameter-specific handlers.
    
    Args:
        ... (existing parameters)
        debug (bool, optional): Whether to collect debug information (default: False)
        debug_dir (str, optional): Directory to save debug info (default: './fam_debug')
        debug_interval (int, optional): Steps between debug dumps (default: 1000)
    """
    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.99, eps=1e-8,
                 weight_decay=0.0, n_bands=8, fam_start_step=100,
                 layer_boost=True, min_size=256, debug=False,
                 debug_dir='./fam_debug', debug_interval=1000):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps,
                       weight_decay=weight_decay, n_bands=n_bands,
                       fam_start_step=fam_start_step, 
                       layer_boost=layer_boost, min_size=min_size)
        self.debug = debug
        self.debug_info = {} if debug else None
        self.debug_dir = debug_dir
        self.debug_interval = debug_interval
        self.last_dump_step = 0
        
        if debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            self.debug_file = os.path.join(
                debug_dir, 
                f"fam_debug_{datetime.now().strftime('%m%d_%H%M%S')}.json"
            )
            with open(self.debug_file, 'w') as f:
                json.dump({
                    "optimizer": "FAMOptimizer",
                    "settings": {
                        "lr": lr,
                        "alpha": alpha,
                        "beta": beta,
                        "n_bands": n_bands,
                        "fam_start_step": fam_start_step,
                    },
                    "parameters": {},
                    "steps_recorded": []
                }, f, indent=2)
        self.handlers = {
            "default": FrequencyHandler(),
            "conv": ConvFrequencyHandler(),
            "attention": AttentionFrequencyHandler(),
            "embedding": EmbeddingFrequencyHandler()
        }
        param_groups = self._add_handlers_to_groups(params)
        super(FAMOptimizer, self).__init__(params=param_groups, defaults=defaults)
    def _add_handlers_to_groups(self, params):
        """Add appropriate handlers to parameter groups based on type"""
        if isinstance(params, list) and all(isinstance(pg, dict) for pg in params):
            for pg in params:
                if 'handler' not in pg:
                    if any('conv' in name.lower() for name in pg.get('names', [])):
                        pg['handler'] = 'conv'
                    elif any(name in name.lower() for name in pg.get('names', []) 
                             for name in ['attention', 'mha', 'self_attn']):
                        pg['handler'] = 'attention'
                    elif any(name in name.lower() for name in pg.get('names', [])
                             for name in ['embed', 'token']):
                        pg['handler'] = 'embedding'
                    else:
                        pg['handler'] = 'default'
            return params
        else:
            return [{'params': params, 'handler': 'default'}]
    
    def get_handler(self, group):
        """Get the appropriate frequency handler for the parameter group"""
        handler_name = group.get('handler', 'default')
        return self.handlers[handler_name]
    
    def dump_debug_info(self, force=False):
        """Save the current debug information to file"""
        if not self.debug or not hasattr(self, 'debug_file'):
            return
        current_step = max([self.state[p]['step'] for p in self.state], default=0)
        if force or (current_step - self.last_dump_step >= self.debug_interval):
            try:
                with open(self.debug_file, 'r') as f:
                    debug_data = json.load(f)
                debug_data["steps_recorded"].append(current_step)
                
                for param_name, param_info in self.debug_info.items():
                    if param_name not in debug_data["parameters"]:
                        debug_data["parameters"][param_name] = {
                            "handler": param_info.get('handler', 'default'),
                            "steps": [],
                            "bands": [],
                            "alpha": []
                        }
                    last_recorded = len(debug_data["parameters"][param_name]["steps"])
                    if last_recorded < len(param_info['steps']):
                        debug_data["parameters"][param_name]["steps"].extend(param_info['steps'][last_recorded:])
                        debug_data["parameters"][param_name]["bands"].extend(param_info['bands'][last_recorded:])
                        debug_data["parameters"][param_name]["alpha"].extend(param_info['alpha'][last_recorded:])
                with open(self.debug_file, 'w') as f:
                    json.dump(debug_data, f)
                
                self.last_dump_step = current_step
                for param_info in self.debug_info.values():
                    param_info['steps'] = param_info['steps'][-10:]
                    param_info['bands'] = param_info['bands'][-10:]
                    param_info['alpha'] = param_info['alpha'][-10:]
                    
            except Exception as e:
                print(f"Error dumping FAM debug info: {e}")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FAMOptimizer does not support sparse gradients')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['freq_history'] = {}
                    state['param_name'] = f"param_{p_idx}"
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                beta = group['beta']
                lr = group['lr']
                n_bands = group['n_bands']
                handler = self.get_handler(group)
                
                should_apply_fam = (
                    state['step'] > group['fam_start_step'] and
                    p.numel() > group['min_size']
                )
                
                if should_apply_fam:
                    try:
                        if p.numel() > 10000:
                            if p.dim() > 1:
                                row_indices = torch.randperm(p.size(0))[:min(p.size(0), 64)]
                                col_indices = torch.randperm(p.size(1))[:min(p.size(1), 64)]
                                grad_sample = grad[row_indices][:, col_indices].flatten()
                            else:
                                sample_idx = torch.randperm(p.numel())[:1000]
                                grad_sample = grad.flatten()[sample_idx]
                        else:
                            grad_sample = grad.flatten()
                        band_powers = handler.analyze(grad_sample, n_bands, group['eps'])
                        if state['step'] <= 10 and p_idx == 0:
                            print(f"Step {state['step']}: Found {len(band_powers)} frequency bands")
                            print(f"Band powers: {[f'{v:.4f}' for v in band_powers]}")
                        for i, power in enumerate(band_powers):
                            band_key = f'band_{i}'
                            if band_key not in state['freq_history']:
                                state['freq_history'][band_key] = power
                            else:
                                state['freq_history'][band_key] = (
                                    beta * state['freq_history'][band_key] +
                                    (1-beta) * power
                                )
                        band_values = [state['freq_history'].get(f'band_{i}', 0) 
                                      for i in range(n_bands)]
                        effective_alpha = handler.get_adaptive_momentum(band_values, alpha)
                        
                        if self.debug:
                            param_name = state['param_name']
                            if param_name not in self.debug_info:
                                self.debug_info[param_name] = {
                                    'steps': [], 
                                    'bands': [], 
                                    'handler': group.get('handler', 'default'),
                                    'alpha': []
                                }
                            
                            if state['step'] % 10 == 0:
                                self.debug_info[param_name]['steps'].append(state['step'])
                                self.debug_info[param_name]['bands'].append(band_values)
                                self.debug_info[param_name]['alpha'].append(effective_alpha)
                        exp_avg.mul_(effective_alpha).add_(grad, alpha=1-effective_alpha)
                    except Exception as e:
                        import traceback
                        print(f"Error in FAM processing for parameter {p_idx}:")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {e}")
                        print(f"Parameter shape: {p.shape}, numel: {p.numel()}")
                        print(traceback.format_exc())
                        exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                else:
                    exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                p.add_(exp_avg, alpha=-lr)
        
        if self.debug:
            self.dump_debug_info()
        
        return loss
    
    def __del__(self):
        """Clean up and final debug dump when optimizer is destroyed"""
        if self.debug:
            self.dump_debug_info(force=True)

def get_parameter_groups(model, lr=1e-3, weight_decay=0.0):
    """
    Create parameter groups for FAMOptimizer with appropriate handlers based on layer type
    """
    param_groups = []
    conv_params = []
    conv_names = []
    
    attn_params = []
    attn_names = []
    
    embed_params = []
    embed_names = []
    
    norm_params = []
    norm_names = []
    
    other_params = []
    other_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if any(x in name.lower() for x in ['conv', 'cnn']):
            conv_params.append(param)
            conv_names.append(name)
        elif any(x in name.lower() for x in ['attention', 'mha', 'self_attn']):
            attn_params.append(param)
            attn_names.append(name)
        elif any(x in name.lower() for x in ['embed', 'token']):
            embed_params.append(param)
            embed_names.append(name)
        elif any(x in name.lower() for x in ['norm', 'batch', 'layer']):
            norm_params.append(param)
            norm_names.append(name)
        else:
            other_params.append(param)
            other_names.append(name)
    if conv_params:
        param_groups.append({
            'params': conv_params,
            'names': conv_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.9,
            'handler': 'conv',
            'n_bands': 10
        })
    
    if attn_params:
        param_groups.append({
            'params': attn_params,
            'names': attn_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.92,
            'handler': 'attention',
            'n_bands': 12
        })
    
    if embed_params:
        param_groups.append({
            'params': embed_params,
            'names': embed_names,
            'lr': lr * 0.8,
            'weight_decay': weight_decay * 1.5,
            'alpha': 0.95,
            'handler': 'embedding',
            'n_bands': 8
        })
    
    if norm_params:
        param_groups.append({
            'params': norm_params,
            'names': norm_names,
            'lr': lr,
            'weight_decay': 0.0,
            'alpha': 0.9,
            'handler': 'default',
            'n_bands': 4
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'names': other_names,
            'lr': lr,
            'weight_decay': weight_decay,
            'alpha': 0.9,
            'handler': 'default',
            'n_bands': 8
        })
    
    return param_groups

import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class FAMSchedulerb(_LRScheduler):
    """
    Scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for the linear warmup
        max_epochs: Total number of epochs
        warmup_start_lr: Initial learning rate for warmup
        eta_min: Minimum learning rate after cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]
import torch
import math

class SimpleFAM(torch.optim.Optimizer):
    """
    Simplified Frequency-Adaptive Momentum optimizer
    
    A lightweight implementation that focuses on the core concepts
    without complex debugging or parameter-specific handlers.
    """
    def __init__(self, params, lr=0.001, alpha=0.9, beta=0.99):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super(SimpleFAM, self).__init__(params, defaults)
        print(f"SimpleFAM initialized with lr={lr}, alpha={alpha}")
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                if p.numel() > 1000 and state['step'] > 100:
                    grad_sample = p.grad.flatten()[:min(1000, p.numel())]
                    freq = torch.fft.rfft(grad_sample.float())
                    power = torch.abs(freq)
                    half = power.shape[0] // 2
                    high_ratio = power[half:].sum() / (power.sum() + 1e-8)
                    effective_alpha = min(0.98, alpha + 0.05 * high_ratio)
                    exp_avg.mul_(effective_alpha).add_(p.grad, alpha=1-effective_alpha)
                else:
                    exp_avg.mul_(alpha).add_(p.grad, alpha=1-alpha)
                p.add_(exp_avg, alpha=-group['lr'])
        
        return loss
    
class FAMScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Step-based learning rate scheduler for FAM optimizer
    with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps=1000, total_steps=100000, 
                 decay_start_step=None, warmup_start_lr=1e-6, eta_min=1e-6, 
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_step = decay_start_step if decay_start_step is not None else warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                    for base_lr in self.base_lrs]
        
        elif self.last_epoch < self.decay_start_step:
            return self.base_lrs
        
        else:
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.decay_start_step) / 
                                (self.total_steps - self.decay_start_step))) / 2 + 1e-8
                   for base_lr in self.base_lrs]

```
