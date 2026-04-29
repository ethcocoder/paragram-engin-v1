import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List, Union

class ASC:
    """
    Amplitude Superposition Cell (ASC) — GPU NATIVE v2.0
    ===================================================
    The primitive of COHERENT MULTIPLICITY.

    UPGRADED: Uses Torch Tensors for massive parallelization.
    Stores the state vector as a complex tensor (B, 2^size).
    """

    def __init__(self, size: int, device: str = 'cpu', amplitudes: Optional[torch.Tensor] = None):
        self.size = size
        self.device = device
        self.dim = 2 ** size
        
        if amplitudes is not None:
            self.amplitudes = amplitudes.to(device).to(torch.complex64)
        else:
            # Default to |0...0> ground state
            self.amplitudes = torch.zeros(self.dim, dtype=torch.complex64, device=device)
            self.amplitudes[0] = 1.0 + 0j

    def normalize(self) -> "ASC":
        norm = torch.norm(self.amplitudes)
        if norm > 1e-12:
            self.amplitudes = self.amplitudes / norm
        return self

    def prune(self, threshold: float = 1e-12) -> "ASC":
        mask = torch.abs(self.amplitudes) > threshold
        self.amplitudes = self.amplitudes * mask.to(torch.complex64)
        return self

    def get_state_vector(self) -> torch.Tensor:
        return self.amplitudes

    def fidelity(self, other: "ASC") -> float:
        inner_product = torch.dot(torch.conj(self.amplitudes), other.amplitudes)
        return torch.abs(inner_product).item() ** 2

    def entropy(self) -> float:
        probs = torch.abs(self.amplitudes) ** 2
        probs = probs[probs > 1e-15]
        return -torch.sum(probs * torch.log2(probs)).item()

    def clone(self) -> "ASC":
        return ASC(self.size, self.device, self.amplitudes.clone())

    def __repr__(self) -> str:
        return f"ASC(size={self.size}, device={self.device}, dim={self.dim})"
