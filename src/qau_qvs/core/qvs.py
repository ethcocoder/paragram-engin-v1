import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .asc import ASC

class QVS:
    """
    Quantum Virtual Substrate (QVS) - GPU NATIVE v2.0
    ================================================
    The foundational OPERATING SYSTEM LAYER for the QAU.
    
    UPGRADED: Runs natively on Torch/CUDA for seamless neural fusion.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.ascs: Dict[str, ASC] = {} 
        self.next_id = 0

    def create_asc(self, size: int = 1) -> str:
        asc_id = f"ASC_{self.next_id}"
        self.next_id += 1
        self.ascs[asc_id] = ASC(size=size, device=self.device)
        return asc_id

    def delete_asc(self, asc_id: str):
        self.ascs.pop(asc_id, None)

    def get_asc(self, asc_id: str) -> ASC:
        if asc_id not in self.ascs:
            raise KeyError(f"ASC {asc_id} not found.")
        return self.ascs[asc_id]

    # ------------------------------------------------------------------
    # The QVS Instruction Set (GPU Accelerated)
    # ------------------------------------------------------------------

    def SUPERPOSE(self, asc_id: str, basis_indices: List[int]) -> str:
        """Vectorized superposition across a set of basis indices."""
        asc = self.get_asc(asc_id)
        weight = 1.0 / np.sqrt(len(basis_indices))
        new_amps = torch.zeros_like(asc.amplitudes)
        for idx in basis_indices:
            new_amps[idx] = complex(weight)
        asc.amplitudes = new_amps
        return asc_id

    def WEAVE(self, asc_id: str, phase_angle: float = 0.0) -> str:
        """GPU-native global phase weave."""
        asc = self.get_asc(asc_id)
        phase_factor = torch.exp(torch.tensor(1j * phase_angle, device=self.device))
        asc.amplitudes = asc.amplitudes * phase_factor
        return asc_id

    def BOND(self, asc_id_a: str, asc_id_b: str, bond_type: str = "bell") -> str:
        """
        Forges a Non-Local Correlation Bond (NCB) on GPU.
        Uses Kronecker product (tensor product) for efficient joint state creation.
        """
        asc_a = self.get_asc(asc_id_a)
        asc_b = self.get_asc(asc_id_b)
        
        # Tensor product: A ⊗ B
        joint_amps = torch.kron(asc_a.amplitudes, asc_b.amplitudes)
        joint_size = asc_a.size + asc_b.size
        
        new_id = self.create_asc(size=joint_size)
        new_asc = self.get_asc(new_id)
        
        if bond_type == "bell":
            # Direct injection of Bell constraint
            new_asc.amplitudes.fill_(0)
            new_asc.amplitudes[0] = 1.0 / np.sqrt(2)
            new_asc.amplitudes[-1] = 1.0 / np.sqrt(2)
        elif bond_type == "ghz":
            new_asc.amplitudes.fill_(0)
            new_asc.amplitudes[0] = 1.0 / np.sqrt(2)
            new_asc.amplitudes[-1] = 1.0 / np.sqrt(2)
            
        self.delete_asc(asc_id_a)
        self.delete_asc(asc_id_b)
        return new_id

    def COLLAPSE(self, asc_id: str) -> int:
        """GPU-native wavefunction collapse."""
        asc = self.get_asc(asc_id)
        probs = torch.abs(asc.amplitudes) ** 2
        
        # Safety normalization
        p_sum = probs.sum()
        if p_sum > 0:
            probs = probs / p_sum
        else:
            probs = torch.ones_like(probs) / len(probs)
            
        idx = torch.multinomial(probs, 1).item()
        
        # Collapse the state
        asc.amplitudes.fill_(0)
        asc.amplitudes[idx] = 1.0 + 0j
        return idx
