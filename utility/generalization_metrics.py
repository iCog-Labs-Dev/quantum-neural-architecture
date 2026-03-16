# utility/generalization_metrics.py
import torch
import math


'''Formulas:
    1. Empirical Fisher: F = (1/N) * sum_i (g_i @ g_i.T)
       - g_i: flattened gradient vector for the i-th data point.
       - N: number of data points (total samples accumulated).
    
    2. Trace-normalized Fisher: F_hat = (d / Tr(F)) * F if Tr(F) > 0, else F.
       - Ensures Tr(F_hat) = d.
       - Adds epsilon for stability.
    
    3. Effective Dimension: d_eff(n) = sum_i (n * λ_i / (1 + n * λ_i))
       - λ_i: eigenvalues of F_hat, clamped >= 1e-12.
       - n = N (number of data points).
    
    4. PAC-style Generalization Bound: sqrt( (d_eff * log(n) + 2 * log(1/δ)) / (2n) )
       - δ: confidence parameter.
       - Handles edge cases like n <= 1 or d_eff <= 0 by returning 0.

    5. Spectral Entropy :
       p_i = λ_i / sum(λ_j)     # normalize eigenvalues to a probability distribution
       H = - sum_i (p_i * log(p_i))     # Shannon entropy (in nats)
       H_norm = H / log(d)              # normalized to [0, 1]
       → Low entropy → spiked spectrum (few large eigenvalues dominate)
       → High entropy → more uniform spectrum
'''


class FisherGeneralizationMetric:
    def __init__(self, delta: float = 0.1, eps: float = 1e-12):
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1.")
        self.delta = delta
        self.eps = eps
        self.reset()

    def reset(self):
        self.S = None
        self.d = None
        self.total_samples = 0

    def accumulate(self, grad_vector: torch.Tensor):
        g = grad_vector.detach().to(torch.float64)
        if self.d is None:
            self.d = g.numel()
            self.S = torch.zeros(
                self.d,
                self.d,
                dtype=torch.float64,
                device=g.device,
            )

        self.S += torch.outer(g, g)
        self.total_samples += 1

    def compute(self):
        N = self.total_samples

        if N <= 1:
            return {
                "effective_dimension": 0.0,
                "generalization_bound": 0.0,
            }

        # 1) Empirical Fisher
        F = self.S / N

        # 2) Trace normalization
        trace_F = torch.trace(F)

        if trace_F > self.eps:
            F_hat = (self.d / (trace_F + self.eps)) * F
        else:
            F_hat = F.clone()

        # Eigenvalues
        eigenvalues = torch.linalg.eigvalsh(F_hat)
        eigenvalues = torch.clamp(eigenvalues, min=self.eps)

        # 3) Effective dimension
        n = N
        d_eff = torch.sum(
            (n * eigenvalues) / (1.0 + n * eigenvalues)
        ).item()

        # 4) PAC-style bound
        if d_eff <= 0:
            gen_gap = 0.0
        else:
            numerator = (
                d_eff * math.log(n)
                + 2.0 * math.log(1.0 / self.delta)
            )
            gen_gap = math.sqrt(numerator / (2.0 * n))

        # 5) Spectral Entropy 
        total = eigenvalues.sum()
        if total > self.eps:
            probs = eigenvalues / total
            # Avoid log(0) issues
            probs = torch.clamp(probs, min=self.eps)
            probs = probs / probs.sum()  
            
            log_probs = torch.log(probs)
            entropy = -torch.sum(probs * log_probs).item()
            
            entropy_normalized = entropy / math.log(self.d) if self.d > 1 else 0.0
        else:
            entropy = 0.0
            entropy_normalized = 0.0

        return {
            "effective_dimension": d_eff,
            "generalization_bound": gen_gap,
            "spectral_entropy": entropy,              # raw entropy (nats)
            "spectral_entropy_normalized": entropy_normalized,  # 0 to 1
        }