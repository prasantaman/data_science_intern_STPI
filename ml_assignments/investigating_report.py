"""
Report: Effect of n (dataset size) and σ (noise) on learning β in Linear Regression

Objective:
To study how dataset size (n) and noise level (σ) affect the model's ability 
to correctly learn the true coefficients β used to generate:
    y = Xβ + e
where e ~ N(0, σ)


Effect of Dataset Size (n):

1. Small n:
   - High variance in β estimation
   - Model may overfit noise
   - Learned β deviates from true β

2. Large n:
   - Better approximation of true β
   - Lower variance
   - Stable convergence

Key Insight:
    Var(β̂) ∝ σ² / n
Increasing n reduces estimation variance.


Effect of Noise (σ):

1. Small σ:
   - Accurate β recovery
   - Low final cost
   - Fast convergence

2. Large σ:
   - β estimation becomes unstable
   - Higher final cost
   - Harder to detect true linear relationship

Key Insight:
    Estimation error ∝ σ / √n

Conclusion:

- Increasing n improves coefficient accuracy.
- Increasing σ makes learning harder.
- Large n can reduce the negative impact of noise.
- Best performance occurs when:
Large n + Small σ
"""
