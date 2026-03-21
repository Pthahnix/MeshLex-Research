## Contents
- 1 Introduction
- 2 Related Work
  - 2.1 Coupled Codebook–Network Optimization
  - 2.2 Predefined Codebook Quantization
- 3 Methodology
  - 3.1 Setup and Notation
  - 3.2 VP-VAE: From Quantization to Perturbation
    - 3.2.1 Scale Alignment via Adaptive Radius Estimation
    - 3.2.2 Distribution Consistency via Metropolis–Hastings Sampling
      - Target density estimation.
      - Proposal distribution.
      - Acceptance probability.
      - Perturbation operator.
    - 3.2.3 Optimization Objective
    - 3.2.4 Codebook Generation
  - 3.3 Finite Scalar Perturbation (FSP)
    - Uniform latent variables via “CDF-like” activation.
    - Lloyd–Max centroids.
    - Simplified perturbation mechanism.
    - Training: perturb-or-quantize mixture.
- 4 Experiments
  - 4.1 Experimental Setup
    - 4.1.1 Datasets and Backbones
    - 4.1.2 Evaluation Metrics
  - 4.2 Results
- 5 Conclusion
- Impact Statement
- References
- Appendix A Experimental Details
  - A.1 Representative Baselines
  - A.2 Training and Optimization
- Appendix B Additional Experiments
  - B.1 Ablation Study
    - B.1.1 VP-VAE Ablation
      - Effect of latent normalization.
      - Effect of Metropolis–Hastings mechanism.
    - B.1.2 FSP Ablation
      - Choice of FSP activation.
  - B.2 Qualitative Analysis
    - Training dynamics of codebook utilization.
    - Comparison of FSP and FSQ quantization schemes.
- Appendix C Limitations and Future Work

## Abstract

Abstract Vector Quantized Variational Autoencoders (VQ-VAEs) are fundamental to modern generative modeling, yet they often suffer from training instability and “codebook collapse” due to the inherent coupling of representation learning and discrete codebook optimization.
In this paper, we propose VP-VAE (Vector Perturbation VAE), a novel paradigm that decouples representation learning from discretization by eliminating the need for an explicit codebook during training.
Our key insight is that, from the neural network’s viewpoint, performing quantization primarily manifests as injecting a structured perturbation in latent space.
Accordingly, VP-VAE replaces the non-differentiable quantizer with distribution-consistent and scale-adaptive latent perturbations generated via Metropolis–Hastings sampling.
This design enables stable training without a codebook while making the model robust to inference-time quantization error.
Moreover, under the assumption of approximately uniform latent variables, we derive FSP (Finite Scalar Perturbation), a lightweight variant of VP-VAE that provides a unified theoretical explanation and a practical improvement for FSQ-style fixed quantizers.
Extensive experiments on image and audio benchmarks demonstrate that VP-VAE and FSP improve reconstruction fidelity and achieve substantially more balanced token usage, while avoiding the instability inherent to coupled codebook training. * * * The example code is available at https://github.com/zhai-lw/vp-vae .

## 1 Introduction

Discrete tokenization of continuous signals is a core element in modern generative modeling.
By converting high-dimensional data (e.g., images or audio) into compact discrete sequences, one can directly leverage sequence models such as Transformers (Chen et al., 2025; Esser et al., 2021; Rombach et al., 2022) for downstream tasks, including image/audio generation and text-to-image/audio synthesis (Ramesh et al., 2021; Chang et al., 2022; Yu et al., 2024; Borsos et al., 2023; Brendel et al., 2024a).
Vector-Quantized VAEs (VQ-VAEs) (Van Den Oord et al., 2017) and their variants (Yu et al., 2021; Zeghidour et al., 2021; Chiu et al., 2022; Mentzer et al., 2023; Huh et al., 2023; Fifty et al., 2024; Zhu et al., 2025) remain a dominant approach for learning such discrete representations.

However, optimizing VQ-VAEs is often unstable due to an inherent *coupling* between representation learning and codebook learning.
During training, encoder outputs must be assigned to their nearest codes via a non-differentiable $\arg\min$ operator, while the codebook vectors must simultaneously move to fit the evolving distribution of encoder outputs.
To enable gradient-based optimization, surrogate gradient estimators—most commonly the straight-through estimator (STE) (Bengio et al., 2013)—are widely adopted.
While STE enables backpropagation, it introduces a gradient mismatch: the forward pass applies a non-differentiable discretization, whereas the backward pass relies on a continuous straight-through approximation.
This gradient mismatch, together with non-stationary optimization targets induced by concurrent codebook updates, makes training sensitive and difficult to scale.
A prominent failure mode is *codebook collapse* (or dead codes): only a small subset of codes are repeatedly selected, while many codes receive few or no assignments and thus stop updating.
Importantly, this forms a self-reinforcing vicious cycle: early imbalances in code selection reduce updates to rarely chosen codes, which in turn makes them even less likely to be selected later, effectively shrinking the model’s usable discrete capacity and reducing the effective bitrate.

Prior work broadly follows two directions, each addressing the above issues only partially.

- 1.
Improved coupled codebook learning.
A line of methods improves the joint optimization of encoder/decoder and codebook via additional regularization or training heuristics (e.g., code resets, orthogonality constraints, or reparameterizations) (Yu et al., 2021; Ramesh et al., 2021; Zeghidour et al., 2021; Takida et al., 2022; Huh et al., 2023; Zheng and Vedaldi, 2023; Zhang et al., 2023; Fifty et al., 2024; Zhu et al., 2025).
While these methods are sometimes effective, they introduce extra objectives and hyperparameters, and the underlying coupling remains.
- 2.
Predefined codebooks.
Another line removes codebook learning by adopting fixed quantizers, such as lattice-based scalar quantization schemes (e.g., FSQ, LFQ) or predefined Gaussian grids (Mentzer et al., 2023; Hsu et al., 2023; Yu et al., 2024; Zhao et al., 2024; Wang et al., 2024).
These approaches are typically stable, yet they impose a rigid prior on the latent space; when the learned latent distribution deviates from the assumed grid geometry, the quantization capacity can be inefficiently used (e.g., many tokens carry redundant probability mass while other regions remain under-represented).

This paper asks a fundamental question: *Can we train the encoder–decoder without knowing the codebook at all, and only instantiate the codebook after the continuous representation has converged?*
Our key observation is that, for the decoder, quantization is effectively experienced as a bounded and local perturbation to the latent representation—namely the *quantization error*.
Therefore, instead of explicitly performing discrete quantization during training (and dealing with its optimization difficulties), we propose to train the decoder to be robust to *appropriate* latent perturbations that emulate the effect of quantization.
If the perturbation distribution matches the magnitude and locality of the quantization error incurred at inference time, the model can be trained without an explicit codebook, and the codebook can be generated after convergence.

To instantiate this idea, we propose VP-VAE (Vector Perturbation VAE).
VP-VAE replaces the discrete quantization operator in training with an *adaptive vector perturbation* mechanism.
The key challenge is to design perturbations that serve as a good approximation for quantization error: a naive choice such as isotropic Gaussian noise is insufficient, because it alters the latent distribution and frequently pushes samples into low-density regions, forcing the decoder to allocate capacity to reconstruct from latent vectors that are unlikely to occur under the data distribution.
In contrast, VP-VAE generates perturbations via a Metropolis–Hastings (MH) sampling procedure with non-parametric density estimation, so that the perturbed latent vectors (denoted by $\tilde{z}$) remain in high-probability regions of the original latent space (denoted by $z$).
At the same time, the perturbation magnitude is designed to be aligned with the expected quantization error implied by a target codebook size.
In experiments, this decoupled training paradigm yields improved reconstruction quality and substantially more balanced token usage.

Moreover, under a simplifying assumption that each latent dimension is constrained to an (approximately) uniform distribution over a bounded interval, our general framework naturally simplifies to a lightweight scalar variant, namely FSP (Finite Scalar Perturbation).
We show that FSP aligns with the principles of optimal scalar quantization in the Lloyd–Max sense, and outperforms strong fixed-quantizer baselines such as FSQ (Mentzer et al., 2023; Brendel et al., 2024b; Parker et al., 2024).
Instead of quantizing to grid edges, FSP corresponds to training with centered perturbations and decoding with interval centroids, providing both a solid theoretical foundation and superior experimental performance.

Our contributions are summarized as follows:

$\bullet$ We propose VP-VAE, a new training paradigm that decouples representation learning from codebook learning, effectively preventing codebook collapse and enabling a more stable, theoretically grounded, and flexible approach to learning discrete representations from continuous signals.

$\bullet$ We develop a perturbation mechanism based on Metropolis–Hastings sampling that injects *scale-adaptive* and *distribution-consistent* latent noise, serving as an approximation for quantization error without requiring a learned codebook during training.

$\bullet$ We derive FSP, a lightweight scalar variant that assumes an approximately uniform latent space, providing both theoretical insight into and empirical improvements over FSQ-style fixed quantizers.

$\bullet$ Across image and audio benchmarks, VP-VAE and FSP consistently improve reconstruction fidelity while achieving higher and more balanced codebook utilization.

## 2 Related Work

Existing approaches to neural discrete representation learning largely fall into two categories: methods that seek to mitigate the instability arising from joint model–codebook optimization, and methods that bypass this coupling by relying on fixed or predefined quantization targets.

### 2.1 Coupled Codebook–Network Optimization

The foundational VQ-VAE (Van Den Oord et al., 2017) introduced the concept of learning a discrete latent bottleneck via nearest-neighbor lookup, utilizing the Straight-Through Estimator (STE) to approximate gradients for the encoder.
While effective, this paradigm suffers inherently from the “codebook collapse” phenomenon, where a significant portion of code vectors receive no gradients and become inactive.
This occurs because the codebook update (moving codes toward encoder outputs) and the encoder update (mapping inputs to codes) are coupled in a feedback loop that rewards already-frequent codes, often resulting in low effective bitrate (Takida et al., 2022; Mentzer et al., 2023).

To mitigate this, substantial efforts have focused on regularization and heuristic stabilization.
Early works employed Exponential Moving Average (EMA) updates and “restart” heuristics—manually re-initializing dead codes with active encoder outputs—to maintain high codebook utilization (Razavi et al., 2019; Zheng and Vedaldi, 2023).
More recent approaches focus on refining gradient estimation and update dynamics.
For instance, SQ-VAE (Takida et al., 2022) replaces quantization with stochastic approximation via Gaussian smoothing to improve gradient estimation.
Similarly, SimVQ (Zhu et al., 2025) argues that the collapse stems from disjoint optimization updates; it proposes reparameterizing the codebook via a learnable linear layer to ensure gradients flow to all codes simultaneously.

Despite these advancements, these methods fundamentally retain the coupled training dynamics: the encoder must still track a moving target (the codebook), and the codebook must adapt to a shifting encoder distribution.
This interdependence often necessitates complex extra losses (e.g., commitment, orthogonality, or entropy loss) and renders the training process sensitive to hyperparameter selection (Chang et al., 2022; Shin et al., 2023; Zhu et al., 2025).
Our proposed VP-VAE differs by removing this coupling entirely during the representation learning phase.

### 2.2 Predefined Codebook Quantization

Recognizing the drawbacks of joint optimization, a parallel line of research explores decoupling representation learning from codebook generation by imposing fixed geometric priors.
RandomVQ (Chiu et al., 2022) demonstrated that in self-supervised speech learning, a randomly initialized and frozen codebook can yield competitive representations, suggesting that the precise location of codes is often less critical than the discretization mechanism itself.

Building on this, Finite Scalar Quantization (FSQ) (Mentzer et al., 2023) and LFQ (Yu et al., 2024) eliminate the vector lookup entirely, instead projecting latent dimensions onto fixed scalar grids forming a hypercube.
By removing the learnable codebook, these methods achieve high stability and near-100% codebook utilization.
However, they impose a rigid bias: they assume the encoder can reshape the latent distribution into a factorial uniform grid.
If the natural latent distribution of the data is non-uniform or complex, these rigid grids lead to inefficient space allocation.

Most recently, TokenBridge (Wang et al., 2024) proposes discretizing the latent space of a pre-trained VAE.
While effective as a post-training strategy, it fundamentally relies on the *predefined* structure imposed by the VAE’s KL-regularization, which forces the data to approximate a standard normal distribution; this continuous space is then sliced into discrete tokens using fixed Gaussian percentiles.
Conceptually, this amounts to utilizing a *predefined Gaussian grid*.
Like the scalar grid methods mentioned above, this approach shifts the burden of adaptation entirely onto the encoder, forcing the latent distribution to conform to a specific prior to adapt the quantizer.
In contrast, VP-VAE imposes no such global distributional constraints (e.g., Gaussianity).
Instead, our adaptive perturbation mechanism generates local perturbations that are consistent with the empirical latent density, training the decoder to be robust to quantization error without forcing the latent distribution to match a predefined grid.

## 3 Methodology

This section introduces VP-VAE, a training framework that decouples representation learning from codebook learning by replacing discrete quantization with adaptive vector perturbation.
We elaborate on the model architecture, the perturbation-based training procedure, and the generation of an explicit codebook for inference (§[3.2](https://arxiv.org/html/2602.17133v1#S3.SS2)).
We then present Finite Scalar Perturbation (FSP), a specialized variant of VP-VAE. When the latent distribution is approximately uniform, FSP enables a simplified perturbation and quantization scheme with reduced computational complexity (§[3.3](https://arxiv.org/html/2602.17133v1#S3.SS3)).

### 3.1 Setup and Notation

Let $x$ denote an input sample, such as an image or an audio signal.
An encoder $E$ maps $x$ to a sequence of token features $\{h_{t}\}_{t=1}^{T}$, where $h_{t}\in\mathbb{R}^{C}$ and $T$ is the number of tokens per sample.
In standard VQ-VAEs, quantization is performed directly in this $C$-dimensional embedding space (e.g., $C=128$ or higher).
Instead, in this paper, we propose a perturbation mechanism based on non-parametric density estimation, designed to ensure that perturbed latent vectors remain within high-density regions of the original latent distribution.
However, performing reliable density estimation in such high-dimensional spaces is non-trivial due to the curse of dimensionality (Silverman, 1986).

To address this issue, VP-VAE introduces a *low-dimensional quantization bottleneck*.
Specifically, each token feature is first projected into a lower-dimensional space via a learnable down-projection, and later mapped back to the original embedding space through a corresponding up-projection:

$$ $\displaystyle z=P_{\downarrow}(h_{t})\in\mathbb{R}^{d},$ $\displaystyle\tilde{h}_{t}=P_{\uparrow}(\tilde{z})\in\mathbb{R}^{C},$ (1) $$

where $P_{\downarrow}:\mathbb{R}^{C}\to\mathbb{R}^{d}$ and $P_{\uparrow}:\mathbb{R}^{d}\to\mathbb{R}^{C}$ denote the down- and up-projection operators, respectively.
Here, $z$ and $\tilde{z}$ represent the latent vectors before and after perturbation, as detailed in §[3.2.2](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS2).
Both perturbation (during training) and quantization (during testing) are performed in the compressed $d$-dimensional space (typically $d\leq 16$), while the decoder reconstructs the input as $\hat{x}=D(\{\tilde{h}_{t}\}_{t=1}^{T})$ from the up-projected features.
This bottleneck enables reliable density estimation while preserving sufficient representational capacity for high-fidelity reconstruction.

### 3.2 VP-VAE: From Quantization to Perturbation

In standard VQ, each latent vector $z$ is mapped to its nearest code $q(z)\in\mathcal{Q}$, introducing a *quantization error* $\epsilon=q(z)-z$.
Our key viewpoint is that, this operation can be viewed as injecting a structured perturbation into the latent space.
This viewpoint motivates a fundamentally different training strategy: instead of performing discrete quantization during training—which requires surrogate gradients and simultaneous codebook updates—we can train the decoder to be robust to latent perturbations that mimic the effect of quantization.

Following this point, VP-VAE replaces the discrete quantization operator with an explicit perturbation operator $\mathcal{T}$:

$$ $\displaystyle\tilde{z}=\mathcal{T}(z;\mathcal{S}),$ $\displaystyle\hat{x}=D\big(\{P_{\uparrow}(\tilde{z})\}_{t=1}^{T}\big),$ (2) $$

where $\mathcal{S}$ is a memory buffer containing recent latent vectors, used to approximate the current latent distribution (detailed in §[3.2.1](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS1)).
If the perturbation distribution accurately emulates quantization error, the model can learn to reconstruct from perturbed latent vectors without ever seeing an explicit codebook during training.

For this emulation to be effective, the perturbation operator $\mathcal{T}$ must satisfy two critical requirements:

- 1.
Scale alignment: The perturbation magnitude should match the expected quantization error implied by a target codebook size $K$.
Perturbations that are too small may fail to prepare the decoder for inference-time quantization, while overly large perturbations unnecessarily degrade reconstruction quality.
- 2.
Distribution consistency: Perturbed latent vectors $\tilde{z}$ should remain within high-density regions of the original latent distribution.
Naive perturbation schemes, such as isotropic Gaussian noise, are ineffective because they often push samples into low-density or out-of-distribution regions, forcing the decoder to model unlikely latent configurations and wasting representational capacity.

#### 3.2.1 Scale Alignment via Adaptive Radius Estimation

To align the perturbation scale with the expected quantization error, we estimate how far a latent vector would typically move under nearest-neighbor quantization with a codebook of size $K$.
Intuitively, if the latent space were partitioned into $K$ Voronoi cells with roughly equal probability mass, each cell would contain approximately $1/K$ of the latent distribution, and the quantization error would scale with the local cell radius.

We realize this intuition using a non-parametric approach.
We maintain a first-in-first-out (FIFO) queue $\mathcal{S}=\{s_{i}\}_{i=1}^{|\mathcal{S}|}$ of recent latent vectors in $\mathbb{R}^{d}$.
To mitigate redundancy caused by strong intra-sample token correlations, we update $\mathcal{S}$ by randomly subsampling a small fraction of tokens from each minibatch rather than storing all tokens.

Given the queue $\mathcal{S}$, let $D_{m}(z|\mathcal{S})$ denote the Euclidean distance from $z$ to its $m$-th nearest neighbor in $\mathcal{S}$.
For a target codebook size $K$, we define the *local quantization radius* as:

$$ $M=\left\lceil\frac{|\mathcal{S}|}{K}\right\rceil,\qquad R(z)=\eta\,D_{M}(z|\mathcal{S}),$ (3) $$

where $\eta>0$ is a hyperparameter controlling the perturbation scale.
The rationale is as follows: if $K$ codebook vectors were to partition the $|\mathcal{S}|$ samples into regions of equal size, each region would contain approximately $M=|\mathcal{S}|/K$ samples.
The distance to the $M$-th nearest neighbor thus serves as a data-adaptive indicator for the radius of the local Voronoi cell, automatically adjusting to both the local density and the target codebook size.

#### 3.2.2 Distribution Consistency via Metropolis–Hastings Sampling

With the perturbation scale determined by Eq. ([3](https://arxiv.org/html/2602.17133v1#S3.E3)), a naive approach would sample perturbations uniformly from a ball of radius $R(z)$.
However, such isotropic noise ignores the geometry of the latent distribution, i.e., perturbations may push latent vectors into low-density regions or even outside the support of the learned latent distribution, creating a train–test mismatch.

To preserve distributional consistency, VP-VAE generates perturbations using a Metropolis–Hastings (MH) transition (Metropolis et al., 1953; Hastings, 1970) whose stationary distribution matches the empirical latent density.
A fundamental property of the MH algorithm is that the transition kernel leaves the target distribution $\pi$ invariant: if $z\sim\pi$, then $\tilde{z}\sim\pi$ as well (Robert et al., 1999).
MH sampling naturally rejects proposals that move into low-density regions, ensuring that accepted perturbations keep $\tilde{z}$ within high-probability regions.

##### Target density estimation.

We approximate the latent density using a $k$-nearest-neighbor (kNN) estimator:

$$ $\pi(z)\propto\frac{1}{\big(D_{k}(z|\mathcal{S})\big)^{d}},$ (4) $$

where $k$ is a small constant, $D_{k}(z|\mathcal{S})$ is the distance to the $k$-th nearest neighbor, and $d$ is the latent dimension.
For brevity, we write $D_{k}(z)=D_{k}(z|\mathcal{S})$ and $D_{M}(z)=D_{M}(z|\mathcal{S})$ when the context is clear.

##### Proposal distribution.

Given a current latent vector $z$, we propose a candidate $z^{\prime}$ by sampling uniformly from the $d$-dimensional ball centered at $z$ with radius $R(z)$:

$$ $z^{\prime}=z+u,\qquad u\sim\mathrm{Unif}\big(\mathcal{B}(0,R(z))\big).$ (5) $$

Uniform sampling in a ball can be implemented by sampling a random direction $v\sim\mathcal{N}(0,I_{d})$ and a random radius $r=R(z)\cdot\rho^{1/d}$ with $\rho\sim\mathrm{Unif}[0,1]$, then setting $u=r\cdot v/\|v\|_{2}$.

##### Acceptance probability.

The MH acceptance probability determines whether a proposed perturbation is applied, which is defined as:

$$ $\alpha(z,z^{\prime})=\min\left(1,\ \frac{\pi(z^{\prime})\,g(z|z^{\prime})}{\pi(z)\,g(z^{\prime}|z)}\right),$ (6) $$

where $g(\cdot|\cdot)$ denotes the proposal density.
Intuitively, this ratio compares how likely the proposed latent vector is under the target density relative to the current latent vector $z$.
As a result, proposals that move toward higher-density regions are likely to be accepted, whereas moves into lower-density regions are rejected.

For the uniform-ball proposal, the reverse density $g(z|z^{\prime})$ is nonzero only if $\|z-z^{\prime}\|\leq R(z^{\prime})$.
Therefore, we explicitly enforce the support condition:

$$ $\alpha(z,z^{\prime})=0\quad\text{if }\|z-z^{\prime}\|>R(z^{\prime}).$ (7) $$

When both support conditions are satisfied, substituting Eq. ([4](https://arxiv.org/html/2602.17133v1#S3.E4)) and the proposal densities yields:

$$ $\alpha(z,z^{\prime})=\min\left(1,\ \left(\frac{D_{k}(z)\cdot D_{M}(z)}{D_{k}(z^{\prime})\cdot D_{M}(z^{\prime})}\right)^{\!d}\right),$ (8) $$

where we use $R(\cdot)=\eta D_{M}(\cdot)$ from Eq. ([3](https://arxiv.org/html/2602.17133v1#S3.E3)).

##### Perturbation operator.

The final perturbed latent vector is obtained by applying the MH accept–reject rule:

$$ $\tilde{z}=\begin{cases}z^{\prime},&\text{with probability }\alpha(z,z^{\prime}),\\ z,&\text{otherwise}.\end{cases}$ (9) $$

If the proposal is rejected, the latent vector passes through unperturbed.
The complete procedure is summarized in [Algorithm 1](https://arxiv.org/html/2602.17133v1#alg1).

Figure: Algorithm 1 Vector Perturbation

#### 3.2.3 Optimization Objective

VP-VAE is trained end-to-end with a reconstruction loss augmented by a latent normalization regularizer:

$$ $\mathcal{L}=\mathcal{L}_{\text{rec}}(x,\hat{x})+\mathcal{L}_{\text{norm}}.$ (10) $$

The reconstruction loss $\mathcal{L}_{\text{rec}}$ can be any differentiable metric appropriate for the data modality (e.g., MSE, perceptual loss, or a combination thereof).

The normalization regularizer encourages the latent distribution to be zero-mean and unit-variance along each dimension:

$$ $\mathcal{L}_{\text{norm}}=\lambda_{1}\|\mu_{batch}\|_{2}^{2}+\lambda_{2}\|\sigma_{batch}^{2}-\mathbf{1}\|_{2}^{2},$ (11) $$

where $\mu_{batch}$ and $\sigma_{batch}^{2}$ are the minibatch mean and variance of the $d$-dimensional latent vectors $z$.
This regularizer serves two purposes: (i) it prevents latent collapse or explosion, and (ii) it facilitates scale estimation by ensuring that the queue $\mathcal{S}$ contains samples from a well-behaved distribution.

#### 3.2.4 Codebook Generation

A distinguishing feature of VP-VAE is that no codebook exists during training.
Once training converges, we generate an explicit codebook offline by clustering the learned latent representations.
Specifically, we run the encoder over the training set to collect a large set of latent vectors $z$ in $\mathbb{R}^{d}$, then apply K-Means clustering (with K-Means++ initialization (Arthur and Vassilvitskii, 2006)) to obtain $K$ centroids, forming the codebook $\mathcal{Q}=\{c_{j}\}_{j=1}^{K}$.

At inference time, each latent vector $z$ is quantized via nearest-neighbor assignment:

$$ $q(z)=\arg\min_{c\in\mathcal{Q}}\|z-c\|_{2},$ (12) $$

followed by the up-projection $P_{\uparrow}$ and decoding.
Because the encoder and decoder are trained to tolerate perturbations of magnitude $R(z)$, and K-Means approximately minimizes quantization error, our model generalizes well to discrete quantization at inference time without requiring additional fine-tuning.

### 3.3 Finite Scalar Perturbation (FSP)

The general VP-VAE framework handles arbitrary latent distributions via non-parametric density estimation.
However, when the latent distribution can be constrained to a simple and known form, the perturbation mechanism can be significantly simplified.
Under this observation, we further derive FSP (Finite Scalar Perturbation), a lightweight variant of VP-VAE that arises under the assumption of approximately uniform latent variables.

##### Uniform latent variables via “CDF-like” activation.

Consider constraining each latent dimension to the unit interval $[0,1]$ using a monotone “CDF-like” activation.
Let $a\in\mathbb{R}^{d}$ denote pre-activation latent vectors, and define:

$$ $z=g(a)\in[0,1]^{d},$ (13) $$

where $g$ is a smooth, monotonically increasing function (e.g., the sigmoid or a normal CDF).
In the idealized case where each $a_{i}$ follows a distribution whose CDF equals $g$, the probability integral transform implies that each $z_{i}$ is marginally uniform on $[0,1]$.
In practice, we do not enforce exact matching of the full distribution; instead, we use a simple moment-matching regularizer on $a$ (described below) and treat the resulting near-uniformity as an approximation.

Concretely, for FSP we apply a normalization regularizer to the pre-activation variables $a$ rather than the post-activation variables $z$:

$$ $\mathcal{L}_{\text{norm}}^{\text{FSP}}=\lambda_{1}\|\mu_{batch}(a)\|_{2}^{2}+\lambda_{2}\|\sigma_{batch}^{2}(a)-\sigma_{g}^{2}\mathbf{1}\|_{2}^{2},$ (14) $$

where $\sigma_{g}^{2}$ is chosen based on the activation $g$ so that $z=g(a)$ is close to uniform on $[0,1]$ in practice.
For example, we use $\sigma_{g}^{2}=1$ for the normal-CDF activation;
$\sigma_{g}^{2}\approx 0.8225$ for tanh activation ($g(a)=(\tanh(a)+1)/2$);
and $\sigma_{g}^{2}\approx 3.29$ for sigmoid activation.

##### Lloyd–Max centroids.

Under the uniform distribution, optimal scalar quantization is characterized by the Lloyd–Max conditions (Max, 1960; Lloyd, 1982).
For a uniform source on $[0,1]$ with $L$ quantization levels, the optimal reconstruction points correspond to the centroids of equal-width intervals:

$$ $\mathcal{C}=\left\{\frac{\ell+1/2}{L}\right\}_{\ell=0}^{L-1}.$ (15) $$

FSP quantizes to these centroids rather than grid boundaries, which provides a principled improvement over rounding-based fixed grids (e.g., FSQ (Mentzer et al., 2023)) when the (approximate) uniform-latent assumption holds.

##### Simplified perturbation mechanism.

When the latent variables are (approximately) uniformly distributed on a bounded support, density estimation becomes trivial: the density is constant within the support and zero outside.
Consequently, the MH acceptance criterion reduces to a simple support check—any proposal that remains within $[0,1]^{d}$ is accepted.
Moreover, the perturbation scale naturally aligns with the quantization bin width $1/L$.

This yields a fully factorized, per-dimension perturbation rule.
For dimension $i$ with $L_{i}$ quantization levels, we propose:

$$ $z^{\prime}_{i}=z_{i}+u_{i},\qquad u_{i}\sim\mathcal{U}\!\left(-\frac{\eta}{2L_{i}},\,\frac{\eta}{2L_{i}}\right),$ (16) $$

and accept iff the proposal remains in the valid range:

$$ $\tilde{z}=\begin{cases}z^{\prime},&\text{if }z^{\prime}\in[0,1]^{d},\\ z,&\text{otherwise}.\end{cases}$ (17) $$

##### Training: perturb-or-quantize mixture.

In practice, we find that mixing perturbation with explicit quantization during training improves gradient quality and accelerates convergence.
Inspired by noise-injection techniques for fixed quantizers (Brendel et al., 2024b), FSP uses a stochastic mixture: at each forward pass, bounded perturbation (Eqs. ([16](https://arxiv.org/html/2602.17133v1#S3.E16))–([17](https://arxiv.org/html/2602.17133v1#S3.E17))) is applied with probability $1/2$, while centroid quantization with STE is applied with probability $1/2$:

$$ $\tilde{z}_{i}=\frac{\ell_{i}+1/2}{L_{i}},\quad\ell_{i}=\mathrm{clip}\big(\lfloor L_{i}z_{i}\rfloor,0,L_{i}-1\big).$ (18) $$

In experiments, FSP consistently outperforms rounding-based fixed quantizers (e.g., FSQ) while remaining simpler than full VP-VAE.
However, when the latent distribution deviates from uniformity, VP-VAE retains a clear advantage due to its greater modeling flexibility.

**Table 1: In-domain reconstruction results. Models are trained and evaluated on COCO (image) and LibriSpeech (audio). “-” indicates training failure due to severe codebook collapse. VP-VAE and FSP demonstrate consistent stability and high fidelity across both modalities, whereas baselines often degrade in specific task.**
| Method | Codebook<br>Size | Image (COCO) |  | Audio (LibriSpeech) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CVU | LPIPS ↓ | PSNR ↑ | SSIM ↑ |  | CVU | PESQ ↑ | STOI ↑ |  |  |
| VQ-VAE | 256 | 0.3679 | 0.2053 | 23.1662 | 0.6704 |  | - | - | - |
| SimVQ | 256 | 0.9429 | 0.1846 | 23.5356 | 0.6918 |  | 0.0040 | 1.2524 | 0.4112 |
| TokenBridge | 256 | 0.9888 | 0.6075 | 11.5386 | 0.2026 |  | 0.9888 | 1.1802 | 0.2881 |
| FSQ | 256 | 0.8607 | 0.2101 | 22.9390 | 0.6689 |  | 0.0960 | 1.2763 | 0.7973 |
| FSP | 256 | 0.8852 | 0.1884 | 23.8131 | 0.6940 |  | 0.1372 | 2.2653 | 0.9083 |
| VP-VAE | 256 | 0.7970 | 0.1929 | 23.4973 | 0.6881 |  | 0.1352 | 2.2718 | 0.9100 |
| VQ-VAE | 1024 | 0.3557 | 0.1821 | 23.727 | 0.6946 |  | 0.0010 | 1.4005 | 0.3628 |
| SimVQ | 1024 | 0.8919 | 0.1662 | 24.0102 | 0.7176 |  | 0.0010 | 1.3213 | 0.4025 |
| TokenBridge | 1024 | 0.9887 | 0.6078 | 11.5145 | 0.1984 |  | 0.9888 | 1.1801 | 0.3124 |
| FSQ | 1024 | 0.8059 | 0.1839 | 23.6006 | 0.6955 |  | 0.0513 | 2.1738 | 0.9001 |
| FSP | 1024 | 0.8515 | 0.1722 | 24.1910 | 0.7177 |  | 0.0810 | 2.4458 | 0.9191 |
| VP-VAE | 1024 | 0.8102 | 0.1717 | 23.8878 | 0.7182 |  | 0.0687 | 2.3826 | 0.9160 |
| VQ-VAE | 4096 | 0.2401 | 0.1693 | 24.1303 | 0.7245 |  | - | - | - |
| SimVQ | 4096 | 0.8604 | 0.1497 | 24.8338 | 0.7455 |  | 0.0002 | 1.2519 | 0.3972 |
| TokenBridge | 4096 | 0.9887 | 0.6086 | 11.5931 | 0.2011 |  | 0.9887 | 1.2249 | 0.3160 |
| FSQ | 4096 | 0.7967 | 0.1647 | 24.4481 | 0.7273 |  | 0.0294 | 2.3821 | 0.9182 |
| FSP | 4096 | 0.8227 | 0.1521 | 24.7291 | 0.7441 |  | 0.0368 | 2.4712 | 0.9273 |
| VP-VAE | 4096 | 0.8180 | 0.1587 | 24.6702 | 0.7488 |  | 0.0564 | 2.4499 | 0.9259 |
| VQ-VAE | 16384 | 0.2826 | 0.1489 | 24.5994 | 0.7555 |  | - | - | - |
| SimVQ | 16384 | 0.8276 | 0.1387 | 25.2429 | 0.7605 |  | 0.0001 | 1.2477 | 0.4436 |
| TokenBridge | 16384 | 0.9887 | 0.6143 | 11.4920 | 0.1963 |  | 0.9887 | 1.2328 | 0.3008 |
| FSQ | 16384 | 0.7921 | 0.1552 | 24.5842 | 0.7354 |  | 0.0149 | 2.4356 | 0.9225 |
| FSP | 16384 | 0.7115 | 0.1464 | 25.1676 | 0.7577 |  | 0.0205 | 2.5699 | 0.9324 |
| VP-VAE | 16384 | 0.7957 | 0.1434 | 25.2032 | 0.7623 |  | 0.0455 | 2.5758 | 0.9339 |

**Table 2: Out-of-distribution generalization results. Models trained on COCO/LibriSpeech are evaluated on unseen datasets: ImageNet (image) and Common Voice (audio). Our decoupled training paradigm yields superior generalization compared to baseline methods.**
| Method | Codebook<br>Size | Image (ImageNet) |  | Audio (Common Voice) |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CVU | LPIPS ↓ | PSNR ↑ | SSIM ↑ |  | CVU | PESQ ↑ | STOI ↑ |  |  |
| VQ-VAE | 256 | 0.3617 | 0.2121 | 23.5942 | 0.6704 |  | - | - | - |
| SimVQ | 256 | 0.9363 | 0.1927 | 23.6599 | 0.6827 |  | 0.0041 | 1.2103 | 0.3781 |
| TokenBridge | 256 | 0.9887 | 0.6079 | 11.6525 | 0.2024 |  | 0.9887 | 1.0967 | 0.2688 |
| FSQ | 256 | 0.8574 | 0.2182 | 23.4555 | 0.6645 |  | 0.1690 | 1.2064 | 0.7630 |
| FSP | 256 | 0.8819 | 0.1989 | 23.7919 | 0.6736 |  | 0.2860 | 1.8148 | 0.8537 |
| VP-VAE | 256 | 0.7859 | 0.1989 | 23.8329 | 0.6759 |  | 0.2055 | 1.8510 | 0.8617 |
| VQ-VAE | 1024 | 0.3604 | 0.1911 | 23.8520 | 0.6851 |  | 0.0010 | 1.2103 | 0.3371 |
| SimVQ | 1024 | 0.8736 | 0.1719 | 24.3343 | 0.7070 |  | 0.0010 | 1.1568 | 0.3627 |
| TokenBridge | 1024 | 0.9887 | 0.6108 | 11.4561 | 0.1985 |  | 0.9887 | 1.1059 | 0.2856 |
| FSQ | 1024 | 0.8000 | 0.1931 | 24.0973 | 0.6878 |  | 0.1012 | 1.7776 | 0.8498 |
| FSP | 1024 | 0.8466 | 0.1795 | 24.2681 | 0.7115 |  | 0.1776 | 1.9549 | 0.8676 |
| VP-VAE | 1024 | 0.7912 | 0.1790 | 24.4100 | 0.7115 |  | 0.1441 | 1.9697 | 0.8701 |
| VQ-VAE | 4096 | 0.2371 | 0.1770 | 24.3765 | 0.7113 |  | - | - | - |
| SimVQ | 4096 | 0.8504 | 0.1578 | 24.8154 | 0.7339 |  | 0.0002 | 1.1440 | 0.3713 |
| TokenBridge | 4096 | 0.9887 | 0.6071 | 11.5554 | 0.1920 |  | 0.9888 | 1.1017 | 0.2787 |
| FSQ | 4096 | 0.7759 | 0.1749 | 24.4162 | 0.7065 |  | 0.0670 | 1.9712 | 0.8707 |
| FSP | 4096 | 0.8074 | 0.1598 | 24.9106 | 0.7323 |  | 0.0936 | 2.0564 | 0.8840 |
| VP-VAE | 4096 | 0.8004 | 0.1653 | 24.9336 | 0.7400 |  | 0.1151 | 2.0841 | 0.8907 |
| VQ-VAE | 16384 | 0.2621 | 0.1592 | 24.8349 | 0.7359 |  | - | - | - |
| SimVQ | 16384 | 0.8164 | 0.1458 | 25.3304 | 0.7496 |  | 0.0001 | 1.1807 | 0.4002 |
| TokenBridge | 16384 | 0.9888 | 0.6166 | 11.4447 | 0.1969 |  | 0.9887 | 1.1202 | 0.2722 |
| FSQ | 16384 | 0.7756 | 0.1627 | 24.8198 | 0.7237 |  | 0.0499 | 1.9566 | 0.8721 |
| FSP | 16384 | 0.6887 | 0.1534 | 25.3974 | 0.7466 |  | 0.0541 | 2.1380 | 0.8906 |
| VP-VAE | 16384 | 0.7718 | 0.1481 | 25.4315 | 0.7535 |  | 0.1218 | 2.1803 | 0.9003 |

## 4 Experiments

### 4.1 Experimental Setup

We evaluate VP-VAE and FSP on two tasks: image reconstruction and audio compression.
To comprehensively assess the trade-off between reconstruction fidelity and codebook utilization, we conduct experiments across four codebook sizes: $K\in\{256,1024,4096,16384\}$.

#### 4.1.1 Datasets and Backbones

Image.
We use the COCO 2017 (Lin et al., 2014) dataset for training.
Evaluation is performed on both the COCO validation set and the ImageNet (Deng et al., 2009) validation set to test out-of-distribution generalization.
All images are resized so that the shortest side is 256, then center-cropped to $256\times 256$.
For image experiments, we utilize the VQGAN architecture (Esser et al., 2021).
The encoder downsamples images by a factor of 8, resulting in a $32\times 32$ token grid for $256^{2}$ inputs.

Audio.
Training is performed on the LibriSpeech (Panayotov et al., 2015) train-clean-460 and train-other-500 splits.
Evaluation uses the LibriSpeech test-clean and test-other splits, as well as the Common Voice (v18.0) (Mozilla, 2024) English test set.
All audio is resampled to 16kHz.
For audio experiments, we employ the encoder–decoder from SQCodec (Zhai et al., 2025).
The temporal downsampling results in a frame rate of approximately 166.67 Hz.

To ensure a fair comparison that focuses solely on the quantization mechanism—and separates reconstruction capability from the hallucination effects of adversarial training—we do not use discriminators or GAN losses for any method.

#### 4.1.2 Evaluation Metrics

Reconstruction quality.
We report standard fidelity metrics.
For images: PSNR, SSIM (Wang et al., 2004), and LPIPS (VGG) (Zhang et al., 2018).
For audio: PESQ (Wideband) (Rix et al., 2001) and STOI (Taal et al., 2010).

Codebook utilization.
Standard “codebook usage” metrics, such as the percentage of codes used at least once, often saturate near 100% on large validation sets and fail to reflect how *evenly* the codebook is utilized.
To measure the balance of codebook utilization, we propose the Codebook Valid Usage (CVU):

$$ $\text{CVU}=\frac{\exp\left(-\sum_{c\in\mathcal{Q}}p(c)\log p(c)\right)}{K},$ (19) $$

where $p(c)$ is the empirical selection probability of code $c$ over all test tokens.
CVU represents the effective number of active codes divided by the target codebook size $K$.
A CVU of 1.0 indicates a perfectly uniform utilization (maximum entropy), while lower values indicate imbalance.

### 4.2 Results

We compare VP-VAE and FSP against the standard coupled baseline (VQ-VAE (Van Den Oord et al., 2017)), a state-of-the-art coupled method (SimVQ (Zhu et al., 2025)), a state-of-the-art fixed-quantizer approach (FSQ (Parker et al., 2024)), and a recent predefined-grid discretization approach (TokenBridge (Wang et al., 2024)).
Quantitative results for in-domain and out-of-distribution settings are presented in [Table 1](https://arxiv.org/html/2602.17133v1#S3.T1) and [Table 2](https://arxiv.org/html/2602.17133v1#S3.T2), respectively.

Cross-Modality Consistency and Stability.
A key finding is that VP-VAE and FSP exhibit superior adaptability across data modalities, whereas baselines tend to specialize or fail.
As shown in [Table 1](https://arxiv.org/html/2602.17133v1#S3.T1), SimVQ excels on images, achieving the lowest LPIPS at all codebook sizes.
However, this performance does not transfer to audio.
On LibriSpeech, SimVQ exhibits near-collapse behavior in code usage, with extremely low CVU (from 0.004 at $K{=}256$ down to $10^{-4}$ at $K{=}16384$), accompanied by substantially degraded reconstruction quality (PESQ around 1.25–1.40), barely surpassing the classical VQ-VAE.
We hypothesize that this instability in audio stems from the statistical nature of the signal: silence and low-amplitude segments yield highly concentrated encoder outputs early in training, and coupled codebook learning can amplify small early assignment biases into a self-reinforcing imbalance (cf. §[1](https://arxiv.org/html/2602.17133v1#S1)) and leading to severe imbalance (or even training failure, as observed for VQ-VAE under some settings).

Conversely, FSQ is consistently stable on audio (e.g., STOI up to 0.9225 at $K{=}16384$) and avoids outright collapse due to its fixed quantization structure.
Nevertheless, its rigid grid prior can be suboptimal for image latents, leading to weaker reconstruction on COCO compared to our methods (e.g., at $K{=}1024$, FSQ achieves 23.60 dB PSNR vs. 24.19 for FSP).

In contrast, VP-VAE and FSP deliver competitive or state-of-the-art results on *both* modalities: on COCO, FSP achieves the highest PSNR at $K{\in}\{256,1024\}$, while VP-VAE obtains the best SSIM at $K{\in}\{1024,4096,16384\}$; on LibriSpeech, both methods cconsistently occupy the top two positions in PESQ and STOI across all codebook sizes, substantially outperforming coupled baselines and improving over fixed quantization.
Overall, these results indicate that decoupling representation learning from discrete code optimization yields a more robust quantization mechanism that transfers across modalities without requiring modality-specific heuristic techniques.

Out-of-Distribution Generalization.
[Table 2](https://arxiv.org/html/2602.17133v1#S3.T2) highlights the robustness of our approach under distribution shift.
On ImageNet, VP-VAE achieves the highest PSNR across all codebook sizes and the best SSIM at three of four cases.
The advantage is even more pronounced in audio.
On Common Voice, VP-VAE consistently achieves the best PESQ and STOI.
Notably, the gap between VP-VAE and strong baselines increases under OOD evaluation: for example, at $K=16384$, VP-VAE’s STOI score drops only marginally from 0.9339 (LibriSpeech) to 0.9003 (Common Voice).
In contrast, FSQ experiences a much sharper degradation, dropping from 0.9225 to 0.8721.

We attribute this robustness to the training objective: VP-VAE optimizes the decoder to tolerate a distribution of local perturbations rather than over-fitting to specific, singular codebook vectors.
This effectively regularizes the latent space, ensuring that quantization errors during inference (even if slightly shifted due to out-of-distribution inputs) remain within the decoder’s tolerance.

FSP v.s. FSQ.
Our theoretical derivation in §[3.3](https://arxiv.org/html/2602.17133v1#S3.SS3) suggests that FSP is the theoretically principled generalization of FSQ for uniform latents.
The empirical data strongly supports this.
Across all experimental conditions—both modalities, all codebook sizes, and both ID/OOD evaluations—FSP consistently outperforms FSQ.
This validates that using Lloyd–Max centroids and proper perturbation intervals is superior to the rounding-to-integer heuristic employed by FSQ.
Crucially, FSP retains the computational simplicity of FSQ while closing the performance gap with fully adaptive methods like VP-VAE.

More experiments and analysis are provided in §[B](https://arxiv.org/html/2602.17133v1#A2).

## 5 Conclusion

We propose VP-VAE, a new perspective on discrete representation learning that treats quantization primarily as a *structured latent perturbation* and, crucially, removes the need for an explicit codebook during training.
VP-VAE demonstrates that designing training-time perturbations to faithfully emulate inference-time quantization error is a principled and effective alternative to coupled codebook learning, offering a promising path toward stable, scalable discrete tokenizers for modern generative modeling.
Experiments on image and audio benchmarks validate the effectiveness of our approach.

## Impact Statement

This paper presents work whose goal is to advance the field of Machine
Learning. There are many potential societal consequences of our work, none
which we feel must be specifically highlighted here.

## References

- D. Arthur and S. Vassilvitskii (2006)
K-means++: the advantages of careful seeding.
Technical report
Stanford.
Cited by: [§3.2.4](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS4.p1.4).
- Y. Bengio, N. Léonard, and A. Courville (2013)
Estimating or propagating gradients through stochastic neurons for conditional computation.
arXiv.
External Links: 1308.3432,
[Document](https://dx.doi.org/10.48550/arXiv.1308.3432)
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p2.1).
- Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour (2023)
AudioLM: A language modeling approach to audio generation.
IEEE/ACM Transactions on Audio, Speech, and Language Processing 31, pp. 2523–2533.
External Links: ISSN 2329-9304,
[Document](https://dx.doi.org/10.1109/TASLP.2023.3288409)
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- A. Brendel, N. Pia, K. Gupta, L. Behringer, G. Fuchs, and M. Multrus (2024a)
Neural speech coding for real-time communications using constant bitrate scalar quantization.
IEEE Journal of Selected Topics in Signal Processing, pp. 1–15.
External Links: ISSN 1941-0484,
[Document](https://dx.doi.org/10.1109/JSTSP.2024.3491575)
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- A. Brendel, N. Pia, K. Gupta, L. Behringer, G. Fuchs, and M. Multrus (2024b)
Neural speech coding for real-time communications using constant bitrate scalar quantization.
IEEE Journal of Selected Topics in Signal Processing, pp. 1–15.
External Links: ISSN 1941-0484,
[Document](https://dx.doi.org/10.1109/JSTSP.2024.3491575)
Cited by: [item 3](https://arxiv.org/html/2602.17133v1#A1.I1.i3.p1.2),
[§1](https://arxiv.org/html/2602.17133v1#S1.p6.1),
[§3.3](https://arxiv.org/html/2602.17133v1#S3.SS3.SSS0.Px4.p1.2).
- H. Chang, H. Zhang, L. Jiang, C. Liu, and W. T. Freeman (2022)
MaskGIT: Masked generative image transformer.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 11315–11325.
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p3.1).
- S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei (2025)
Neural codec language models are zero-shot text to speech synthesizers.
IEEE Transactions on Audio, Speech and Language Processing 33, pp. 705–718.
External Links: ISSN 2998-4173,
[Document](https://dx.doi.org/10.1109/TASLPRO.2025.3530270)
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- C. Chiu, J. Qin, Y. Zhang, J. Yu, and Y. Wu (2022)
Self-supervised learning with random-projection quantizer for speech recognition.
In Proceedings of the International Conference on Machine Learning (ICML),
pp. 3915–3924.
External Links: ISSN 2640-3498
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§2.2](https://arxiv.org/html/2602.17133v1#S2.SS2.p1.1).
- J. Deng, W. Dong, R. Socher, L. Li, K. Li, and L. Fei-Fei (2009)
ImageNet: A large-scale hierarchical image database.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 248–255.
External Links: ISSN 1063-6919,
[Document](https://dx.doi.org/10.1109/CVPR.2009.5206848)
Cited by: [§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p1.3).
- P. Esser, R. Rombach, and B. Ommer (2021)
Taming transformers for high-resolution image synthesis.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 12873–12883.
Cited by: [§A.2](https://arxiv.org/html/2602.17133v1#A1.SS2.p1.2),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p1.3).
- C. Fifty, R. G. Junkins, D. Duan, A. Iyengar, J. W. Liu, E. Amid, S. Thrun, and C. Re (2024)
Restructuring vector quantization with the rotation trick.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- W. K. Hastings (1970)
Monte carlo sampling methods using markov chains and their applications.
Cited by: [§3.2.2](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS2.p2.4).
- K. Hsu, W. Dorrell, J. Whittington, J. Wu, and C. Finn (2023)
Disentanglement via latent quantization.
Advances in Neural Information Processing Systems 36, pp. 45463–45488.
Cited by: [item 2](https://arxiv.org/html/2602.17133v1#S1.I1.i2.p1.1).
- M. Huh, B. Cheung, P. Agrawal, and P. Isola (2023)
Straightening out the straight-through estimator: Overcoming optimization challenges in vector quantized networks.
In Proceedings of the International Conference on Machine Learning (ICML),
pp. 14096–14113.
External Links: ISSN 2640-3498
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- T. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick (2014)
Microsoft COCO: Common objects in context.
In Proceedings of European Conference on Computer Vision (ECCV), D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars (Eds.),
Cham, pp. 740–755.
External Links: [Document](https://dx.doi.org/10.1007/978-3-319-10602-1%5F48),
ISBN 978-3-319-10602-1
Cited by: [§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p1.3).
- S. Lloyd (1982)
Least squares quantization in pcm.
IEEE transactions on information theory 28 (2), pp. 129–137.
Cited by: [§3.3](https://arxiv.org/html/2602.17133v1#S3.SS3.SSS0.Px2.p1.2).
- J. Max (1960)
Quantizing for minimum distortion.
IRE Transactions on Information Theory 6 (1), pp. 7–12.
Cited by: [§3.3](https://arxiv.org/html/2602.17133v1#S3.SS3.SSS0.Px2.p1.2).
- F. Mentzer, D. Minnen, E. Agustsson, and M. Tschannen (2023)
Finite scalar quantization: VQ-VAE made simple.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 3](https://arxiv.org/html/2602.17133v1#A1.I1.i3.p1.2),
[item 2](https://arxiv.org/html/2602.17133v1#S1.I1.i2.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p6.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p1.1),
[§2.2](https://arxiv.org/html/2602.17133v1#S2.SS2.p2.1),
[§3.3](https://arxiv.org/html/2602.17133v1#S3.SS3.SSS0.Px2.p1.3).
- N. Metropolis, A. W. Rosenbluth, M. N. Rosenbluth, A. H. Teller, and E. Teller (1953)
Equation of state calculations by fast computing machines.
The journal of chemical physics 21 (6), pp. 1087–1092.
Cited by: [§3.2.2](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS2.p2.4).
- Mozilla (2024)
Mozilla common voice.
Note: https://commonvoice.mozilla.org/
Cited by: [§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p2.1).
- V. Panayotov, G. Chen, D. Povey, and S. Khudanpur (2015)
Librispeech: An ASR corpus based on public domain audio books.
In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pp. 5206–5210.
Cited by: [§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p2.1).
- J. D. Parker, A. Smirnov, J. Pons, C. J. Carr, Z. Zukowski, Z. Evans, and X. Liu (2024)
Scaling transformers for low-bitrate high-quality speech coding.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 3](https://arxiv.org/html/2602.17133v1#A1.I1.i3.p1.2),
[§1](https://arxiv.org/html/2602.17133v1#S1.p6.1),
[§4.2](https://arxiv.org/html/2602.17133v1#S4.SS2.p1.1).
- A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever (2023)
Robust speech recognition via large-scale weak supervision.
In Proceedings of the International Conference on Machine Learning (ICML),
pp. 28492–28518.
Cited by: [§A.2](https://arxiv.org/html/2602.17133v1#A1.SS2.p1.2).
- A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever (2021)
Zero-shot text-to-image generation.
In Proceedings of the International Conference on Machine Learning (ICML),
pp. 8821–8831.
External Links: ISSN 2640-3498
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- A. Razavi, A. van den Oord, and O. Vinyals (2019)
Generating diverse high-fidelity images with VQ-VAE-2.
In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS),
Vol. 32.
Cited by: [§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p2.1).
- A.W. Rix, J.G. Beerends, M.P. Hollier, and A.P. Hekstra (2001)
Perceptual evaluation of speech quality (PESQ)-a new method for speech quality assessment of telephone networks and codecs.
In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
Cited by: [§4.1.2](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS2.p1.1).
- C. P. Robert, G. Casella, and G. Casella (1999)
Monte carlo statistical methods.
Vol. 2, Springer.
Cited by: [§3.2.2](https://arxiv.org/html/2602.17133v1#S3.SS2.SSS2.p2.4).
- R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022)
High-resolution image synthesis with latent diffusion models.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 10684–10695.
Cited by: [§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- W. Shin, G. Lee, J. Lee, E. Lyou, J. Lee, and E. Choi (2023)
Exploration into translation-equivariant image quantization.
In Proceedings of IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pp. 1–5.
External Links: ISSN 2379-190X,
[Document](https://dx.doi.org/10.1109/ICASSP49357.2023.10096052)
Cited by: [§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p3.1).
- B. W. Silverman (1986)
Density estimation for statistics and data analysis.
CRC Press.
Cited by: [§3.1](https://arxiv.org/html/2602.17133v1#S3.SS1.p1.8).
- C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen (2010)
A short-time objective intelligibility measure for time-frequency weighted noisy speech.
In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pp. 4214–4217.
Cited by: [§4.1.2](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS2.p1.1).
- Y. Takida, T. Shibuya, W. Liao, C. Lai, J. Ohmura, T. Uesaka, N. Murata, S. Takahashi, T. Kumakura, and Y. Mitsufuji (2022)
SQ-VAE: Variational bayes on discrete representation with self-annealed stochastic quantization.
In Proceedings of the International Conference on Machine Learning (ICML),
pp. 20987–21012.
External Links: ISSN 2640-3498
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p2.1).
- A. Van Den Oord, O. Vinyals, et al. (2017)
Neural discrete representation learning.
In Proceedings of the Conference on Neural Information Processing Systems (NeurIPS),
NIPS’17, Red Hook, NY, USA, pp. 6309–6318.
External Links: ISBN 978-1-5108-6096-4
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#A1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p1.1),
[§4.2](https://arxiv.org/html/2602.17133v1#S4.SS2.p1.1).
- Y. Wang, Z. Lin, Y. Teng, Y. Zhu, S. Ren, J. Feng, and X. Liu (2024)
Bridging continuous and discrete tokens for autoregressive visual generation.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
Cited by: [item 4](https://arxiv.org/html/2602.17133v1#A1.I1.i4.p1.1),
[item 2](https://arxiv.org/html/2602.17133v1#S1.I1.i2.p1.1),
[§2.2](https://arxiv.org/html/2602.17133v1#S2.SS2.p3.1),
[§4.2](https://arxiv.org/html/2602.17133v1#S4.SS2.p1.1).
- Z. Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli (2004)
Image quality assessment: From error visibility to structural similarity.
IEEE Transactions on Image Processing 13 (4), pp. 600–612.
External Links: ISSN 1941-0042,
[Document](https://dx.doi.org/10.1109/TIP.2003.819861)
Cited by: [§4.1.2](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS2.p1.1).
- J. Yu, X. Li, J. Y. Koh, H. Zhang, R. Pang, J. Qin, A. Ku, Y. Xu, J. Baldridge, and Y. Wu (2021)
Vector-quantized image modeling with improved VQGAN.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, A. Gupta, X. Gu, A. G. Hauptmann, B. Gong, M. Yang, I. Essa, D. A. Ross, and L. Jiang (2024)
Language model beats diffusion - tokenizer is key to visual generation.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 2](https://arxiv.org/html/2602.17133v1#S1.I1.i2.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§2.2](https://arxiv.org/html/2602.17133v1#S2.SS2.p2.1).
- N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi (2021)
SoundStream: An End-to-End Neural Audio Codec.
IEEE/ACM Transactions on Audio, Speech, and Language Processing 30, pp. 495–507.
External Links: ISSN 2329-9290,
[Document](https://dx.doi.org/10.1109/TASLP.2021.3129994)
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1).
- L. Zhai, H. Ding, C. Zhao, f. wang, G. Wang, W. Zhi, and W. Xi (2025)
One quantizer is enough: Toward a lightweight audio codec.
arXiv.
External Links: 2504.04949,
[Document](https://dx.doi.org/10.48550/arXiv.2504.04949)
Cited by: [§A.2](https://arxiv.org/html/2602.17133v1#A1.SS2.p1.2),
[§4.1.1](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS1.p2.1).
- J. Zhang, F. Zhan, C. Theobalt, and S. Lu (2023)
Regularized vector quantization for tokenized image synthesis.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 18467–18476.
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1).
- R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang (2018)
The unreasonable effectiveness of deep features as a perceptual metric.
In Proceedings of IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 586–595.
Cited by: [§A.2](https://arxiv.org/html/2602.17133v1#A1.SS2.p1.2),
[§4.1.2](https://arxiv.org/html/2602.17133v1#S4.SS1.SSS2.p1.1).
- Y. Zhao, Y. Xiong, and P. Kraehenbuehl (2024)
Image and video tokenization with binary spherical quantization.
In Proceedings of the International Conference on Learning Representations (ICLR),
Cited by: [item 2](https://arxiv.org/html/2602.17133v1#S1.I1.i2.p1.1).
- C. Zheng and A. Vedaldi (2023)
Online clustered codebook.
In Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 22798–22807.
Cited by: [item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p2.1).
- Y. Zhu, B. Li, Y. Xin, Z. Xia, and L. Xu (2025)
Addressing representation collapse in vector quantized models with one linear layer.
In Proceedings of IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 22968–22977.
Cited by: [item 2](https://arxiv.org/html/2602.17133v1#A1.I1.i2.p1.1),
[item 1](https://arxiv.org/html/2602.17133v1#S1.I1.i1.p1.1),
[§1](https://arxiv.org/html/2602.17133v1#S1.p1.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p2.1),
[§2.1](https://arxiv.org/html/2602.17133v1#S2.SS1.p3.1),
[§4.2](https://arxiv.org/html/2602.17133v1#S4.SS2.p1.1).

## Appendix A Experimental Details

### A.1 Representative Baselines

We evaluate our approaches against four representative baselines:

- 1.
VQ-VAE (Van Den Oord et al., 2017): The foundational coupled quantization baseline trained with the Straight-Through Estimator (STE).
- 2.
SimVQ (Zhu et al., 2025): A state-of-the-art coupled method that reparameterizes the codebook via a learnable linear layer to improve gradient flow.
- 3.
FSQ (Mentzer et al., 2023): A fixed scalar quantizer.
We utilize the symmetric variant with noise injection (Parker et al., 2024; Brendel et al., 2024b) as a stronger baseline.
The dimensionality $d$ and the number of quantization levels per dimension are set according to the recommendations in (Mentzer et al., 2023) to match the target codebook sizes $K$:
•
$K=256$: Levels $[8,6,5]$ ($d=3$).
•
$K=1024$: Levels $[8,5,5,5]$ ($d=4$).
•
$K=4096$: Levels $[7,5,5,5,5]$ ($d=5$).
•
$K=16384$: Levels $[8,8,8,6,5]$ ($d=5$).
- 4.
TokenBridge (Wang et al., 2024): A method that discretizes the latent space using fixed Gaussian percentile grids.
Following the official implementation, we first train a KL-regularized VAE to enforce a standard normal prior on the latent space.
Subsequently, we discretize the continuous representations using a fixed grid derived from the percentiles of the Gaussian distribution (4 bins per dimension).

For FSP implementation, we employ the exact same dimensionality $d$ and level configurations ($[L_{1},\dots,L_{d}]$) as FSQ for each target codebook size.
We utilize the Tanh-based bounded activation, consistent with FSQ.
For VP-VAE, we also set the bottleneck dimension $d$ (cf. Eq. [1](https://arxiv.org/html/2602.17133v1#S3.E1)) to match the dimensionality of the corresponding FSQ/FSP configuration (e.g., $d=3$ for $K=256$). This ensures that the density estimation complexity and the representational dimension remain consistent across baselines.

### A.2 Training and Optimization

All models are trained for 50 epochs using the same optimizer with the same hyperparameter settings as in the original paper (Esser et al., 2021; Zhai et al., 2025).
For images, we minimize a combination of $L_{1}$ loss and LPIPS perceptual loss (Zhang et al., 2018) (VGG backbone).
For audio, we use an $L_{1}$ reconstruction loss in the time domain, a multi-resolution STFT loss, and a perceptual loss based on intermediate features from a frozen Whisper encoder (Radford et al., 2023).
All experiments are conducted on 2 NVIDIA RTX4090 GPUs.

## Appendix B Additional Experiments

### B.1 Ablation Study

We conduct ablation studies to validate the key design choices in VP-VAE and FSP.
Unless otherwise specified, experiments are conducted on the image modality with a target codebook size of $K{=}1024$.

#### B.1.1 VP-VAE Ablation

**Table 3: Ablation on VP-VAE components. We evaluate the contribution of latent normalization and Metropolis–Hastings mechanism on image reconstruction ($K{=}1024$). Both components contribute to reconstruction quality and codebook balance.**
| Setting | CVU | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- |
| VP-VAE (full) | 0.8102 | 0.1717 | 23.8878 | 0.7182 |
| w/o $\mathcal{L}_{\text{norm}}$ | 0.8052 | 0.1850 | 24.0219 | 0.7167 |
| w/o MH (always accept) | 0.7467 | 0.1756 | 23.8053 | 0.7138 |

##### Effect of latent normalization.

The normalization regularizer (Eq. [11](https://arxiv.org/html/2602.17133v1#S3.E11)) is important for the stability of our scale estimation mechanism.
Since the kNN-based radius estimation (Eq. [3](https://arxiv.org/html/2602.17133v1#S3.E3)) relies on Euclidean distance, it is sensitive to the relative scaling of latent dimensions.
Without normalization, dimensions with naturally larger variances can dominate the distance calculation, making the kNN-based radius estimation in Eq. ([3](https://arxiv.org/html/2602.17133v1#S3.E3)) less reliable and consequently weakening the scale alignment of perturbations.
As shown in [Table 3](https://arxiv.org/html/2602.17133v1#A2.T3), removing $\mathcal{L}_{\text{norm}}$ degrades perceptual reconstruction quality (LPIPS) and slightly reduces utilization balance.

##### Effect of Metropolis–Hastings mechanism.

We compare our full MH procedure against a simplified variant that always accepts proposals (equivalent to injecting uniform noise within the estimated radius).
The MH step acts as a filter for *distribution consistency*: it rejects perturbations that would push the latent vector into low-density regions or out-of-distribution areas.
Without MH, the decoder is forced to reconstruct from “invalid latent states”, effectively wasting model capacity.
[Table 3](https://arxiv.org/html/2602.17133v1#A2.T3) confirms that removing the acceptance criterion reduces both CVU (0.81 $\to$ 0.75) and reconstruction quality.

#### B.1.2 FSP Ablation

**Table 4: FSP activation functions on images. We compare three CDF-like activations, which affect how well the “approximately uniform” assumption holds and thus the effectiveness of FSP ($K{=}1024$).**
| Setting | CVU | LPIPS ↓ | PSNR ↑ | SSIM ↑ |
| --- | --- | --- | --- | --- |
| FSP (Tanh) | 0.8515 | 0.1722 | 24.1910 | 0.7177 |
| FSP (Normal CDF) | 0.8663 | 0.1689 | 24.2945 | 0.7254 |
| FSP (Laplace CDF) | 0.8390 | 0.1685 | 24.2433 | 0.7263 |

**Table 5: FSP activation functions on audio. We evaluate three CDF-like activations on audio ($K{=}1024$).**
| Setting | CVU | PESQ ↑ | STOI ↑ |
| --- | --- | --- | --- |
| FSP (Tanh) | 0.0810 | 2.4458 | 0.9191 |
| FSP (Normal CDF) | 0.0654 | 2.3772 | 0.9154 |
| FSP (Laplace CDF) | 0.0751 | 2.3760 | 0.9198 |

##### Choice of FSP activation.

FSP assumes the latent variables can be mapped to an approximate uniform distribution via an activation function $g(\cdot)$.
We compare three choices: Tanh (rescaled), Normal CDF, and Laplace CDF.
As shown in [Table 4](https://arxiv.org/html/2602.17133v1#A2.T4), different activations yield modest trade-offs across metrics.
We also evaluate these activations on the audio modality ([Table 5](https://arxiv.org/html/2602.17133v1#A2.T5)) and observe similar patterns: no single activation dominates across all metrics, suggesting that the choice can be tuned per application.

### B.2 Qualitative Analysis

Figure: Figure 1: Codebook utilization during training. CVU curves for different methods on image reconstruction ($K{=}1024$). VQ-VAE and FSQ exhibit an initial rise followed by a decline. VP-VAE and FSP maintain stable, high utilization throughout training.
Refer to caption: x1.png

##### Training dynamics of codebook utilization.

[Figure 1](https://arxiv.org/html/2602.17133v1#A2.F1) visualizes CVU over training epochs for different methods at $K{=}1024$.
We observe two distinct behaviors regarding stability.
First, VQ-VAE and FSQ exhibit an initial rise followed by a decline: utilization increases initially as the encoder explores the space, but subsequently degrades.
This decline reflects certain codes become increasingly dominant, triggering the self-reinforcing vicious cycle where rarely used codes receive fewer gradient updates and become even less likely to be chosen.(The trend of FSQ is slower than VQ-VAE due to the fixed codebook.)
In contrast, VP-VAE and FSP maintain stable CVU throughout training, demonstrating the stability benefits of decoupled training.

Second, we analyze the relationship between utilization and quality.
TokenBridge achieves near-perfect utilization (CVU $\approx 0.99$) yet yields significantly lower reconstruction quality (e.g., LPIPS $\approx 0.6$).
This phenomenon can be explained by the trade-off between capacity and granularity.
The codebook size determines the discrete representation capacity and directly controls the granularity of quantization: larger codebooks yield finer partitions, while smaller codebooks impose coarser discretization.
TokenBridge is originally designed for massive codebooks ($K\geq 2^{48}$) where a fixed grid provides sufficient resolution.
At the moderate codebook sizes evaluated here ($K\leq 16384$), the rigid Gaussian percentile grid is too coarse to capture fine-grained data details, regardless of how uniformly the bins are used.
This underscores that high CVU alone is insufficient; effective discretization requires that the induced quantization error be compatible with the decoder’s robustness.

Consistent with the image results in [Table 1](https://arxiv.org/html/2602.17133v1#S3.T1), only SimVQ, VP-VAE, and FSP achieve a favorable balance of both high utilization and high fidelity in the image domain.

Figure: Figure 2: Output distributions of fixed quantization schemes. Given a uniform latent distribution, we compare the quantized output distributions produced by FSQ, FSQ with noise, symmetric FSQ with noise, and FSP. They are all configured with $L{=}4$ quantization levels. FSP produces a more uniform output distribution, aligning with the Lloyd–Max optimality principle.
Refer to caption: x2.png

##### Comparison of FSP and FSQ quantization schemes.

[Figure 2](https://arxiv.org/html/2602.17133v1#A2.F2) provides an intuitive comparison of how different fixed quantization schemes transform a uniform latent distribution.
With $L{=}4$ quantization levels, standard FSQ rounds to grid boundaries, producing a non-uniform output distribution skewed toward the boundaries.
In contrast, FSP quantizes to interval centroids, which are precisely the Lloyd–Max optimal reconstruction points for a uniform source.
This produces an output distribution that remains approximately uniform, explaining FSP’s consistent advantage over FSQ variants in our experiments.

## Appendix C Limitations and Future Work

While VP-VAE offers stable and high-fidelity training, it relies on non-parametric density estimation via $k$-nearest-neighbor search, which introduces additional computational overhead during training, particularly as the size of the memory queue $\mathcal{S}$ grows.
We mitigate this cost through random subsampling and by applying perturbations within a low-dimensional bottleneck.
We envision that further acceleration techniques (e.g., approximate nearest-neighbor search) could be beneficial for scaling to very large models.
Additionally, our current experiments focus on autoencoder-based tokenization; integrating VP-VAE with autoregressive or diffusion-based token generators remains an interesting direction.