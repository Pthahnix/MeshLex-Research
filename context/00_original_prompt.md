<!-- markdownlint-disable -->
## Research Task Description: Large Mesh Model (LMM)

### Overview
I am proposing a research project centered on a large-scale generative model for 3D mesh 
representation, referred to as the **Large Mesh Model (LMM)**. The core motivation is to 
establish a new unified paradigm that subsumes both 3D reconstruction and 3D generation 
within a single large model framework, analogous to how large language models and large 
vision-language models have unified diverse tasks in their respective domains.

### Target Publication Venue
CCF-A ranked conferences in computer vision, computer graphics, or artificial intelligence 
(e.g., CVPR, ICCV, ECCV, SIGGRAPH / SIGGRAPH Asia, NeurIPS, ICML, AAAI, or equivalent).

---

### Model Architecture Scope

The model is expected to operate at large-model scale (in terms of parameters and training 
data). The architecture space of interest includes, but is not limited to:

- **Pure Transformer-based** autoregressive or non-autoregressive generative architectures 
  for 3D mesh tokens
- **Transformer + Diffusion hybrid**: e.g., using Transformer for high-level semantic 
  reasoning or structure planning, and Diffusion models for geometry detail synthesis or 
  iterative refinement. Reference context: NVIDIA has previously described concepts where 
  Transformer handles generation/planning and Diffusion handles "thinking"/refinement — 
  this hybrid paradigm is of particular interest.
- **Transformer + Mamba hybrid**: leveraging state-space models (SSM/Mamba) for efficient 
  long-sequence modeling of mesh tokens or geometry sequences
- Any other emerging hybrid architectures that combine the scalability of Transformers with 
  the inductive biases suitable for 3D structural data

The 3D representation target is **explicit mesh** (vertices, faces, topology), not implicit 
representations such as NeRF, SDF, or Gaussian Splatting — though conversion or auxiliary 
use of such representations during intermediate stages may be considered.

---

### Input Modalities and Task Coverage

LMM should support a spectrum of conditional inputs, unifying the following tasks:

1. **Single-view image → 3D mesh** (image-conditioned reconstruction)
2. **Multi-view images → 3D mesh** (multi-view reconstruction)
3. **Text description → 3D mesh** (text-conditioned generation)
4. **Image + Text → 3D mesh** (text-driven reconstruction or stylized/guided reconstruction-
   generation, e.g., "reconstruct from this image but make it more futuristic")
5. Optionally: unconditional or class-conditioned generation

The unification of reconstruction (where a ground-truth object is implied) and generation 
(where creativity and diversity are valued) under a single model is the central novel claim 
of the paradigm.

---

### Relevant Research Areas for Reference

The following domains are provided as broad reference contexts. The goal is not to limit 
the research to 3D-adjacent work, but to encourage cross-domain synthesis and inspiration 
from any field where structurally similar problems, solutions, or paradigms may exist.

#### 3D Geometry & Reconstruction (Foundational Context)
- Autoregressive and diffusion-based 3D mesh generation (PolyGen, MeshGPT, MeshAnything, etc.)
- Feed-forward large reconstruction models (LRM family, InstantMesh, Zero123, etc.)
- 3D tokenization strategies for Transformer consumption

#### Large Language Models & Generative Paradigms
- Autoregressive generation at scale: how LLMs unify diverse tasks under a single 
  next-token prediction objective — is there a structural analog for 3D mesh tokens?
- Instruction following and multi-task unification in LLMs (T5, GPT-4, LLaMA, etc.)
- In-context learning, few-shot generalization: can similar mechanisms emerge for 
  geometric conditioning?
- Chain-of-thought and "thinking before generating": slow vs. fast generation, 
  reasoning-then-synthesis pipelines — parallels to the Transformer-plans + 
  Diffusion-refines hybrid concept
- Speculative decoding and token prediction hierarchies

#### Vision-Language Models & Multi-modal Alignment
- CLIP, BLIP, LLaVA, Flamingo: how cross-modal grounding is achieved
- Unified multi-modal models that handle image, text, video under one architecture
- Modality-agnostic tokenization strategies (e.g., treating images, text, and geometry 
  as sequences in a shared token space)

#### Agentic and Planning Architectures
- LLM-based agents with tool use, world models, and hierarchical planning
- Model-based RL and world models (Dreamer, RSSM): treating 3D structure as a "world 
  state" to be imagined and refined
- Hierarchical generation: coarse-to-fine planning as an architectural principle 
  (analogies to how agents decompose tasks into sub-goals)
- Monte Carlo Tree Search and search-based generation: using structured search over 
  a generative space

#### Hybrid and Compositional Architectures
- Diffusion + Transformer (DiT, MAR, etc.): scalable generative architectures beyond 
  pure autoregression
- Flow Matching as a unifying framework between discrete and continuous generation
- Mamba / SSM: efficient long-sequence modeling, particularly relevant for high-resolution 
  or high-polygon mesh sequences
- Mixture-of-Experts (MoE): sparse activation for scaling generalist models
- Discrete vs. continuous latent spaces: VQ-VAE, VQGAN, and their role in bridging 
  perception and generation

#### Neuroscience & Cognitive Science Analogies
- Dual-process theory (System 1 / System 2 thinking): fast intuitive generation vs. 
  slow deliberative refinement — direct conceptual parallel to the 
  Transformer-generation + Diffusion-thinking hybrid
- Predictive coding and hierarchical inference in the brain as a model for 
  generative reconstruction
- Mental rotation and 3D spatial reasoning in humans: what inductive biases might 
  a model need to replicate this?

#### Physics & Simulation
- Physics-informed neural networks (PINNs): incorporating structural or physical 
  constraints into generative models — meshes must be manifold, watertight, etc.
- Symmetry and equivariance in geometric deep learning (E(3)-equivariant networks, 
  SE(3) transformers): how physical symmetries constrain or regularize 3D generation
- Statistical mechanics and energy-based models: diffusion models as discretized 
  Langevin dynamics, connections to thermodynamic annealing

#### Mathematics & Topology
- Discrete differential geometry: the mathematical foundations of mesh representation, 
  curvature, Laplacian operators — may inform loss design or tokenization
- Topological data analysis (TDA): persistent homology as a tool for evaluating or 
  constraining generated mesh topology
- Information theory: rate-distortion tradeoffs in compressed 3D representations, 
  mutual information between conditioning signals and generated geometry
- Optimal transport: Wasserstein distances and their use in generative model training 
  and evaluation for structured outputs

#### Compression & Coding Theory
- Traditional mesh compression (Draco, progressive mesh coding): how geometry is 
  efficiently encoded into sequences — may inspire tokenization design
- Neural compression and the relationship between generative models and compressors 
  (bits-back coding, etc.)
- The "generation as decompression" viewpoint: a generative model as a learned 
  prior over a compressed latent code

#### Scaling Laws & Emergence
- Empirical scaling laws in LLMs (Chinchilla, Kaplan et al.): do similar laws hold 
  for 3D generative models? What are the compute-optimal training recipes?
- Emergent capabilities at scale: which reconstruction or generation behaviors might 
  only appear above certain model/data thresholds?
- Data scaling strategies: synthetic data generation, web-scale 3D data curation, 
  and their role in large model training

---

### Research Planning Request

Based on the above description, please help me develop a comprehensive and structured 
research plan for LMM. This should include (but is not limited to): problem formulation, 
related work analysis, proposed methodology, architecture design rationale, training 
strategy, dataset requirements, evaluation protocol, ablation study design, novelty 
positioning, and a realistic timeline targeting a CCF-A submission. Please identify 
potential technical risks and open questions as well.
