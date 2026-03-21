## Contents
- 1 Introduction
- 2 Related works
  - 2.1 Image-to-3D Generation
  - 2.2 Flexible Diffusion Models
- 3 Preliminary
  - 3.1 Ordered mesh representation
  - 3.2 Rectified flow
- 4 Methodology
  - 4.1 Encoding meshes into face-level continuous tokens
  - 4.2 Mesh generation with the flow-based DiT
- 5 Experiments
  - 5.1 Experiment Settings
    - 5.1.1 Datasets
    - 5.1.2 Evaluation
    - 5.1.3 Implementation Details
  - 5.2 Experiment Results
    - 5.2.1 Results on ShapeNet dataset
    - 5.2.2 Results on Objaverse dataset
  - 5.3 Ablation Studies
    - 5.3.1 Comparisons of auto-encoder selections
    - 5.3.2 Effects of CFG weights
  - 5.4 Limitations
- 6 Conclusion
- References
- 7 Additional details of the model design
  - 7.1 Detailed architectures of modules
  - 7.2 Logit-normal sampling
  - 7.3 The importance of QK-norm
- 8 Additional quantitative results

## Abstract

Abstract In the domain of 3D content creation, achieving optimal mesh topology through AI models has long been a pursuit for 3D artists.
Previous methods, such as MeshGPT, have explored the generation of ready-to-use 3D objects via mesh auto-regressive techniques.
While these methods produce visually impressive results, their reliance on token-by-token predictions in the auto-regressive process leads to several significant limitations. These include extremely slow generation speeds and an uncontrollable number of mesh faces.
In this paper, we introduce MeshCraft, a novel framework for efficient and controllable mesh generation, which leverages continuous spatial diffusion to generate discrete triangle faces.
Specifically, MeshCraft consists of two core components: 1) a transformer-based VAE that encodes raw meshes into continuous face-level tokens and decodes them back to the original meshes, and 2) a flow-based diffusion transformer conditioned on the number of faces, enabling the generation of high-quality 3D meshes with a predefined number of faces.
By utilizing the diffusion model for the simultaneous generation of the entire mesh topology, MeshCraft achieves high-fidelity mesh generation at significantly faster speeds compared to auto-regressive methods. Specifically, MeshCraft can generate an 800-face mesh in just 3.2 seconds—35 × \times × faster than existing baselines. Extensive experiments demonstrate that MeshCraft outperforms state-of-the-art techniques in both qualitative and quantitative evaluations on ShapeNet dataset and demonstrates superior performance on Objaverse dataset. Moreover, it integrates seamlessly with existing conditional guidance strategies, showcasing its potential to relieve artists from the time-consuming manual work involved in mesh creation.

## 1 Introduction

With advances in fields such as gaming and 3D printing, the creation of high-quality, topologically sound 3D meshes has become increasingly important. However, generating well-structured 3D meshes from simple inputs—such as textual descriptions or a single 2D image—requires significant artistic expertise and often entails labor-intensive manual processes. Therefore, the ability to rapidly generate 3D meshes with high-quality topology that meet the needs of artists and professionals using AI models is a key goal.

Significant efforts have been made to automate the generation of 3D meshes. Many works [30, 39] are modeling with dense triangle meshes, which are extracted from neural fields using iso-surfacing methods [25, 28, 16]. Specifically, these methods first generate intermediate 3D representations and then post-process them into final meshes. Although these approaches yield impressive visualizations in producing neural representations, the resulting meshes often suffer from excessive face counts and artifacts due to misalignment that occur during the conversion of intermediate representations into final meshes via re-meshing techniques [21]. The efficiency of these processes is also limited by the post-processing step.

To address these issues, another group of works [26, 32, 1, 11], modeling with triangle meshes that accurately reflect the compactness of artists’ design, offers great flexibility for manipulation and efficiency for storage. This approach, referred to as ”native mesh generation,” focuses on explicitly modeling mesh distributions. It first transforms the mesh into latent sequences by leveraging the spatial relationships among faces, vertices, and coordinates. These methods then directly generate the sequences, and even shows superior long sequence scalability. However, these approaches are often limited by generation effectiveness or are validated only on small datasets, such as ShapeNet [2]. Furthermore, most of the methods lack user controllability, which is essential for the 3D creation process.

To address these challenges, we introduce MeshCraft, an efficient and controllable approach for high-fidelity mesh generation method that leverages the strengths of continuous diffusion transformers with rectified flow (see [Fig. 1](https://arxiv.org/html/2503.23022v1#S0.F1)(b) and (c) for comparisons with prior works). Unlike auto-regressive methods such as Meshtron [11], which prioritize scalability, we explore the possibilities of different routes starting from the underlying principles of implementing native mesh generation, with the goal of demonstrating the potential of diffusion models for the practical usage of controllable and efficient mesh generation, rather than immediate scalability.

MeshCraft saves token numbers for at most 9 times, and speed up for 35 times. It is designed as a two-stage pipeline that consists of directly modeling the mesh distribution in a latent space via a Variational Auto-Encoder(VAE) [17] and then generating meshes with a diffusion transformer. Unlike previous works that compress meshes into discrete tokens, we explore the modeling of meshes in a low-dimensional and continuous latent space. Face features of the mesh are sent into the VAE encoder, regularized with KL divergence, and are subsequently decoded into vertex coordinates of each face. For the generative model, the diffusion transformer is modified to generate the continuous tokens with varying lengths with inspirations from recent advances in image generation [9, 22].
This approach enables MeshCraft to generate meshes at high speed. The guidance and masking strategy also allow users to control the number of faces of generated meshes in a user-friendly manner.

We validate the effectiveness of MeshCraft on the ShapeNet dataset [2] and demonstrate its great potential as a generic mesh generator on Objaverse [7], which includes a diverse array of 3D objects across various categories. Extensive experiments illustrate the effectiveness, and controllability of MeshCraft from both qualitative and quantitative aspects.
In summary, our contributions are as follows:

- •
We propose a transformer-based VAE that encodes discrete triangle meshes into a continuous latent space and decodes them back to the original mesh, achieving competitive reconstruction performance compared to state-of-the-art mesh auto-encoders based on vector quantization.
- •
We introduce a flow-based transformer diffusion model conditioned on the number of faces, integrating classifier-free guidance for both image inputs and face number to enable effective control over the mesh generation process.
- •
MeshCraft significantly outperforms existing methods, achieving new state-of-the-art results on the ShapeNet dataset while being 35$\times$ faster than MeshGPT, and demonstrating the effectiveness of face number control on the large-scale Objaverse dataset.

## 2 Related works

Recent advances [26, 1, 32, 38, 4, 11] have seen pioneering efforts to generate meshes directly, with many employing auto-regressive models for this task. For instance, PolyGen [26] utilizes two transformers to separately learn vertex and face distributions. MeshGPT [32] first encodes meshes into face-level quantized tokens via GNN-based Vector Quantization Variational Auto-Encoder(VQ-VAE) [37], followed by the application of GPT-style transformers for auto-regressive generation. MeshXL [4] introduces another sequential mesh representation for one-stage auto-regressive generation. Even though Meshtron [11] shows the scalability of auto-regressive generations, however, the generative capabilities of these models are restricted by the rapidly increasing number of tokens, leading to slow inference speeds. Additionally, users cannot precisely assign an exact number of object faces, which limits their practicality in the existing workflow for 3D creations.
Among these works, Polydiff [1] is the most relevant to our approach. It trains a class-conditioned discrete diffusion model using discrete state transition matrices and cross-entropy loss, which is challenging to optimize and hard to benefit from existing conditional guidance method. In contrast, we decouple modeling and generation: first compressing meshes into a semantically rich latent space with a KL-regularized VAE, then applying a flow-based DiT to model this space for mesh generation. MeshCraft is the first to introduce a fine-grained controllable mesh generator based on diffusion transformers.

### 2.1 Image-to-3D Generation

With the effective exploration of 3D representations [25, 28, 16], large-scale datasets [7], and popular 2D generative models [31], conditional generation of diverse and high-fidelity 3D assets has emerged as a promising area of research. Most studies[30, 35, 20, 12] first learn neural 3D representations and then post-process [21, 5] them into meshes, which can lead to overly dense results. Some approaches [30, 18, 39] utilize pre-trained text-to-image models to optimize targeted meshes based on given conditions, resulting in significant time costs. In this paper, we focus on ”directly” generating high-fidelity meshes in face-level representations, effectively balancing quality and efficiency.

### 2.2 Flexible Diffusion Models

The rise of text-to-image diffusion models has prompted the consideration of more customized demands, particularly in generating images with unrestricted resolutions. Recent works [22, 42, 3] have explored this area with flexible training, inference strategies, and model designs. Inspired by these advancements, we propose MeshCraft, a controllable mesh generation pipeline based on flow-based diffusion transformers. Our user-friendly method enables the rapid generation of compact meshes while allowing for extensive manipulation of the results, thereby facilitating practical applications of AI-generated 3D assets in the industry.

## 3 Preliminary

### 3.1 Ordered mesh representation

In this work, we adopt the ordering rule used in previous native mesh generation studies [26, 32, 38] and consider meshes as ordered sequences consisting of three progressively defined components: face level, vertex level, and coordinate level. Let $\mathcal{M}$ be a mesh that includes $n$ faces $\{\mathbf{f}_{i}\}_{i=1,2,\cdots,n}$, where each face $\mathbf{f}_{i}$ comprises $k$ vertices, represented as $\mathbf{f}_{i}=\{\mathbf{v}_{i}^{1},\mathbf{v}_{i}^{2},\cdots,\mathbf{v}_{i}^{
k}\}$. Each vertex is defined in the coordinate system as $\mathbf{v}_{i}^{j}=(x_{i}^{j},y_{i}^{j},z_{i}^{j})$. Consequently, the mesh $\mathcal{M}$ can be represented in the following manner (taking $k=3$ as an example):

$$ $\begin{split}\mathcal{M}&=\{\mathbf{f}_{1},\mathbf{f}_{2},\cdots,\mathbf{f}_{n }\}\\ &=\{\mathbf{v}_{1}^{1},\mathbf{v}_{1}^{2},\mathbf{v}_{1}^{3},\mathbf{v}_{2}^{1 },\mathbf{v}_{2}^{2},\mathbf{v}_{2}^{3},\cdots,\mathbf{v}_{n}^{1},\mathbf{v}_{ n}^{2},\mathbf{v}_{n}^{3}\}\\ &=\{x_{1}^{1},y_{1}^{1},z_{1}^{1},x_{1}^{2},y_{1}^{2},z_{1}^{2},\cdots,x_{n}^{ 3},y_{n}^{3},z_{n}^{3}\}\end{split}$ (1) $$

For sequence ordering, faces are sorted by vertex indices from lowest to highest, while vertices are sorted by their z-y-x coordinates in the same manner to ensure the uniqueness of the mesh. Unlike prior works using representations at the vertex or coordinate levels to generate meshes, we generate face-level tokens using flow-based diffusion transformers. This approach significantly reduces the number of tokens and enhances the efficiency of mesh generation.

Figure: Figure 2: Pipeline of MeshCraft. Our framework comprises two stages. We firstly compress meshes into face-level tokens ([Sec. 4.1](https://arxiv.org/html/2503.23022v1#S4.SS1)). Then the tokens are used for training the flow-based DiT, which is guided by the input face number and the image conditions ([Sec. 4.2](https://arxiv.org/html/2503.23022v1#S4.SS2)).
Refer to caption: x2.png

### 3.2 Rectified flow

With the growing popularity of generative models in recent years, diffusion models [31, 14, 33, 29, 24] have been widely recognized for their powerful modeling capabilities. Score-based models, such as [33] and DDPM [14], are commonly employed, formulating the diffusion process through stochastic differential equations (SDEs). However, these methods often suffer from a slow iterative de-noising process, resulting in inefficient inference time. In contrast, rectified flow [19] is an implicit probabilistic model designed for fast generation, based on ordinary differential equations (ODE). It aims to transport the distribution $\pi_{0}$ to $\pi_{1}$ by following straight-line paths as much as possible. This preference is both theoretically and computationally advantageous, allowing for few-step or even one-step sampling.

Given two distributions $\pi_{0}$ and $\pi_{1}$, the rectified flow induced from $(X_{0},X_{1})$, where $X_{0}\sim\pi_{0}$ and $X_{1}\sim\pi_{1}$, is modeled as an ODE over time $t\in[0,1]$,

$$ $\mathrm{d}Z_{t}=v(Z_{t},t)\mathrm{d}t$ (2) $$

which converts $Z_{0}$ from $\pi_{0}$ to a $Z_{1}$ following $\pi_{1}$. The drift force $v:\mathbb{R}^{d}\rightarrow\mathbb{R}^{d}$ is to drive the flow to follow the linear direction $X_{1}-X_{0}$ as much as possible by solving a simple least squares regression problem:

$$ $\min_{v}\int_{0}^{1}{\mathbb{E}\left[\|(X_{1}-X_{0})-v(X_{t},t)\|^{2}\right] \mathrm{d}t}$ (3) $$

where $X_{t}=tX_{1}+(1-t)X_{0}$ is the linear interpolation of $X_{0}$ and $X_{1}$. $v$ is parameterized by the models in practice.

Rectified flow not only avoids crossing paths when finding the solution but also reduces errors arising from discrete-time scheduling and transport costs. To leverage rectified flow and advanced model architectures, we follow the implementation in SiT [24] and introduce several techniques to adapt it to the task of mesh generation.

Figure: Figure 3: Reconstruction quality using different tokenizers. ”Continuous” means using KL-divergence loss to regularize continuous-space tokens, while ”Discrete” stands for using RVQ [40] to quantize discrete tokens for reconstruction. Refer to [Tab. 3](https://arxiv.org/html/2503.23022v1#S5.T3) for quantitative results.
Refer to caption: x3.png

## 4 Methodology

Existing works [26, 32, 38, 4] that generate artist-like meshes predominantly utilize auto-regressive models. However, these approaches require discretizing coordinates into a limited vocabulary, resulting in long token sequences for each mesh. This can lead to information loss and inefficient inference speeds.
In contrast, we propose generating latent tokens of faces in the continuous space through a transformer-based auto-encoder, and we model the distribution using flow-based DiTs. As [Fig. 3](https://arxiv.org/html/2503.23022v1#S3.F3) shows, the continuous tokenizer achieves higher reconstruction accuracy than the discrete one, justifying our choice of continuous diffusion over discrete AR, which is also a foundation motivation for the diffusion-based exploration.
Moreover, this approach facilitates fast and controllable mesh generation. Furthermore, it allows our method to integrate with popular guidance strategies [13], enabling control over the diffusion process with different conditions and enhancing generation quality. An overview of MeshCraft is illustrated in [Fig. 2](https://arxiv.org/html/2503.23022v1#S3.F2).

Figure: Figure 4: Qualitative comparisons on ShapeNet. MeshCraft produces high-quality meshes with sharp edges and smooth faces.
Refer to caption: x4.png

### 4.1 Encoding meshes into face-level continuous tokens

As described in [Sec. 3.1](https://arxiv.org/html/2503.23022v1#S3.SS1), we formulate meshes as ordered sequences according to [Eq. 1](https://arxiv.org/html/2503.23022v1#S3.E1). To effectively learn their distribution, the sequences are fed into encoder $E$ along with associated geometric information (normals, angles, areas, and adjacency among faces), which is aggregated using a single Graph Convolutional Network (GCN) layer, preserving geometric information and enhancing the representation robustness. Subsequently, $N_{E}$ transformer-style blocks are employed to extract face-level features $F_{i}$.

Unlike previous works [32, 38] that utilize residual vector quantization [40] to obtain discrete tokens, we linearly project the features into continuous space:

$$ $\begin{split}\mathrm{FC}_{\mu}(F_{i})=\left(\mu_{i,j}\right)_{j\in[1,2,\cdots, C_{KL}]}\\ \mathrm{FC}_{\sigma}(F_{i})=\left(\log\sigma^{2}_{i,j}\right)_{j\in[1,2,\cdots ,C_{KL}]}\end{split}$ (4) $$

where $\mathrm{FC}_{\mu}(\cdot)$ and $\mathrm{FC}_{\sigma}(\cdot)$ are linear projection layers. The continuous tokens $F_{i}^{\prime}$ can then be sampled from $(\mu_{i},\sigma_{i})$. To better adapt them for training the diffusion model, we use KL-divergence to regularize them:

$$ $\mathcal{L}_{KL}\left(\{F_{i}\}_{i=1}^{n}\right)=\frac{1}{n\cdot C_{KL}}\sum_{ i=1}^{n}\sum_{j=1}^{C_{KL}}\frac{1}{2}\left(\mu^{2}_{i,j}+\sigma^{2}_{i,j}- \log\sigma^{2}_{i,j}\right)$ (5) $$

As the findings of Fluid [10] claim, this transformation eliminates the need for the codebook, and also results in significantly shorter token lengths, leading to accurate reconstructions (also discussed in [Sec. 5.3.1](https://arxiv.org/html/2503.23022v1#S5.SS3.SSS1)).

The sampled tokens $F_{i}^{\prime}$ are then passed to the decoder $D$, which consists of $N_{D}$ transformer-style blocks and concludes with an MLP layer. The outputs are reshaped into coordinates to compute the loss $\mathcal{L}_{AE}$ against the input sequences. Following [32], we use a cross-entropy loss $\mathcal{L}_{AE}$ to guide the training process.

**Table 1: Reconstruction performance on ShapeNet dataset. Our continuous auto-encoder behaves competitively with prior works using the vector quantization.**
| Method | Tri. Accu.$(\%)\uparrow$ | L2 Dist.$(\times 10^{-2})\downarrow$ |
| --- | --- | --- |
| MeshGPT [32] | 99.99 | 0.00 |
| PivotMesh [38] | 98.88 | 0.86 |
| Ours | 99.42 | 0.06 |

**Table 2: Quantitative results on ShapeNet dataset. MeshCraft outperforms the baselines on shape quality, visual and compactness metrics. MMD values are multiplied by $10^{3}$. COV and 1-NNA are scaled by $10^{2}$. * stands for using the released pre-trained models.**
| Class | Method | COV$\uparrow$ | MMD$\downarrow$ | 1-NNA | JSD$\downarrow$ | FID$\downarrow$ | KID$\downarrow$ | Class | Method | COV$\uparrow$ | MMD$\downarrow$ | 1-NNA | JSD$\downarrow$ | FID$\downarrow$ | KID$\downarrow$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chair | MeshGPT [32] | 45.98 | 10.34 | 60.06 | 11.67 | 25.43 | 4.10 | Table | MeshGPT [32] | 48.85 | 9.23 | 57.82 | 8.50 | 21.98 | 2.99 |
| PivotMesh [38] | 47.99 | 10.00 | 60.06 | 13.51 | 34.40 | 10.33 | PivotMesh [38] | 47.42 | 9.08 | 58.35 | 10.42 | 24.97 | 7.99 |  |  |
| MeshXL* [4] | 49.43 | 10.17 | 56.90 | 11.37 | 20.09 | 1.70 | MeshXL* [4] | 50.98 | 9.38 | 57.82 | 9.07 | 22.08 | 2.88 |  |  |
| Ours | 51.44 | 9.61 | 54.31 | 11.03 | 20.40 | 1.76 | Ours | 55.42 | 8.74 | 54.26 | 8.73 | 16.63 | 1.70 |  |  |
| Bench | MeshGPT [32] | 56.06 | 8.44 | 58.33 | 28.34 | 66.30 | 9.45 | Lamp | MeshGPT [32] | 43.90 | 20.82 | 60.37 | 36.21 | 73.21 | 6.04 |
| PivotMesh [38] | 59.09 | 8.25 | 48.48 | 25.76 | 64.48 | 5.17 | PivotMesh [38] | 50.00 | 19.17 | 56.71 | 39.75 | 67.76 | 7.09 |  |  |
| MeshXL* [4] | 59.09 | 7.74 | 53.79 | 26.37 | 19.30 | 3.44 | MeshXL* [4] | 42.68 | 21.64 | 63.41 | 35.96 | 62.46 | 5.32 |  |  |
| Ours | 57.58 | 7.90 | 50.76 | 27.17 | 59.83 | 1.53 | Ours | 62.20 | 18.69 | 48.17 | 37.33 | 62.81 | 2.17 |  |  |

### 4.2 Mesh generation with the flow-based DiT

As illustrated in the right part of [Fig. 2](https://arxiv.org/html/2503.23022v1#S3.F2), the face-level tokens are fed into the flexible diffusion transformer for training. However, standard SiT [24] does not support direct training with token sequences of variable lengths. To address this, we introduce several techniques to adapt the architecture for mesh generation.

First, it is essential that the input tokens are of the same length. By padding the token sequences to match the length of the longest mesh sequence in the batch, the model can be trained using mesh sequences of varying lengths. The corresponding masks for these samples are also provided to the model, functioning as attention masks within the SiT blocks and guiding the unpadding process. RoPE [34] is applied to keys/values in attention layer. And the attention scores are computed as follows:

$$ $\mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d}}+M\right)$ (6) $$

where $\mathbf{Q}$ and $\mathbf{K}$ represent the queries and keys, respectively, and $d$ is their dimension. The value of $M$ is set to be 0 for noised tokens and $-\infty$ for padding tokens. Masks are also applied to exclude padding tokens before calculating losses, ensuring that generated tokens align with the input tokens. (refer to the supplementary for our detailed architectures)

In addition to masking techniques, we explicitly use the number of faces $c_{f}$ as guidance. Following the class conditioning approach implemented with adaLN-Zero blocks in DiT [29], we embed the number of faces using an embedding layer and add it to the timestep embeddings. This modification provides additional information and models a more accurate distribution of meshes according to the targeted number of faces, significantly enhancing generation quality when combined with classifier-free guidance (CFG) [13]:

$$ $\widetilde{v}_{t}=v_{\theta}(z_{t},\emptyset)+w\cdot(v_{\theta}(z_{t},c_{f})-v _{\theta}(z_{t},\emptyset))$ (7) $$

where $\theta$ and $z_{t}$ denote the network parameters and noised tokens, respectively. $\widetilde{v}_{t}$ represents the estimated velocity at each de-noising step. For conditioning the mesh geometry (e.g., from images or texts), we employ cross-attention modules to inject the corresponding features. Thus, for image-conditioned generation (similar to text conditions), we have two conditions: the input image $c_{i}$ and the assigned number of faces $c_{f}$. We find that the generation process benefits from multiple different CFG weights. Therefore, our modified estimation equation for the multiple conditions $c_{f}$ and $c_{i}$ can be expressed as:

$$ $\begin{split}\widetilde{v}_{t}&=v_{\theta}(z_{t},\emptyset,\emptyset)\\ &=+w_{1}\cdot(v_{\theta}(z_{t},c_{f},\emptyset)-v_{\theta}(z_{t},\emptyset, \emptyset))\\ &=+w_{2}\cdot(v_{\theta}(z_{t},c_{f},c_{i})-v_{\theta}(z_{t},c_{f},\emptyset)) \end{split}$ (8) $$

In [Sec. 5.3.2](https://arxiv.org/html/2503.23022v1#S5.SS3.SSS2), we present the effects of CFG weights on the generated meshes.

Additionally, to ensure more stable training, we introduce sandwich normalization [8], replace the MLP with SwiGLU, and employ QK-norm [6] techniques inspired by previous works [36, 42]. The QK-norm is crucial for stabilizing the training of transformer models, especially when the token length gets more flexible and longer.
Benefiting from the elaborate architecture and excellent properties of rectified flow’s linear sampling, training becomes more efficient and stable. We also draw inspiration from observations during the noising process to prioritize intermediate steps of diffusion, as the middle and final stages are more challenging during training. To alleviate this, we adopt the logit-normal sampling strategy from SD3 [9] to adjust the sampling weights:

$$ $\pi_{\mathrm{ln}}(t;m,s)=\frac{1}{s\sqrt{2\pi}t(1-t)}\exp\left(-\frac{(\log(t/ (1-t))-m)^{2}}{2s^{2}}\right)$ (9) $$

where $m$ and $s$ represent the location and scale parameters, respectively.

Finally, we utilize an MSE loss $\mathcal{L}_{Diff}$ to predict the velocity $v$ using our model.

Figure: Figure 5: Mesh completion results. Given some partial observation of a mesh, MeshCraft can produce diverse completed results.
Refer to caption: x5.png

## 5 Experiments

We trained our model on two datasets: ShapeNet [2] and Objaverse [7]. The model on ShapeNet demonstrates the effectiveness of our method (see [Fig. 4](https://arxiv.org/html/2503.23022v1#S4.F4) and [Tab. 2](https://arxiv.org/html/2503.23022v1#S4.T2) for the main results), which achieves state-of-the-art performance compared with prior works. Furthermore, experiments ([Fig. 6](https://arxiv.org/html/2503.23022v1#S5.F6)) conducted on Objaverse show that MeshCraft is able to generate diverse samples of different face numbers conditioned on the same single image, which shows its potential to be a practical mesh generator in larger-scale scenarios.

### 5.1 Experiment Settings

Figure: Figure 6: Generation diversity on Objaverse dataset. The number below each asset represents for the face number of it. MeshCraft is about to produce diverse samples with different seeds and face numbers.
Refer to caption: x6.png

#### 5.1.1 Datasets

Following the previous convention [32, 38, 4] for preprocessing, we apply planar decimation to meshes with more than $800$ faces and use further filtering by comparing the Hausdorff distance [15] between decimated and the original meshes with a pre-set threshold $\sigma_{hausdorff}$ for the ShapeNet dataset. As for the Objaverse dataset, we select assets whose number of faces is in $[1024,1536]$ to train our image-conditioned model. The dataset size is about 10k and 65k, respectively. We split the training set and validation set in 10:1 and 100:1. The coordinate space resolution is set as 128 and 256 for ShapeNet and Objaverse dataset, respectively. We normalize the coordinate range into $[-1,1]$ and augment the data using random scaling on each axis (from 0.95 to 1.05) and random rotations for the reconstruction stage to improve auto-encoder’s robustness. Across all experiments, we use triangular meshes (each face $f_{i}$ consists of 3 vertices) as training data for fair comparisons with baselines.

#### 5.1.2 Evaluation

We choose three recent baselines to compare the results on ShapeNet dataset, including MeshGPT [32], PivotMesh [38] and MeshXL [4]. For evaluating the reconstruction quality, we follow previous works [38, 32, 1] to use two metrics, triangle accuracy and l2 distance.
We evaluate the methods following previous mesh generation works [38]. The metrics include Minimum Matching Distance (MMD), Coverage (COV), 1-Nearest-Neighbor Accuracy (1-NNA), Jensen-Shannon Divergence (JSD) for measuring 3D geometry, FID and KID for visual perceptions. For COV, higher is better; for 1-NNA, 50% is the optimal; for the rest of metrics, lower is better. Following PivotMesh [38], we use a Chamfer Distance measure for computing these metrics on 1024-dim point clouds uniformly sampled from meshes.
For each method, we generate 1000 samples to calculate the quantitative results.

#### 5.1.3 Implementation Details

For transformer-styled blocks in the auto-encoder, we set the encoder part as 12 layers with a hidden size of 768, and 18 layers with a hidden size of 384 for the decoder part. The channel dimension for face-level tokens is set to 8 for the balance of reconstruction quality and compression capability. For the diffusion model, we adopt a 24-layer transformer with a hidden size of 864, which has similar number of parameters compared with baselines. For the implementation of baselines, we adopt the public code([https://github.com/lucidrains/meshgpt-pytorch](https://github.com/lucidrains/meshgpt-pytorch)) to reimplement MeshGPT [32], and official training code of PivotMesh [38]. For MeshXL [4], we use their released 350M pre-trained models for evaluation. All models are trained following the settings claimed in the original paper. Without further specification, we generate meshes in the distribution of data’s face numbers for MeshCraft and follow the default settings of baselines. For auto-encoders, we trained about 2 days on an 8$\times$A100 80GB machine with a batch size of 8. For diffusion transformers, we train for 3 days on the ShapeNet dataset and for around 3 weeks on the Objaverse dataset. During the training of diffusion transformers, we use bf16 mixed precision to accelerate the training process. All the generated results are sampled with the 50-step Euler method.

### 5.2 Experiment Results

#### 5.2.1 Results on ShapeNet dataset

Firstly, we evaluate our method on the most commonly used benchmark, ShapeNet, focusing on four different categories: chair, table, bench, and lamp. [Tab. 1](https://arxiv.org/html/2503.23022v1#S4.T1) shows that our continuous latent space VAE achieves competitive reconstruction performance compared with baselines that adopt an advanced vector quantization technique, RVQ [40], to discretize vertex coordinates into indices of a codebook. Furthermore, the number of tokens in face sequences decreases ninefold compared with prior works, which not only reduces memory requirements but also speeds up the subsequent generation process.
For the comparisons in the generation part, we follow the previous setting [38, 32, 1, 4], first pretraining our model on a mixed dataset composed of the four categories and subsequently fine-tuning each of them separately to perform comparisons from both qualitative and quantitative aspects. For the CFG weight $w$ of the face number condition, we set $w=8.0$ for better generation quality. As shown in [Fig. 4](https://arxiv.org/html/2503.23022v1#S4.F4), our method can produce diverse and high-quality 3D meshes. As [Tab. 2](https://arxiv.org/html/2503.23022v1#S4.T2) demonstrates, our method significantly outperforms the baselines, performing better on all three metrics. In [Fig. 5](https://arxiv.org/html/2503.23022v1#S4.F5), we also demonstrate the completion ability of our model by adapting the operation from an image inpainting work [23]. Compared with autoregressive methods, our diffusion-based method achieves state-of-the-art generation results while maintaining a superior generation speed. Specifically, as shown in [Fig. 1](https://arxiv.org/html/2503.23022v1#S0.F1)(b) and (c), our method decreases the token number by up to 9 times and speeds up generation by 35 times, demonstrating the advantages of diffusion-based models.

#### 5.2.2 Results on Objaverse dataset

An image-conditioned model is also trained to demonstrate the effectiveness of face number control on large-scale dataset Objaverse. To enhance the capability of capturing details in the image condition, we use DINOv2 ViT-L/14 [27] as the image feature extractor and fine-tuned with 3D assets whose front view occupies more than 20% of the frame. For the CFG weights $w1,w2$, which respectively controls the face number and the input image condition, we set them as $w1=1.0,w2=5.0$.
[Fig. 6](https://arxiv.org/html/2503.23022v1#S5.F6) displays that our model has the potential for the capability of diverse generation on the large-scale dataset, which can be also combined well with existing mature conditional techniques [13]. Notably, increasing the face number primarily enhances fine local details rather than uniformly refining the entire shape. For example, in the third row, the tiger’s facial features become more detailed, whereas broader regions such as the torso show only slight improvements.

**Table 3: Reconstruction performance study on Objverse.**
| Method | Tri. Accu.$(\%)\uparrow$ | L2 Dist.$(\times 10^{-2})\downarrow$ |
| --- | --- | --- |
| KL (4-dim) | 90.41 | 0.63 |
| KL (8-dim) | 99.66 | 0.12 |
| RVQ [40] | 65.12 | 8.63 |

### 5.3 Ablation Studies

#### 5.3.1 Comparisons of auto-encoder selections

At the reconstruction stage, we compare auto-encoders operating in different continuous and discrete latent spaces. As shown in [Tab. 3](https://arxiv.org/html/2503.23022v1#S5.T3), KL regularization with 8-dimensional token channels yields better performance, striking a good balance between reconstruction quality and compression capability compared to using 4-dimensional channels. We believe that using an excessively high compression ratio leads to severe information loss. Additionally, we compare the continuous tokenizer with the discrete one by replacing the compression component using KL regularization with residual vector quantization (RVQ) [40]. It is worth mentioning that discrete tokenizers, limited by their codebook size, can cause significant information loss when compressing variable data. In contrast, models with continuous tokens produce meshes of higher quality.

#### 5.3.2 Effects of CFG weights

We also investigate the effects of CFG weights on $w$ and $w_{1},w_{2}$, respectively. For the weight $w$ controlling the face number, [Tab. 4](https://arxiv.org/html/2503.23022v1#S5.T4) shows that as $w$ increases, the quantitative metrics improve, reaching a peak when $w$ is around 8.0.
For multiple conditions (single images $c_{i}$ and the number of faces $c_{f}$), [Fig. 7](https://arxiv.org/html/2503.23022v1#S5.F7) provides a visualization of the effects corresponding to different scales of $w_{1},w_{2}$. The best result appears when $w1=1.0,w2=5.0$. These two sets of experiments show that low weights can result in weak conditional control, while excessive weights are detrimental to the results.

**Table 4: Effects of CFG weights over the face number condition to mesh generation results on ShapeNet dataset.**
| CFG | COV$\uparrow$ | MMD$\downarrow$ | 1-NNA |
| --- | --- | --- | --- |
| 0.0 | 30.70 | 12.61 | 84.75 |
| 1.0 | 38.20 | 11.34 | 75.00 |
| 2.0 | 44.80 | 10.72 | 67.65 |
| 3.0 | 48.60 | 10.33 | 61.05 |
| 4.0 | 50.80 | 9.89 | 58.40 |
| 5.0 | 51.70 | 10.05 | 57.55 |
| 6.0 | 51.00 | 9.97 | 56.55 |
| 7.0 | 51.70 | 9.65 | 56.70 |
| 8.0 | 53.30 | 9.90 | 56.85 |
| 9.0 | 52.50 | 9.85 | 56.75 |
| 10.0 | 52.30 | 9.86 | 57.90 |

### 5.4 Limitations

Though MeshCraft shows promising results, there are still some limitations: (1) The extrapolation capability of our diffusion model is limited due to the face number embedder we use is learnable, and objects with unseen face numbers cannot be produced directly; (2) When the domain of images and assigned face numbers is far from the training domain, MeshCraft fails to generate completed meshes. We will further explore more generalizable models with improved training strategies and architectures.

Figure: Figure 7: CFG weights over face number and image conditions. $w_{1}$ controls on the condition of the face number $c_{f}$, while $w_{2}$ yields weights further over the single-image condition $c_{i}$.
Refer to caption: x7.png

## 6 Conclusion

In this paper, we introduce a novel method, namely MeshCraft, for generating ready-to-use 3D meshes with high efficiency and controllability. Regarding meshes as face-level sequences, we first compress them into continuous tokens and subsequently generate the tokens with a flow-based diffusion transformer. Our method demonstrates superior speed (35 $\times$ speed up) and shows competitive performance in both qualitative and quantitative experiments. MeshCraft shows the potential to alleviate artists from time-consuming manual work.

## References

- Alliegro et al. [2023]
Antonio Alliegro, Yawar Siddiqui, Tatiana Tommasi, and Matthias Nießner.
Polydiff: Generating 3d polygonal meshes with diffusion models.
*arXiv preprint arXiv:2312.11417*, 2023.
- Chang et al. [2015]
Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al.
Shapenet: An information-rich 3d model repository.
*arXiv preprint arXiv:1512.03012*, 2015.
- Chen et al. [2024a]
Junsong Chen, Yue Wu, Simian Luo, Enze Xie, Sayak Paul, Ping Luo, Hang Zhao, and Zhenguo Li.
Pixart-$\{$$\backslash$delta$\}$: Fast and controllable image generation with latent consistency models.
*arXiv preprint arXiv:2401.05252*, 2024a.
- Chen et al. [2024b]
Sijin Chen, Xin Chen, Anqi Pang, Xianfang Zeng, Wei Cheng, Yijun Fu, Fukun Yin, Yanru Wang, Zhibin Wang, Chi Zhang, et al.
Meshxl: Neural coordinate field for generative 3d foundation models.
*arXiv preprint arXiv:2405.20853*, 2024b.
- Chen et al. [2024c]
Yiwen Chen, Tong He, Di Huang, Weicai Ye, Sijin Chen, Jiaxiang Tang, Xin Chen, Zhongang Cai, Lei Yang, Gang Yu, et al.
Meshanything: Artist-created mesh generation with autoregressive transformers.
*arXiv preprint arXiv:2406.10163*, 2024c.
- Dehghani et al. [2023]
Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al.
Scaling vision transformers to 22 billion parameters.
In *International Conference on Machine Learning*, pages 7480–7512. PMLR, 2023.
- Deitke et al. [2023]
Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi.
Objaverse: A universe of annotated 3d objects.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 13142–13153, 2023.
- Ding et al. [2021]
Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, et al.
Cogview: Mastering text-to-image generation via transformers.
*Advances in neural information processing systems*, 34:19822–19835, 2021.
- Esser et al. [2024]
Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al.
Scaling rectified flow transformers for high-resolution image synthesis.
In *Forty-first International Conference on Machine Learning*, 2024.
- Fan et al. [2024]
Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian.
Fluid: Scaling autoregressive text-to-image generative models with continuous tokens.
*arXiv preprint arXiv:2410.13863*, 2024.
- Hao et al. [2024]
Zekun Hao, David W Romero, Tsung-Yi Lin, and Ming-Yu Liu.
Meshtron: High-fidelity, artist-like 3d mesh generation at scale.
*arXiv preprint arXiv:2412.09548*, 2024.
- He et al. [2025]
Xianglong He, Junyi Chen, Sida Peng, Di Huang, Yangguang Li, Xiaoshui Huang, Chun Yuan, Wanli Ouyang, and Tong He.
Gvgen: Text-to-3d generation with volumetric representation.
In *European Conference on Computer Vision*, pages 463–479. Springer, 2025.
- Ho and Salimans [2022]
Jonathan Ho and Tim Salimans.
Classifier-free diffusion guidance.
*arXiv preprint arXiv:2207.12598*, 2022.
- Ho et al. [2020]
Jonathan Ho, Ajay Jain, and Pieter Abbeel.
Denoising diffusion probabilistic models.
*Advances in neural information processing systems*, 33:6840–6851, 2020.
- Huttenlocher et al. [1993]
Daniel P Huttenlocher, Gregory A. Klanderman, and William J Rucklidge.
Comparing images using the hausdorff distance.
*IEEE Transactions on pattern analysis and machine intelligence*, 15(9):850–863, 1993.
- Kerbl et al. [2023]
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
3d gaussian splatting for real-time radiance field rendering.
*ACM Trans. Graph.*, 42(4):139–1, 2023.
- Kingma [2013]
Diederik P Kingma.
Auto-encoding variational bayes.
*arXiv preprint arXiv:1312.6114*, 2013.
- Lin et al. [2023]
Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin.
Magic3d: High-resolution text-to-3d content creation.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 300–309, 2023.
- Liu et al. [2022]
Xingchao Liu, Chengyue Gong, and Qiang Liu.
Flow straight and fast: Learning to generate and transfer data with rectified flow.
*arXiv preprint arXiv:2209.03003*, 2022.
- Liu et al. [2024]
Ying-Tian Liu, Yuan-Chen Guo, Guan Luo, Heyi Sun, Wei Yin, and Song-Hai Zhang.
Pi3d: Efficient text-to-3d generation with pseudo-image diffusion.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19915–19924, 2024.
- Lorensen and Cline [1998]
William E Lorensen and Harvey E Cline.
Marching cubes: A high resolution 3d surface construction algorithm.
In *Seminal graphics: pioneering efforts that shaped the field*, pages 347–353. 1998.
- Lu et al. [2024]
Zeyu Lu, Zidong Wang, Di Huang, Chengyue Wu, Xihui Liu, Wanli Ouyang, and Lei Bai.
Fit: Flexible vision transformer for diffusion model.
*arXiv preprint arXiv:2402.12376*, 2024.
- Lugmayr et al. [2022]
Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool.
Repaint: Inpainting using denoising diffusion probabilistic models.
In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 11461–11471, 2022.
- Ma et al. [2024]
Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie.
Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers.
*arXiv preprint arXiv:2401.08740*, 2024.
- Mildenhall et al. [2021]
Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng.
Nerf: Representing scenes as neural radiance fields for view synthesis.
*Communications of the ACM*, 65(1):99–106, 2021.
- Nash et al. [2020]
Charlie Nash, Yaroslav Ganin, SM Ali Eslami, and Peter Battaglia.
Polygen: An autoregressive generative model of 3d meshes.
In *International conference on machine learning*, pages 7220–7229. PMLR, 2020.
- Oquab et al. [2023]
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al.
Dinov2: Learning robust visual features without supervision.
*arXiv preprint arXiv:2304.07193*, 2023.
- Park et al. [2019]
Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove.
Deepsdf: Learning continuous signed distance functions for shape representation.
In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 165–174, 2019.
- Peebles and Xie [2023]
William Peebles and Saining Xie.
Scalable diffusion models with transformers.
In *Proceedings of the IEEE/CVF International Conference on Computer Vision*, pages 4195–4205, 2023.
- Poole et al. [2022]
Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall.
Dreamfusion: Text-to-3d using 2d diffusion.
*arXiv preprint arXiv:2209.14988*, 2022.
- Rombach et al. [2022]
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.
High-resolution image synthesis with latent diffusion models.
In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pages 10684–10695, 2022.
- Siddiqui et al. [2024]
Yawar Siddiqui, Antonio Alliegro, Alexey Artemov, Tatiana Tommasi, Daniele Sirigatti, Vladislav Rosov, Angela Dai, and Matthias Nießner.
Meshgpt: Generating triangle meshes with decoder-only transformers.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 19615–19625, 2024.
- Song et al. [2020]
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole.
Score-based generative modeling through stochastic differential equations.
*arXiv preprint arXiv:2011.13456*, 2020.
- Su et al. [2024]
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu.
Roformer: Enhanced transformer with rotary position embedding.
*Neurocomputing*, 568:127063, 2024.
- Tang et al. [2025]
Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu.
Lgm: Large multi-view gaussian model for high-resolution 3d content creation.
In *European Conference on Computer Vision*, pages 1–18. Springer, 2025.
- Touvron et al. [2023]
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.
Llama: Open and efficient foundation language models.
*arXiv preprint arXiv:2302.13971*, 2023.
- Van Den Oord et al. [2017]
Aaron Van Den Oord, Oriol Vinyals, et al.
Neural discrete representation learning.
*Advances in neural information processing systems*, 30, 2017.
- Weng et al. [2024]
Haohan Weng, Yikai Wang, Tong Zhang, CL Chen, and Jun Zhu.
Pivotmesh: Generic 3d mesh generation via pivot vertices guidance.
*arXiv preprint arXiv:2405.16890*, 2024.
- Xu et al. [2024]
Yinghao Xu, Zifan Shi, Wang Yifan, Hansheng Chen, Ceyuan Yang, Sida Peng, Yujun Shen, and Gordon Wetzstein.
Grm: Large gaussian reconstruction model for efficient 3d reconstruction and generation.
*arXiv preprint arXiv:2403.14621*, 2024.
- Zeghidour et al. [2021]
Neil Zeghidour, Alejandro Luebs, Ahmed Omran, Jan Skoglund, and Marco Tagliasacchi.
Soundstream: An end-to-end neural audio codec.
*IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 30:495–507, 2021.
- Zhao et al. [2024]
Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao.
Michelangelo: Conditional 3d shape generation based on shape-image-text aligned latent representation.
*Advances in Neural Information Processing Systems*, 36, 2024.
- Zhuo et al. [2024]
Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al.
Lumina-next: Making lumina-t2x stronger and faster with next-dit.
*arXiv preprint arXiv:2406.18583*, 2024.

## 7 Additional details of the model design

### 7.1 Detailed architectures of modules

The proposed model architecture comprises intricate transformer-styled blocks with carefully designed components. As illustrated in [Fig. 8](https://arxiv.org/html/2503.23022v1#S6.F8)(a), the transformer-styled block is composed of two attention layers, followed by an RMS normalization and a feed-forward layer. [Fig. 8](https://arxiv.org/html/2503.23022v1#S6.F8)(b) and (c) provide a detailed visualization of our DiT-styled block and the final layer in the diffusion transformer, highlighting the nuanced design choices that contribute to the model’s performance.

### 7.2 Logit-normal sampling

Drawing inspiration from the training methodology of SD3 [9], we inspect the diffusion process of noises to the latent tokens, visualizing in [Fig. 9](https://arxiv.org/html/2503.23022v1#S7.F9). The visualization in [Fig. 9](https://arxiv.org/html/2503.23022v1#S7.F9) reveals critical insights into the mesh generation process, demonstrating that intermediate and final diffusion steps play a pivotal role in generating complete meshes. Consequently, we implemented a logit-normal sampling approach to emphasize these crucial stages of the generation process. By setting the distribution parameters to $m=0.5$ and $s=1.0$, informed by empirical results from SD3, we effectively prioritize sampling in the most informative regions of the diffusion process.

Figure: Figure 9: Process of adding noises. The complete mesh is gradually transformed into noises from standard normal distribution from $t=1$ to $t=0$.
Refer to caption: x9.png

Figure: Figure 10: Loss during training on Objaverse dataset. The QK-norm is of vital importance for stabilizing the training process.
Refer to caption: extracted/6319938/pictures/objaverse_loss.png

### 7.3 The importance of QK-norm

Recent advancements in transformer research [6, 22, 42] have highlighted the inherent challenges of training large-parameter models with flexible data sequences. Our mesh generation experiments on the Objaverse dataset corroborated these observations, revealing significant training instabilities. To address this critical issue, we implemented the QK-norm technique, a proven strategy for mitigating training volatility. Formally, we modified the attention scores as follows:

$$ $\mathrm{Softmax}\left(\frac{\mathrm{LN}(\mathbf{Q})\mathrm{LN}(\mathbf{K})^{T} }{\sqrt{d}}+M\right)$ (10) $$

By applying LayerNorm to the query and key matrices, we effectively stabilize the attention mechanism. [Fig. 10](https://arxiv.org/html/2503.23022v1#S7.F10) demonstrates that QK-norm helps to stabilize the process.

Figure: Figure 11: Generation gallery on ShapeNet. Additional results on the subset of bench, lamp, chair and table.
Refer to caption: x10.png

Figure: Figure 12: Additional generation results on Objaverse.
Refer to caption: x11.png

Figure: Figure 13: Point cloud conditioned generation on the ShapeNet bench dataset.
Refer to caption: x12.png

## 8 Additional quantitative results

We present comprehensive visualization of generation results across multiple datasets, showcasing the versatility of our approach in [Fig. 11](https://arxiv.org/html/2503.23022v1#S7.F11), [Fig. 12](https://arxiv.org/html/2503.23022v1#S7.F12), and [Fig. 13](https://arxiv.org/html/2503.23022v1#S7.F13). To demonstrate the model’s adaptability, we also extended our method to point cloud conditioning on the ShapeNet bench dataset. Our implementation leverages a pre-trained point cloud encoder inspired by the Michelangelo [41] architecture. We integrated the point cloud features into our model using cross-attention techniques, analogous to image feature injection. Specifically, we employed linear projections to seamlessly adapt and align the point cloud representations with our model’s internal feature space. Refer to [Fig. 13](https://arxiv.org/html/2503.23022v1#S7.F13) for the results.