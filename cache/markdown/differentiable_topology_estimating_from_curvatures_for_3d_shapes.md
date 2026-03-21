## Contents
- I Introduction
- II Methods
  - II-A Local Tangent Voronoi Area
  - II-B Self-Adjoint Weingarten Fields and Curvatures
  - II-C Differentiable Topology Estimation
- III Experiments
  - III-A Performance of Curvature Estimation
  - III-B Performance on Topology Estimation
- IV Conclusion
- References

## Abstract

Abstract In the field of data-driven 3D shape analysis and generation, the estimation of global topological features from localized representations such as point clouds, voxels, and neural implicit fields is a longstanding challenge. This paper introduces a novel, differentiable algorithm tailored to accurately estimate the global topology of 3D shapes, overcoming the limitations of traditional methods rooted in mesh reconstruction and topological data analysis. The proposed method ensures high accuracy, efficiency, and instant computation with GPU compatibility. It begins with an efficient calculation of the self-adjoint Weingarten map for point clouds and its adaptations for other modalities. The curvatures are then extracted, and their integration over tangent differentiable Voronoi elements is utilized to estimate key topological invariants, including the Euler number and Genus. Additionally, an auto-optimization mechanism is implemented to refine the local moving frames and area elements based on the integrity of topological invariants. Experimental results demonstrate the method’s superior performance across various datasets. The robustness and differentiability of the algorithm ensure its seamless integration into deep learning frameworks, offering vast potential for downstream tasks in 3D shape analysis.

## I Introduction

In recent years, 3D shape analysis and generation have become increasingly popular research topics in computer vision, computer graphics, and machine learning. Many impressive studies and applications have been developed to achieve 3D object recognition, reconstruction, and generation. However, the data-driven 3D shape analysis and generation still face many challenges regardless of the kind of localized data representation used, from point clouds, occupancy or signature distance fields (SDF) [1, 2, 3, 4], to neural radiance fields (NeRF) [5, 6, 7] and Gaussian Splattings (GS) [8, 9, 10, 11]. One essential challenge is that existing methods are usually locally focused and lack consistency and coherence in terms of global view, which causes several deformities and artifacts in the generated 3D shapes. Although researchers have made great efforts to improve the global coherence of 3D shapes via more advanced architectures and supervision mechanics in deep learning [2, 1, 8, 5], few studies have focused on the global topology of 3D shapes and its estimation from the localized data representations in a differentiable manner. In this paper, we propose a differentiable algorithm to estimate the global topology of 3D shapes, which can be easily integrated into deep learning frameworks.

The global topology of 3D shapes is a fundamental property that characterizes the shape’s connectivity and the number of holes. The main reason is that the 3D shape contains rich and essential information from its global topology, which is usually hard to learn by a locally focused optimization from deep learning. Therefore, a differentiable algorithm that can accurately estimate the global topology of 3D shapes with modalities adaptivity is highly demanded. On the other hand, efficient topological estimation is also challenging in traditional computer graphics and computational geometry, let alone the requirement for differentiability. Established methods to analyze the topology of discrete 3D shapes include the Reeb graph [12], Morse theory [13], and persistent homology
[14, 15, 16] from topological data analysis (TDA). However, current methods have inherent limitations in differentiability and computational efficiency, making them difficult to integrate into the deep learning frameworks.

To achieve globality and differentiability simultaneously, we propose a novel algorithm to estimate topology based on the estimation of the local curvatures and its stable integration, inspired by the great idea of Gauss-Bonnet theorem [17]. The Gauss-Bonnet theorem is a fundamental result in differential geometry that establishes a relationship between the local curvature and the global topology of a surface. It states that the integral of the Gaussian curvature $K$ over a surface (manifold) ${M}$ is equal to $2\pi$ times the Euler characteristic $\chi_{{M}}$, i.e.,

$$ $\frac{1}{2\pi}\int_{{M}}KdA=\chi_{{M}},$ (1) $$

where $dA$ is the integral area element.

From now on, the question turns to how to estimate the local curvatures and integrate them in a differentiable and stable way. Efficient estimation of (Gaussian/mean/principal) curvature is also an essential but difficult problem. Taking the point cloud as an example (unordered point sets contain the least information and can be easily extracted from other modalities), several methods have been proposed to estimate the curvature. [18] introduced the Voronoi covariance measure (VCM) to estimate the curvature for noised point clouds. Robust methods varied from local statistics on the sampled neighborhoods are the prevalent alternatives to estimate directional curvatures while coping with noise and irregular sampling, such as [19] involving least squares and [20] adopting local re-weighting. The classic paper [21] defined the discrete Weingarten map in an integration formula on meshed and [22] adopted it to estimate the principal curvature on point clouds. [23] proposed a method for computing the covariance matrices of normal vectors to approximate the shape operator (equivalent to the Weingarten map to some extent). More recently, [24] used robust statistical methods to estimate the curvature tensor of a point cloud by considering normal variation in a neighborhood of a given point. [25] derived analytic expressions for computing principal curvatures based on the implicit definition of the moving least squares surface by [26]. [22] formulated the statistical algorithms and gave out the convergence analysis. However, none of them shows sufficient robustness and accuracy to support the integration for global topology where error accumulation occurs.

We tend to estimate the curvature by the Weingarten map following the idea of [22], but especially emphasize the self-adjoint property of the Weingarten map, which is crucial for the stability of the curvature estimation. Moreover, our method will be specifically designed to avoid non-differentiable operations and non-stable computations, which would probably cause the gradient explosion or vanishing during the back-propagation in deep learning. Subsequently, the curvatures will be integrated over tangent differentiable Voronoi elements [27], where we obtain the initial estimation of the Euler characteristic and genus. Finally, a self-optimization mechanic leveraging the integrity of topological invariants is implemented to rectify the local moving frames and area elements, from which the fine estimation of the topology can be achieved. The self-optimization addresses the unreliability of the initial normal estimation and the area elements from local statistics. The occupancy information of spacial points contained in occupancy or SDF will be coded into the winding number and play the role of regularization in the differentiable form inspired by [28].

## II Methods

In this section, we present our methodology, starting with the estimation of moving frames and local area elements. We then analyze the self-adjoint Weingarten map and propose a highly robust approach for estimating Gaussian curvature. Finally, we incorporate curvature estimates with a self-optimization technique that refines local frames and area elements, optimizing them in alignment with the integrity-well loss to preserve the global topology.

### II-A Local Tangent Voronoi Area

Consider a point cloud $\mathcal{P}:=\{\mathbf{p}_{i}\}_{i=1}^{N}$ lying on an underlying surface ${M}\subseteq\mathbb{R}^{3}$. For each point $\mathbf{p}_{i}$, a moving orthonormal frame $\{\mathbf{t}_{i},\mathbf{t}^{\prime}_{i},\mathbf{n}_{i}\}$ is defined by a group of basis spanning the $3$-dimensional linear space originating from $\mathbf{p}_{i}$, where $\mathbf{t}_{i}$ and $\mathbf{t}^{\prime}_{i}$ span the tangent space $T_{\mathbf{p}_{i}}M$, and $\mathbf{n}_{i}$ is the normal vector.

In a discrete setting, the orthonormal frame at a point $\mathbf{p}_{i}$ can be estimated via local principal component analysis (PCA) of its $k$-nearest neighbors (kNN). Let $\mathcal{N}_{i}=\{\mathbf{p}_{ij}\}_{j=1}^{k}$ denote the $k$-nearest neighbors of $\mathbf{p}_{i}$, organized as a $k\times 3$ matrix.
Then eigenvectors of $\operatorname{cov}(\mathcal{N}_{i})$ corresponding to the first two largest eigenvalues represent the tangent vectors $\mathbf{t}_{i}$ and $\mathbf{t}^{\prime}_{i}$, while the eigenvector associated with the smallest eigenvalue defines the normal vector $\mathbf{n}_{i}$.

Since the exact eigenvalues are not needed for subsequent computations, we perform local PCA using a normalized version of the covariance matrix, $\operatorname{cov}(\mathcal{N}_{i})/\mathrm{Tr}(\operatorname{cov}(\mathcal{N}
_{i}))$, where division by the trace $\mathrm{Tr}(\operatorname{cov}(\mathcal{N}_{i}))$ helps mitigate scale effects and improves computational stability.

In theory, the local positional offsets $\mathrm{d}\mathbf{p}_{ij}=\mathbf{p}_{ij}-\mathbf{p}_{i}$ lie in the tangent space $T_{\mathbf{p}_{i}}M$. However, due to sparsity in local neighborhoods, they may not be perfectly tangent to the surface. Practically, $\mathrm{d}\mathbf{p}_{ij}$ is projected onto the tangent space using inner products with $\mathbf{t}_{i}$ and $\mathbf{t}^{\prime}_{i}$. This projection yields a 2-dimensional local tangent coordinate system for $\mathcal{N}_{i}$, i.e.,

$$ $\mathrm{d}\widetilde{\mathbf{p}}_{ij}=\mathrm{d}\mathbf{p}_{ij}[\mathbf{t}_{i} ,\mathbf{t}^{\prime}_{i}],$ (2) $$

where $\mathrm{d}\widetilde{\mathbf{p}}_{ij}$ forms a $k\times 2$ matrix representing a 2D local point set on the tangent plane.

Figure: Figure 2: Tangent Voronoi Diagram for estimating areas.
Refer to caption: x2.png

The local area element at point $\mathbf{p}_{i}$ is estimated using a tangent Voronoi diagram constructed on the 2D point set $\mathrm{d}\widetilde{\mathbf{p}}_{ij}$, as illustrated in Fig. [2](https://arxiv.org/html/2412.00140v1#S2.F2). The Voronoi diagram [27] partitions the entire region into polygonal cells, each associated with a seed point. The central Voronoi cell of $\mathbf{p}_{i}$ defines an exclusive region whose area represents the local area element of $\mathbf{p}_{i}$.

To enable differentiable computation, the local area element can be estimated using Monte Carlo simulation. For point $\mathbf{p}_{i}$, a 2D mesh grid is constructed over a suitably scaled tangent bounding square (scaled by $\times 1.1$ in practice) encompassing $\mathrm{d}\widetilde{\mathbf{p}}_{ij}$. This grid is denoted by $\mathcal{G}_{i}$. Let $\mathcal{G}_{i}^{c}$ denote the grid points within the central Voronoi cell, identified by 1-nearest neighbor (1-NN) searching as follows:

$$ $\mathcal{G}_{i}^{c}=\left\{v\in\mathcal{G}_{i}\mid d(v,\mathbf{p}_{i})\leq d(v ,\mathcal{N}_{i})\right\},$ $$

where $d(v,\mathbf{p}_{i})$ is the distance from grid point $v$ to $\mathbf{p}_{i}$, and $d(v,\mathcal{N}_{i})$ is the minimum distance to any neighbor in $\mathcal{N}_{i}$. The area element $A_{i}$ is approximated by the Monte Carlo ratio:

$$ $A_{i}\approx\frac{\#\mathcal{G}_{i}^{c}}{\#\mathcal{G}_{i}}A_{bbx},$ (3) $$

where $\#$ denotes the number of points in a finite set, $A_{bbx}$ is the area of the bounding square, and the ratio represents the proportion of grid points in the central Voronoi cell. The summation (discrete integration of the constant function $f\equiv 1$) over all local Voronoi area elements provides a differentiable estimate of the total area of the 3D shape.

Fig. [2](https://arxiv.org/html/2412.00140v1#S2.F2) demonstrates tangent Voronoi area estimation on a 10K point cloud of the model "Father’s Strength"(^1^11The 3D model ”Father’s Strength” is available on [TurboSquid](https://www.turbosquid.com/3d-models/father-s-strength-880248), created by Lurisay, and will be consistently used as the showcase in Section 2.). The estimated area $A=9.1376$ closely approximates the ground truth $\hat{A}=9.220$, yielding a relative error of $0.93\%$. Refer to Section [III](https://arxiv.org/html/2412.00140v1#S3) for additional experimental results.

### II-B Self-Adjoint Weingarten Fields and Curvatures

Figure: Figure 3: Illustration of the Weingarten map for the manifold $M$.
Refer to caption: x3.png

With a method to estimate the local area element, we can now proceed to estimate the curvatures of the 3D shape. The Weingarten map is the foundation for estimating curvatures, where two principal curvatures are two eigenvalues of the Weingarten map. The Weingarten map is defined as the derivative of the normal (Gaussian) map. Under the fact that

$$ $\langle\mathbf{n}_{i},\mathrm{d}\mathbf{n}_{i}\rangle=\frac{1}{2}\mathrm{d} \langle\mathbf{n}_{i},\mathbf{n}_{i}\rangle=0,$ $$

$\mathrm{d}\mathbf{n}_{i}$ is tengent to $M$ and the tangent space of $M$ at $\mathbf{p}_{i}$ is naturally identifed to the tangent space of standard sphere $S^{2}$ at $\mathbf{n}_{i}$ , i.e.,
$T_{\mathbf{p}_{i}}M\cong T_{\mathbf{n}_{i}}S^{2}.$
Hence the Weingarten map $W_{i}$ can be viewed as an endomorphism on $T_{\mathbf{p}_{i}}M$. In our settings, we have

$$ $\displaystyle W_{i}:$ $\displaystyle T_{\mathbf{p}_{i}}M\rightarrow T_{\mathbf{n}_{i}}S^{2}\cong T_{ \mathbf{p}_{i}}M,$ (4) $\displaystyle W_{i}(\mathbf{v})=-\nabla_{\mathbf{v}}\mathbf{n}_{i},\ \forall \mathbf{v}\in T_{\mathbf{p}_{i}}M.$ $$

Notice that $W_{i}:\mathrm{d}\mathbf{p}_{ij}\mapsto\mathrm{d}\mathbf{n}_{ij}$ only holds inside the tangent plane in idealized cases, we approximate the matrix equation $W_{i}\mathrm{d}\mathbf{p}_{ij}^{\intercal}=\mathrm{d}\mathbf{n}_{ij}^{\intercal}$ by their tangent projections $W_{i}\mathrm{d}\widetilde{\mathbf{p}}_{ij}^{\intercal}=\mathrm{d}\widetilde{
\mathbf{n}}_{ij}^{\intercal}$, where $W_{i}$ is written as a $2\times 2$ matrix. Figure [3](https://arxiv.org/html/2412.00140v1#S2.F3) demonstrates the above conceptions. The estimation of $W_{i}$ attributes to solving the least squares problem:

$$ $W_{i}=\arg\min_{W_{i}}\|W_{i}\mathrm{d}\widetilde{\mathbf{p}}_{ij}^{\intercal} -\mathrm{d}\widetilde{\mathbf{n}}_{ij}^{\intercal}\|_{F}^{2}.$ (5) $$

Traditionally, the solution to Eq. ([5](https://arxiv.org/html/2412.00140v1#S2.E5)) is obtained through the pseudo-inverse,

$$ $W_{i}=(\mathrm{d}\widetilde{\mathbf{p}}_{ij}^{\intercal}\mathrm{d}\widetilde{ \mathbf{p}}_{ij})^{-1}\mathrm{d}\widetilde{\mathbf{p}}_{ij}^{\intercal}\mathrm {d}\widetilde{\mathbf{n}}_{ij},$ (6) $$

as implemented in [22]. However, an important property of the Weingarten map is that it is self-adjoint (or Hermitian [29]), i.e., $W_{i}=W_{i}^{\intercal}$. This property follows from the definition of the Weingarten map ([4](https://arxiv.org/html/2412.00140v1#S2.E4)) and its commutativity with the inner product:

$$ $\displaystyle\mathbf{u}^{\intercal}W_{i}\mathbf{v}=$ $\displaystyle-\langle\nabla_{\mathbf{u}}\mathbf{n}_{i},\mathbf{v}\rangle= \langle\mathbf{n}_{i},\nabla_{\mathbf{u}}\mathbf{v}\rangle$ $\displaystyle=$ $\displaystyle-\langle\mathbf{n}_{i},\nabla_{\mathbf{v}}\mathbf{u}\rangle=- \langle\mathbf{u},\nabla_{\mathbf{v}}\mathbf{n}_{i}\rangle$ $\displaystyle=$ $\displaystyle\ \mathbf{v}^{\intercal}W_{i}\mathbf{u},\quad\forall\mathbf{u}, \mathbf{v}\in T_{\mathbf{p}_{i}}M,$ $$

where the second and fourth equalities hold due to the orthogonality of the normal vector $\mathbf{n}_{i}$ with the tangent vectors $\mathbf{u}$ and $\mathbf{v}$. The third equality follows from the Lie bracket

$$ $[\mathbf{v},\mathbf{u}]=\nabla_{\mathbf{u}}\mathbf{v}-\nabla_{\mathbf{v}} \mathbf{u},$ $$

which lies in the tangent plane and is orthogonal to the normal vector. The self-adjoint property of the Weingarten map guarantees real eigenvalues, which correspond to the principal curvatures. Therefore, obtaining a symmetric solution for Eq. ([5](https://arxiv.org/html/2412.00140v1#S2.E5)) is essential for stable curvature estimation.

Combining the linearity of $W_{i}$, $W_{i}=W_{i}^{\intercal}$, and Eq. ([4](https://arxiv.org/html/2412.00140v1#S2.E4)), we can derive

$$ $\displaystyle{\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{ \mathbf{n}}_{ij}+{\rm d}\widetilde{\mathbf{n}}_{ij}^{\intercal}{\rm d} \widetilde{\mathbf{p}}_{ij}=\left({\rm d}\widetilde{\mathbf{p}}_{ij}^{ \intercal}{\rm d}\widetilde{\mathbf{p}}_{ij}\right)W_{i}+W_{i}\left({\rm d} \widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{\mathbf{p}}_{ij} \right),$ (7) $$

which defines a Sylvester equation [30] for the self-adjoint Weingarten map $W_{i}$. The algebraic properties of the Sylvester equation are thoroughly described in [31, 32], and the solution to Eq. ([7](https://arxiv.org/html/2412.00140v1#S2.E7)) can be computed using the following Algorithm [1](https://arxiv.org/html/2412.00140v1#alg1).

Figure: Algorithm 1 Solution of the Self-adjoint Weingarten Map

Figure: Figure 4: The estimation of the Weingarten map and the curvatures.
Refer to caption: x4.png

By the eigen-decomposition of the symmetric matrix $W_{i}$, the Gaussian curvature is calculated as the product of the principal curvatures equal to the determinate of $W_{i}$,

$$ $K_{i}=\lambda_{\min}(W_{i})\lambda_{\max}(W_{i})=\det(W_{i}).$ (8) $$

The mean curvature is given by the average of the principal curvatures:

$$ $H_{i}=\frac{1}{2}(\lambda_{\min}(W_{i})+\lambda_{\max}(W_{i}))=\frac{1}{2} \operatorname{Tr}(W_{i}).$ $$

Additionally, the total curvature can be derived from the Frobenius norm of the Weingarten map:

$$ $F_{i}=\|W_{i}\|_{F}=\operatorname{Tr}(W_{i}^{\intercal}W_{i}).$ $$

Figure [4](https://arxiv.org/html/2412.00140v1#S2.F4) illustrates the curvature estimation results.

Algorithm [1](https://arxiv.org/html/2412.00140v1#alg1) exclusively uses differentiable operations, enabling gradient backpropagation. However, multiple steps of eigen-decompositions can be computationally intensive and may lead to gradient explosion. To enhance stability and efficiency, we propose two strategies to simplify Algorithm [1](https://arxiv.org/html/2412.00140v1#alg1), particularly for Gaussian curvature estimation.

The first approach assumes that $W_{i}$ and ${\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{\mathbf{p}}_{ij}$ commute, implying

$$ $W_{i}{\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{\mathbf{p }}_{ij}={\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{ \mathbf{p}}_{ij}W_{i}.$ $$

This assumption simplifies the Sylvester equation ([7](https://arxiv.org/html/2412.00140v1#S2.E7)) as follows:

$$ $W_{i}{\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{\mathbf{p }}_{ij}=\frac{1}{2}\left({\rm d}\widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d} \widetilde{\mathbf{n}}_{ij}+{\rm d}\widetilde{\mathbf{n}}_{ij}^{\intercal}{\rm d }\widetilde{\mathbf{p}}_{ij}\right).$ $$

Taking the determinant on both sides yields that

$$ $K_{i}=\det(W_{i})\approx\frac{\det\left({\rm d}\widetilde{\mathbf{p}}_{ij}^{ \intercal}{\rm d}\widetilde{\mathbf{n}}_{ij}+{\rm d}\widetilde{\mathbf{n}}_{ij }^{\intercal}{\rm d}\widetilde{\mathbf{p}}_{ij}\right)}{4\det\left({\rm d} \widetilde{\mathbf{p}}_{ij}^{\intercal}{\rm d}\widetilde{\mathbf{p}}_{ij} \right)},$ (9) $$

where eigen-decomposition is avoided. The non-negativity and symmetry of the covariance matrix in the denominator help ensure stability, except for singular local distributions, which can be addressed with small perturbations. The second approach involves directly symmetrizing the non-Hermitian solution of Eq.([5](https://arxiv.org/html/2412.00140v1#S2.E5)) by updating

$$ $W_{i}\leftarrow\frac{1}{2}(W_{i}+W_{i}^{\intercal}),$ (10) $$

where the $W_{i}$ on the right-hand side is given by Eq.([6](https://arxiv.org/html/2412.00140v1#S2.E6)).

### II-C Differentiable Topology Estimation

By integrating the differentiable estimation of local area elements and curvatures, we can approximate the global topology of the point cloud. The Euler characteristic $\chi_{M}$ is estimated using the discrete Gauss-Bonnet theorem:

$$ $\chi_{M}\approx\frac{1}{2\pi}\sum_{i=1}^{N}K_{i}A_{i},$ (11) $$

where $K_{i}$ is the Gaussian curvature and $A_{i}$ is the area element of point $p_{i}$. From the Riemann-Hurwitz formula [29], the genus $g(M)$ of the closed surface can be derived as

$$ $\chi_{M}=2-2g(M).$ (12) $$

Though, in theory, topological invariants are integers, our estimates yield real numbers due to differentiable calculations and estimation errors. However, this minor discrepancy can works as a regularization mechanism for lower-order geometric features like local offsets, normals, area elements, and tangent frames.

We refine the topology estimate by introducing a self-optimization mechanism, using an integrity-well loss on topological invariants. The integrity-well loss function is defined as

$$ $w_{int}(x)=\left(\sin\left(\pi x-\frac{\pi}{2}\right)+1\right)^{2},$ $$

illustrated in Fig. [5](https://arxiv.org/html/2412.00140v1#S2.F5).
This loss is designed to favor integer values and even-numbered invariants (aligned with the Riemann-Hurwitz formula Eq.([12](https://arxiv.org/html/2412.00140v1#S2.E12))), as most real-world 3D shapes have even-genus surfaces not exceeding $2$.

Figure: Figure 5: Integrity-well Loss.
Refer to caption: x5.png

The self-optimization mechanism backpropagates the integrity-well loss over local offsets, normals, area elements, and frames. In normal optimization, for example, the loss is added to the normal estimation loss function. Typically, the normal fields are initialized using negative gradients of the signed distance function (SDF) or local PCA. Given $\mathbf{n}_{i}=(\varphi_{i},\theta_{i})$, the $k$-step self-optimization of normals minimizes the loss function for unit vector fields ${(\varphi_{i},\theta_{i})}_{i=1}^{N}$ as

$$ $\displaystyle(\varphi_{i},\theta_{i})^{(k+1)}-(\varphi_{i},\theta_{i})^{(k)}$ $\displaystyle=$ $\displaystyle-lr\frac{\partial}{\partial(\varphi_{i},\theta_{i})}\left(\|\chi^ {(k)}-\chi_{Gt}\|+w_{int}\left(\chi^{(k)}\right)\right),$ (13) $$

where $\chi^{(k)}$ is the current Euler characteristic estimate, $lr$ is the learning rate, and $\chi_{Gt}$ is the ground truth. If $\chi_{Gt}$ is unavailable, global winding number supervision may be used as suggested in [28].

Figure: Figure 6: Three independent handles indicate an estimated genus of $g(M)=3$.
Refer to caption: x6.png

Starting with ambiguous normal estimates, the optimization process refines these normals, leading to a near-integer approximation of the topology. For example, in our experiment with the "Father’s Strength" model, the Euler characteristic converged to $-4\pm 0.02$ from an initial estimate of $-3.5\pm 0.02$, accurately indicating a genus $g=3$, consistent with the observed model structure shown in Fig.[6](https://arxiv.org/html/2412.00140v1#S2.F6). Details of the optimization process are illustrated in Fig.[7](https://arxiv.org/html/2412.00140v1#S2.F7).

Figure: Figure 7: Convergence of the self-optimization over normals.
Refer to caption: x7.png

With these novel algorithms and the designed architecture, we achieved a differentiable estimation of the global topology of 3D shapes. Algorithm [2](https://arxiv.org/html/2412.00140v1#alg2) summarizes the complete pipeline of our method. For other forms of 3D representations, point clouds can be easily extracted, and additional information about local frames can be naturally incorporated. For instance, gradient vectors of implicit field can imply normal directions, and Gaussian Splatting can directly provide local frames through the rotation component of the covariance matrix. Therefore, our method can be applied to various 3D representations.

In the following section, we will present experimental results that demonstrate our method’s superior performance compared to existing approaches.

Figure: Algorithm 2 Differentiable Topology Estimation

## III Experiments

Figure: Figure 8: Comparison on noised parameterized surface samples.
Refer to caption: x8.png

In this section, we present the experimental results of our method to demonstrate its effectiveness and efficiency. Comparisons with existing methods show that our approach achieves superior performance across various 3D models for both curvature and topology estimation. The ablation study illustrates the impact of our proposed simplifications in the robust estimation of the Weingarten map and the self-optimization mechanism. Our method is implemented in PyTorch with CUDA (v2.0, CUDA 11.8) on Linux (Ubuntu 20.04). All experiments are conducted on a single NVIDIA RTX A6000 GPU with 48GB of memory and an AMD EPYC 7313P 16-core CPU.

### III-A Performance of Curvature Estimation

**TABLE I: Comparison of Curvature Estimation on Parametric Surface**
|  | Ellipsoid | Torus |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | $x^{2}+y^{2}+z^{2}=1$, 2k | $x^{2}+\frac{y^{2}}{2}+\frac{y^{2}}{4}=1$, 10k | 10k, +2.5%$\mathcal{N}$ | $R=5,r=1$, 10k | $R=5,r=3$, 70k | 70k, +2.5%$\mathcal{N}$ |  |  |  |  |  |  |  |  |  |  |  |  |
| Gaussian Curvature |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| log-Error | Max | Mean | Euler | Max | Mean | Euler | Max | Mean | Euler | Max | Mean | Euler | Max | Mean | Euler | Max | Mean | Euler |
| PCA | 8.020 | 7.450 | -7.2e+3 | 7.931 | 7.425 | -1.4e+4 | 9.035 | 8.391 | -5.6e+4 | 6.583 | 6.317 | -4.7e+4 | 9.376 | 8.886 | -2.0e+6 | 10.156 | 9.592 | -4.4e+6 |
| Taubin | 2.620 | -0.260 | 2.149 | 3.079 | -0.129 | 3.125 | 5.021 | 2.521 | 8.268 | 3.739 | 0.617 | 16.41 | 3.324 | -0.360 | 16.04 | 4.271 | 1.673 | 77.48 |
| Normal | -0.010 | -0.040 | 0.006 | 0.176 | -0.045 | 0.024 | 0.249 | -0.045 | 0.159 | -0.234 | -0.311 | 1.34 | -0.640 | -0.830 | 0.58 | -0.525 | -0.840 | 4.94 |
| Quadric | -1.980 | -2.540 | 2.005 | -0.295 | -0.583 | 0.502 | 3.790 | 1.618 | 6.976 | 2.775 | -0.234 | 16.05 | 2.527 | -0.795 | 15.66 | 3.899 | 0.728 | 54.28 |
| Our | -3.020 | -3.700 | 2.001 | 0.744 | -0.074 | 1.551 | 0.800 | -0.081 | 1.512 | -0.437 | -0.893 | -0.04 | -2.538 | -3.523 | -0.0002 | -1.038 | -2.420 | -0.016 |
| Mean Curvature |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| log-Error | Max | Mean | Time | Max | Mean | Time | Max | Mean | Time | Max | Mean | Time | Max | Mean | Time | Max | Mean | Time |
| PCA | 4.011 | 3.680 | 0.673 | 3.984 | 3.674 | 0.913 | 4.522 | 4.109 | 0.702 | 3.334 | 3.181 | 0.396 | 4.688 | 4.407 | 1.881 | 5.079 | 4.744 | 1.834 |
| Taubin | 1.247 | -0.845 | 0.458 | 1.555 | -0.505 | 0.302 | 2.279 | 0.888 | 2.279 | 1.628 | -0.144 | -0.136 | 1.469 | -0.692 | 1.205 | 1.915 | 0.448 | 1.008 |
| Normal | -0.001 | -0.001 | 0.451 | 0.301 | -0.096 | 0.323 | 0.353 | -0.114 | 0.394 | -0.005 | -0.876 | -0.112 | -0.304 | -1.272 | 1.038 | -0.101 | -1.037 | 1.011 |
| Quadric | -2.284 | -2.854 | 0.475 | -0.299 | -0.580 | 0.394 | 2.467 | 0.575 | 0.433 | 1.373 | -0.280 | -0.013 | 1.004 | -0.807 | 1.096 | 2.522 | 0.093 | 1.066 |
| Our | -3.146 | -3.916 | -3.042 | -0.387 | -1.764 | -3.013 | 0.364 | -0.716 | -3.017 | -0.169 | -0.336 | -3.410 | -0.339 | -0.538 | -2.439 | -0.297 | -0.538 | -2.434 |

We first evaluate the performance of the self-adjoint Weingarten map estimation method on the task of curvature estimation, comparing it with state-of-the-art methods, including the Taubin method [21], the robust statistical method [24], and the quadratic fitting method [19]. Since the theoretical curvature values can only be obtained from analytical surfaces, ground truth supervised evaluations are conducted solely on ellipsoids and tori with varying parameters, sampling densities, and noise levels. Estimation results for both Gaussian and mean curvatures are assessed by the maximum absolute error (Max), mean absolute error (Mean) over the entire surface, and the Euler characteristic (Euler) to measure error accumulation. The theoretical values of Gaussian and mean curvature are derived directly from the definitions of curvature in differential geometry.

**TABLE II: Comparison of Topology Estimation on 3D Models**
|  | Ground-Truth | 5k | 10k | 10k + 25% |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D Model Name | Euler | Genus | Taubin | Normal | Quadric | Ours | Taubin | Normal | Quadric | Ours | Taubin | Normal | Quadric | Ours |
| RindStrips | -12 | 7 | -187.574 | 42.150 | 24.451 | -12.721 | -364.580 | 18.763 | 31.421 | -12.281 | -513.549 | 24.296 | 34.491 | -11.316 |
| ChicagoLion | -4 | 3 | -134.305 | 11.153 | -39.028 | -2.243 | -231.445 | 4.101 | 4.344 | -3.891 | -261.315 | 4.984 | 22.265 | -7.979 |
| HolesSculpture | -402 | 202 | -255.650 | 0.000 | 8.211 | -380.564 | -338.778 | 0.000 | 14.388 | -403.540 | -211.279 | 0.000 | -1.280 | -195.819 |
| TriakisTetrahedron | -20 | 11 | -240.753 | 246.402 | -4.811 | -13.386 | -156.516 | 83.043 | -18.717 | -19.301 | -278.019 | 109.510 | -13.049 | -27.020 |
| OrganicSphere | 2 | 0 | -43.351 | 50.699 | 4.538 | 1.911 | -44.778 | 14.752 | 5.289 | 2.015 | -168.501 | 17.065 | 49.377 | 1.417 |
| Hilb64Thick | 0 | 1 | -97.563 | 904.153 | 11.984 | 1.641 | -168.468 | 271.507 | 10.889 | 0.707 | -277.612 | 290.248 | 5.732 | 2.586 |
| Quirrel | 2 | 0 | -33.761 | 200.322 | 3.842 | 1.911 | -27.654 | 52.086 | 4.752 | 2.293 | -76.149 | 74.805 | 4.702 | 1.348 |
| Icosphere | 2 | 0 | 0.802 | 0.000 | 2.006 | 1.985 | 1.038 | 0.000 | 1.967 | 1.997 | -332.351 | 0.003 | 3.132 | 1.952 |
| Knot | 0 | 1 | -38.954 | 14.771 | -0.928 | -0.854 | -34.389 | 3.175 | -0.895 | -0.376 | -96.946 | 6.737 | -1.898 | 0.125 |
| Venus | 2 | 0 | -119.564 | 2694.091 | 5.868 | 1.534 | -110.949 | 861.677 | 2.980 | 2.100 | -157.928 | 911.814 | 4.339 | 1.471 |
| Lcositetrahedron | -44 | 23 | -166.592 | 2102.811 | 6.868 | -42.751 | -229.629 | 698.427 | -16.535 | -43.262 | -256.594 | 700.772 | -17.060 | -35.146 |
| Kitten | 0 | 1 | -38.744 | 0.000 | 0.530 | 0.515 | -41.165 | 0.000 | 1.027 | 0.504 | -334.060 | 0.001 | 0.659 | -0.113 |
| TeaPot | 0 | 1 | -43.676 | 4507.909 | 6.331 | -0.473 | -60.219 | 2243.370 | 5.132 | -1.136 | -50.036 | 1586.369 | 4.848 | -2.123 |
| Bunny | 2 | 0 | -62.907 | 2181.226 | 8.352 | 1.897 | -128.493 | 836.904 | 3.647 | 1.898 | -109.532 | 728.377 | 14.158 | 1.370 |
| FatherStrength | -4 | 3 | -77.286 | 0.001 | 5.762 | -2.613 | -107.531 | 0.000 | 11.583 | -3.671 | -345.582 | 0.003 | 3.190 | -3.626 |
| QueenAnneChair | -2 | 2 | -90.733 | 73.852 | 0.369 | -1.214 | -98.113 | 23.431 | -5.794 | -2.324 | -263.281 | 31.454 | -2.391 | -1.519 |
| Art3dprint | -102 | 52 | -246.946 | 0.000 | -1.416 | -90.612 | -329.704 | 0.000 | -17.058 | -90.593 | -230.682 | 0.000 | 3.206 | -109.380 |
| Art3dprint2 | -18 | 10 | -170.480 | 0.000 | -8.640 | -17.703 | -282.578 | 0.000 | -14.653 | -17.545 | -237.336 | 0.000 | 5.192 | -12.158 |
| Mushroom | 0 | 1 | -106.634 | 2468.301 | -2.093 | -0.018 | -98.255 | 733.907 | 2.477 | 0.429 | -135.657 | 654.618 | 6.820 | -2.199 |
| Genus6surface | -10 | 6 | -195.469 | 38.372 | -7.271 | -9.204 | -117.047 | 11.750 | -11.734 | -10.072 | -284.386 | 13.549 | -9.051 | -9.626 |

Specifically, for an ellipsoid $\frac{x^{2}}{a^{2}}+\frac{y^{2}}{b^{2}}+\frac{z^{2}}{c^{2}}=1$, we have

$$ $\displaystyle K_{g}(x,y,z)=\frac{1}{a^{2}b^{2}c^{2}\left(\frac{x^{2}}{a^{4}}+ \frac{y^{2}}{b^{4}}+\frac{z^{2}}{c^{4}}\right)^{2}},$ (14) $\displaystyle H(x,y,z)=\frac{x^{2}+y^{2}+z^{2}-a^{2}-b^{2}-c^{2}}{2(a^{2}b^{2} c^{2})\sqrt{\left(\frac{x^{2}}{a^{4}}+\frac{y^{2}}{b^{4}}+\frac{z^{2}}{c^{4}} \right)^{3}}}.$ $$

For a torus parameterized by

$$ $\left\{\begin{array}[]{l l}x=(R+r\cos(u))\cos(v),\\ y=(R+r\cos(u))\sin(v),\\ z=r\sin(u),\end{array}\right.$ (15) $$

we obtain the curvatures as

$$ $\displaystyle K_{g}=\frac{\cos(v)}{r(R+r\cos(u))},$ (16) $\displaystyle H=\frac{1}{2}\left(\frac{1}{r}+\frac{\cos(v)}{R+r\cos(u)}\right).$ $$

The results are presented in Table [I](https://arxiv.org/html/2412.00140v1#S3.T1), with errors recorded on a logarithmic scale. The comparison shows that our method outperforms state-of-the-art methods for both Gaussian and mean curvature estimation across different sampling densities and noise levels. Notably, our method uniquely achieves Euler characteristic estimation and resists error accumulation, showing insensitivity to noise. Additionally, by leveraging parallel computation, our method is significantly faster than traditional approaches, particularly for high-density point clouds or high-resolution implicit fields.

For general 3D models, where ground truth curvature is difficult to obtain, we indirectly evaluate curvature estimation via topology estimation. Additionally, curvature-aware decimation of point clouds and surface reconstruction will further demonstrate the effectiveness of our curvature estimation, as discussed in the following subsections.

### III-B Performance on Topology Estimation

First, we evaluate the performance of our method on the topology estimation task, comparing it with the curvature-based methods introduced in the previous subsection. This comparison is conducted on 20 3D models with varying topologies and geometric complexities from the SHREC dataset [33], the ModelNet40 dataset [34], and the ShapeNet dataset [35]. The results, presented in Table [II](https://arxiv.org/html/2412.00140v1#S3.T2), demonstrate that our method achieves an average accuracy of nearly 90% for topology estimation, significantly outperforming state-of-the-art methods.

Additionally, our experiments indicate that our method is robust to different sampling schemes and noise density. Uniform sampling generally poses greater implementation challenges as it requires careful attention to local surface areas. Through self-optimization during integration, our method obtains correct topology estimations with random sampling and added noise, as illustrated in Fig. [9](https://arxiv.org/html/2412.00140v1#S3.F9) and the final column in Table [II](https://arxiv.org/html/2412.00140v1#S3.T2).

Figure: Figure 9: Comparison on parameterized surface samples with noise.
Refer to caption: x9.png

On the other hand, we compare the efficiency of our method with the topological data analysis (TDA) method, specifically persistent homology [36, 37, 38], which is the prevailing approach for topology estimation. Our comparison shows that our method is not only significantly faster than the TDA method but also less ambiguous.
Persistent homology (PH) constructs a 2D Vietoris-Rips complex [39] and generates a persistence diagram that represents the topological features of a shape. This diagram captures information about the 0 0-dimensional and $1$-dimensional homology groups: the 0 0-dimensional group corresponds to the number of connected components, while the $1$-dimensional group reveals the genus (the number of holes). In PH, each segment on the resulting barcode corresponds to an independent topological feature, with a long-lasting 0 0-dimensional bar indicating a stable connected component and a persistent $1$-dimensional bar signifying a stable hole.

Interpreting topology from these barcodes, however, can be challenging due to scaling differences across different genera.
For instance, Fig. [10](https://arxiv.org/html/2412.00140v1#S3.F10) shows the persistence diagrams’ barcodes for the topology estimation of the Kitten (connected, 1-genus) and ChicagoLion (connected, 3-genus). Both models exhibit a single, long-standing 0 0-bar. While the Kitten has a unique long-lasting $1$-bar, the ChicagoLion shows more than three components, which makes it hard to align with its known topological structures. Additionally, PH is computationally intensive; in our experiments, PH required 49.86 seconds to process a 7k-point cloud. In contrast, our method directly approximates the genus with minimal ambiguity, which is effectively resolved through a self-optimization mechanism. Furthermore, unlike the iterative loops in PH, which are difficult to parallelize, our approach is fully differentiable and can be efficiently implemented on GPUs.

Figure: Figure 10: Barcode of Persistence diagram
Refer to caption: x10.png

## IV Conclusion

In this paper, we propose a differentiable algorithm to estimate the curvature and global topology of 3D shapes. Curvature estimation provides an important intrinsic geometric feature of 3D shapes. Our instant topology estimation, achieved for the first time, provides a reliable initialization for mesh-deformation-based reconstruction and generation tasks. Differentibility allows our method to be integrated into deep learning frameworks, enabling end-to-end training of neural networks for shape analysis tasks. We demonstrate the effectiveness of our method on both synthetic and real-world datasets.

## References

- [1]
K. Genova, F. Cole, D. Vlasic, A. Sarna, W. T. Freeman, and T. Funkhouser, “Learning shape templates with structured implicit functions,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2019, pp. 7154–7164.
- [2]
J. J. Park, P. Florence, J. Straub, R. Newcombe, and S. Lovegrove, “Deepsdf: Learning continuous signed distance functions for shape representation,” in *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, 2019, pp. 165–174.
- [3]
Z. Marschner, S. Sellán, H.-T. D. Liu, and A. Jacobson, “Constructive solid geometry on neural signed distance fields,” in *SIGGRAPH Asia 2023 Conference Papers*, 2023, pp. 1–12.
- [4]
J. Zhang, Y. Yao, and L. Quan, “Learning signed distance field for multi-view surface reconstruction,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, pp. 6525–6534.
- [5]
B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, “Nerf: Representing scenes as neural radiance fields for view synthesis,” *Communications of the ACM*, vol. 65, no. 1, pp. 99–106, 2021.
- [6]
T. Müller, A. Evans, C. Schied, and A. Keller, “Instant neural graphics primitives with a multiresolution hash encoding,” *ACM transactions on graphics (TOG)*, vol. 41, no. 4, pp. 1–15, 2022.
- [7]
B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, “Dreamfusion: Text-to-3d using 2d diffusion,” *arXiv preprint arXiv:2209.14988*, 2022.
- [8]
B. Kerbl, G. Kopanas, T. Leimkühler, and G. Drettakis, “3d gaussian splatting for real-time radiance field rendering.” *ACM Trans. Graph.*, vol. 42, no. 4, pp. 139–1, 2023.
- [9]
T. Wu, Y.-J. Yuan, L.-X. Zhang, J. Yang, Y.-P. Cao, L.-Q. Yan, and L. Gao, “Recent advances in 3d gaussian splatting,” *Computational Visual Media*, vol. 10, no. 4, pp. 613–642, 2024.
- [10]
G. Chen and W. Wang, “A survey on 3d gaussian splatting,” *arXiv preprint arXiv:2401.03890*, 2024.
- [11]
B. Huang, Z. Yu, A. Chen, A. Geiger, and S. Gao, “2d gaussian splatting for geometrically accurate radiance fields,” in *ACM SIGGRAPH 2024 Conference Papers*, 2024, pp. 1–11.
- [12]
V. Pascucci, G. Scorzelli, P.-T. Bremer, and A. Mascarenhas, “Robust on-line computation of reeb graphs: simplicity and speed,” in *ACM SIGGRAPH 2007 papers*, 2007, pp. 58–es.
- [13]
J. W. Milnor, *Morse theory*.   Princeton university press, 1963, no. 51.
- [14]
H. Edelsbrunner and J. Harer, “Computational topology,” *Duke University*, 2008.
- [15]
Edelsbrunner, Letscher, and Zomorodian, “Topological persistence and simplification,” *Discrete & computational geometry*, vol. 28, pp. 511–533, 2002.
- [16]
G. Carlsson, “Topology and data,” *Bulletin of the American Mathematical Society*, vol. 46, no. 2, pp. 255–308, 2009.
- [17]
H.-H. Wu, “Historical development of the gauss-bonnet theorem,” *Science in China Series A: Mathematics*, vol. 51, no. 4, p. 777, 2008.
- [18]
Q. Mérigot, M. Ovsjanikov, and L. J. Guibas, “Voronoi-based curvature and feature estimation from point clouds,” *IEEE Transactions on Visualization and Computer Graphics*, vol. 17, no. 6, pp. 743–756, 2010.
- [19]
M. Wardetzky, M. Bergou, D. Harmon, D. Zorin, and E. Grinspun, “Discrete quadratic curvature energies,” *Computer Aided Geometric Design*, vol. 24, no. 8-9, pp. 499–518, 2007.
- [20]
D. Panozzo, E. Puppo, and L. Rocca, “Efficient multi-scale curvature and crease estimation,” in *2nd International Workshop on Computer Graphics, Computer Vision and Mathematics, GraVisMa*, 2010, pp. 9–16.
- [21]
G. Taubin, “Estimating the tensor of curvature of a surface from a polyhedral approximation,” in *Proceedings of IEEE International Conference on Computer Vision*.   IEEE, 1995, pp. 902–907.
- [22]
Y. Cao, D. Li, H. Sun, A. H. Assadi, and S. Zhang, “Efficient weingarten map and curvature estimation on manifolds,” *Machine Learning*, vol. 110, no. 6, pp. 1319–1344, 2021.
- [23]
J. Berkmann and T. Caelli, “Computation of surface geometry and segmentation using covariance techniques,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 16, no. 11, pp. 1114–1116, 1994.
- [24]
E. Kalogerakis, P. Simari, D. Nowrouzezahrai, and K. Singh, “Robust statistical estimation of curvature on discretized surfaces,” in *Symposium on Geometry Processing*, vol. 13, 2007, pp. 110–114.
- [25]
P. Yang and X. Qian, “Direct computing of surface curvatures for point-set surfaces.” in *PBG@ Eurographics*.   Citeseer, 2007, pp. 29–36.
- [26]
N. Amenta and Y. J. Kil, “The domain of a point set surfaces,” in *Eurographics Symposium on Point-based Graphics*, vol. 1, 2004.
- [27]
F. Aurenhammer, “Voronoi diagrams—a survey of a fundamental geometric data structure,” *ACM Computing Surveys (CSUR)*, vol. 23, no. 3, pp. 345–405, 1991.
- [28]
R. Xu, Z. Dou, N. Wang, S. Xin, S. Chen, M. Jiang, X. Guo, W. Wang, and C. Tu, “Globally consistent normal orientation for point clouds by regularizing the winding-number field,” *ACM Transactions on Graphics (TOG)*, vol. 42, no. 4, pp. 1–15, 2023.
- [29]
W. Chen, S.-s. Chern, and K. S. Lam, *Lectures on differential geometry*.   World Scientific Publishing Company, 1999, vol. 1.
- [30]
A. Ward, “A general analysis of sylvester’s matrix equation,” *International Journal of Mathematical Education in Science and Technology*, vol. 22, no. 4, pp. 615–620, 1991.
- [31]
Y. Luo, S. Zhang, Y. Cao, and H. Sun, “Geometric characteristics of the wasserstein metric on spd (n) and its applications on data processing,” *Entropy*, vol. 23, no. 9, p. 1214, 2021.
- [32]
S. Zhang, Y. Cao, W. Li, F. Yan, Y. Luo, and H. Sun, “A new riemannian structure in spd (n),” in *2019 IEEE International Conference on Signal, Information and Data Processing (ICSIDP)*.   IEEE, 2019, pp. 1–5.
- [33]
B. Li, Y. Lu, C. Li, A. Godil, T. Schreck, M. Aono, M. Burtscher, Q. Chen, N. K. Chowdhury, B. Fang *et al.*, “A comparison of 3d shape retrieval methods based on a large-scale benchmark supporting multimodal queries,” *Computer Vision and Image Understanding*, vol. 131, pp. 1–27, 2015.
- [34]
Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, “3d shapenets: A deep representation for volumetric shapes,” in *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2015, pp. 1912–1920.
- [35]
A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su *et al.*, “Shapenet: An information-rich 3d model repository,” *arXiv preprint arXiv:1512.03012*, 2015.
- [36]
H. Edelsbrunner, “Persistent homology: theory and practice,” 2013.
- [37]
A. Zomorodian and G. Carlsson, “Computing persistent homology,” in *Proceedings of the twentieth annual symposium on Computational geometry*, 2004, pp. 347–356.
- [38]
N. Otter, M. A. Porter, U. Tillmann, P. Grindrod, and H. A. Harrington, “A roadmap for the computation of persistent homology,” *EPJ Data Science*, vol. 6, pp. 1–38, 2017.
- [39]
J.-C. Hausmann *et al.*, *On the Vietoris-Rips complexes and a cohomology theory for metric spaces*.   Université de Genève-Section de mathématiques, 1994.