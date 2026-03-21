## Contents
- 1 Introduction
- 2 Empirical Validation
  - 2.1 The Vocabulary of Time Series
    - 2.1.1 Vocabulary Construction
    - 2.1.2 Vocabulary Statistics
      - Cluster Structure and Separability.
      - Zipfian Frequency Distribution.
      - Holistic Performance and Vocabulary Diversity.
      - Evidence for Distributional Tokens.
  - 2.2 The Quasi-Linguistic Properties of Time Series
    - 2.2.1 The Discovery of a Zipfian Distribution in the Temporal Lexicon
    - 2.2.2 Robustness and Dynamic Adaptability of the Vocabulary Structure
    - 2.2.3 Quantitative Confirmation of the Structure’s Intrinsic Nature
  - 2.3 The Grammar of Time Series
    - The Principle of State Inertia.
    - A Highly Structured and Sparse Language.
    - Macroscopic Diversity from Microscopic ”Chunking”.
- 3 Theoretical Foundation
  - 3.1 Feasibility and Structure of the Temporal Vocabulary
    - Fidelity of Representation.
      - Theorem 3.1 ( ε 𝜀 \varepsilon italic_ε - Covering Guarantees Bounded Information Loss) .
    - Statistical Structure of the Vocabulary.
      - Theorem 3.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .
  - 3.2 Representational Power and Generalization Guarantees
    - Expressiveness of Patch Representations.
      - Theorem 3.3 (Capacity Measure Monotonicity) .
    - Generalization on Dependent Data.
      - Lemma 3.1 ( β 𝛽 \beta italic_β -Mixing Preservation) .
      - Theorem 3.4 (Dependence Generalisation Bound) .
  - 3.3 The Information-Theoretic Advantage of Patching
    - Theorem 3.5 (Patch Representation as an Effective Information Bottleneck) .
- 4 Conclusion
- References
- Appendix A Theoretical Analysis
  - A.1 ϵ italic-ϵ \epsilon italic_ϵ - Statistical Sufficiency and Zipf-like Long-Tail Frequency of Patch Tokens
    - A.1.1 ε 𝜀 \varepsilon italic_ε - Statistical Sufficiency
      - Theorem A.1 ( ε 𝜀 \varepsilon italic_ε - Covering Guarantees Bounded Information Loss) .
      - Proof.
      - Remark A.1 (On the Practical Construction of the Codebook) .
    - A.1.2 Zipf-like Long-Tail Frequency
      - Theorem A.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .
      - Proof.
      - Remark A.2 (Connection to Zipf’s Law in Linguistics) .
  - A.2 Non-decreased Representational Capacity and Non-increased Optimal-ERM Risk
    - A.2.1 Monotone Representational Capacity
      - Definition A.1 (Pointwise Hypothesis Space) .
      - Definition A.2 (Patch Hypothesis Space) .
      - Assumption A.1 .
      - Lemma A.1 (Idealized Subclass Relation) .
      - Proof.
      - Theorem A.3 (Capacity Measure Monotonicity) .
      - Proof.
      - Corollary A.1 (Capacity Monotonicity for Patch Methods) .
      - Remark A.3 (Computational Complexity) .
      - Remark A.4 (Generalization Bounds) .
      - Remark A.5 (Practical Trade-offs) .
      - Proposition A.1 (Extension to Other Measures) .
      - Remark A.6 (Connection to PatchTST) .
    - A.2.2 Non-increased Optimal-ERM Risk
      - Corollary A.2 (Optimal-ERM Risk Non-Increase) .
      - Proof.
  - A.3 A Rigorous Bound for Token Sequence Dependence
    - Definition A.3 (Patch Construction) .
    - Definition A.4 (Quantization) .
    - Lemma A.2 (Patch β 𝛽 \beta italic_β -Mixing Preservation) .
    - Proof.
    - Remark A.7 (Non-overlapping Condition) .
    - Remark A.8 (Overlapping Case) .
    - Remark A.9 (Quantization Independence) .
    - Theorem A.4 (Dependence Generalisation Bound) .
    - Proof.
  - A.4 Non-Decreasing Task-Relevant Mutual Information
    - Theorem A.5 (Patch Representation as an Effective Information Bottleneck) .
    - Proof.
    - Remark A.10 .
    - Theorem A.6 ( ε 𝜀 \varepsilon italic_ε -MI Preservation via Lipschitz Continuity) .
    - Proof.
- Appendix B Datasets Overview
- Appendix C Detailed Dataset Descriptions
  - C.1 Environmental Domain Datasets
  - C.2 Energy Domain Datasets
  - C.3 Transportation Domain Datasets
  - C.4 Financial Domain Datasets
  - C.5 Healthcare Domain Datasets
  - C.6 Astronomical Domain Datasets
- Appendix D Dataset Statistics Summary
- Appendix E Qualitative Analysis of the Learned Temporal Vocabulary
  - A Hierarchy from Simple States to Complex Events.
  - Qualitative Validation of the Linguistic Analogy.

## Abstract

Abstract With the rise of large language models, the paradigm of training foundation models with massive parameter counts on vast datasets has been adopted in multiple domains to achieve remarkable success. Time series foundation models represent a significant extension of this paradigm, demonstrating exceptional expressive power, generalization, and cross-domain transferability. However, this gives rise to a fundamental paradox: time series data reflect distinct dynamical systems, making cross-domain transfer intuitively implausible, yet this is contradicted by the models’ empirical success.
To resolve this paradox, this paper investigates, from both theoretical and experimental perspectives, the representation learning mechanisms and generalization capabilities of patch-based time series foundation models. We argue that such models are not merely applying a new architecture but are fundamentally generalizing the representation paradigm of language models by extending deterministic vector-based representations to latent probabilistic distributional forms . Our theoretical analysis supports this framework by demonstrating that continuous time-series patches can be faithfully quantized into a discrete vocabulary whose key statistical properties are highly consistent with those of natural language.
This generalization allows time series models to inherit the robust representation and transfer abilities of large language models, thereby explaining their superior performance in temporal tasks. Ultimately, our work provides a rigorous theoretical cornerstone for understanding, evaluating, and improving the safety and reliability of large-scale time series foundation models.

## 1 Introduction

Time series data chronicle the evolution of complex systems through sequences of numerical observations sampled at uniform intervals, offering a quantitative fingerprint of their dynamics Montgomery et al. (2015). The inherent characteristics of these data—most notably temporal dependencies, underlying trends, and seasonal cycles—make them invaluable for a vast array of applications, including traffic flow forecasting, logistics optimization, climate change analysis, and human mobility modeling Zheng & Huang (2020); Xie et al. (2023); Fu et al. (2024); Barbosa et al. (2018).

Mirroring the paradigm shift driven by large language models, time-series foundation models have recently achieved remarkable success through large-scale pre-training and fine-tuning Ansari et al. (2024); Goswami et al. (2024); Das et al. (2024b). By pre-training on massive and diverse corpora, often encompassing billions of temporal data points, these models can be adapted using zero-shot or few-shot strategies. This has led to substantial improvements in forecasting accuracy, robust cross-domain generalization, and impressive performance even with limited data Ye et al. (2024); Das et al. (2024a).

Yet, this empirical success stands in stark contrast to a growing body of critical analysis questioning the internal mechanisms, the true efficacy of domain transfer, and the fundamental in-context learning capabilities of these models Zhang et al.; Ding et al. (2025); He et al. (2023); Zhao et al. (2024). The core challenge is clear: each time series represents a unique system with its own distinct temporal patterns. Consequently, transferring a model between disparate domains—for instance, from energe consumption to climate science—inevitably incurs a significant distributional shift. This chasm between what these models can do and why they should work raises fundamental questions about their safety, reliability, and theoretical underpinnings. Are their impressive feats merely an over-fitting coincidence of massive data and computation, or can they be anchored in a rigorous theoretical foundation?

In this paper, we bridge this gap from a theoretical standpoint. We argue that a patch-embedding-based time series foundation model can be formally understood as a generalization of a large language model—one that operates not on discrete tokens, but on token distributions.

Our core argument lies in re-conceptualizing the fundamental unit of input: while language models process discrete tokens (words), time-series models should treat patches—short temporal segments—as their basic unit. Unlike words that map to isolated points in latent space, time-series patches correspond to families of patterns, or recurring temporal motifs Mueen (2014); Lonardi & Patel (2002). For instance, a gradual decrease motif may manifest as variants with differing slopes or noise levels (see Figure [1](https://arxiv.org/html/2507.00078v1#S1.F1)), which, despite numerical differences, belong to the same conceptual family. Consequently, their embeddings should form distributions in latent space rather than single points (see Figure [2](https://arxiv.org/html/2507.00078v1#S1.F2)). Notably, motifs exhibit significant co-occurrence relationships: two peaks necessarily imply an intervening trough (and vice versa); sharp upward trends are likely followed by decay or correction phases, forming steep rise-sudden drop co-occurrence patterns; or certain motifs appear in pairs to constitute complete cycles, creating rising edge-falling edge binding relationships. Beyond these intuitive examples, there exist numerous other motif relationships that defy verbal description yet exist in practice—a phenomenon remarkably analogous to lexical co-occurrence in language models.

Figure: Figure 1: Visualization of similar time-series segments and their alignment. Left: Three highlighted patches (Patch A/B/C) in the full signal share an identical trend shape despite differing amplitudes. Right: After Z-score normalization and temporal alignment, the curves almost overlap, indicating that the segments belong to the same latent temporal motif.
Refer to caption: x1.png

Figure: Figure 2: Distributed embeddings of language tokens versus time-series patches. Left: In a language model, token embeddings appear as discrete, sparsely located singleton points. Right: In a time-series model, patch embeddings form probability clouds with finite thickness; patches of the same motif (Pattern A/B) cluster into separable yet internally continuous regions, illustrating the concept of a “distributional token.”
Refer to caption: x2.png

Furthermore, these pattern families are not strictly partitioned or mutually exclusive; a single patch might simultaneously exhibit characteristics of multiple motifs, leading to overlapping latent-vector distributions. Crucially, we posit that this extension from point-wise representations to distributional ones is what allows the model to inherit the expressive power and powerful generalization capabilities of LLMs. It is this ability to learn an abstract vocabulary of continuous temporal patterns, rather than memorizing specific numerical sequences, that provides a rigorous theoretical justification for the observed success of patch-embedding-based time-series foundation models.

To substantiate our proposed ”distributional token” hypothesis, this paper unfolds an in-depth investigation from both empirical and theoretical perspectives. We conduct a series of carefully designed empirical studies aimed at validating the key assumptions of our theoretical framework and demonstrating their manifestation in real-world data. Concurrently, through a set of rigorous and hierarchical theoretical derivations, we establish a solid mathematical foundation for this novel perspective, transforming intuitive insights into provable conclusions.

Our main contributions can be summarized as follows:

- •
Empirical Discovery of ”Quasi-Linguistic” Properties of Time: We are the first to show, through large-scale empirical analysis, that after patches extracted from diverse time-series datasets are quantized into tokens, their frequency distribution strikingly follows a Zipf-like law. This provides strong statistical evidence for the concept of a ”language of time.” Furthermore, our experiments validate the natural denoising effect of the patching operation across various tasks.
- •
Construction of a Hierarchical Theoretical Framework: We build a complete theoretical analysis framework to support our claims. This framework:
–
Establishes the fidelity of representing continuous patches with discrete tokens, starting from covering number theory.
–
Proves that the patch-based hypothesis space has non-decreasing representational capacity, using principles from learning theory to ensure model expressiveness.
–
Leverages the Information Bottleneck principle to reveal how patching acts as an effective compression scheme that filters out noise while preserving task-relevant information.
–
Finally, by proving the key property of dependence preservation, it connects our framework with generalization theory for dependent sequences, thereby providing guarantees for the model’s generalization ability.
- •
A Bridge Between Theory and Practice: We clearly elucidate the link between theory and practice by demonstrating why patch-based methods naturally achieve pattern abstraction. This explains the fundamental reason for the successful cross-domain transfer of time-series foundation models: they learn a universal and composable vocabulary of temporal dynamic ”motifs,” rather than memorizing domain-specific numerical sequences.

## 2 Empirical Validation

Our central hypothesis posits that time series data harbors a deep statistical structure analogous to that of natural language. To empirically validate this claim, we conducted a large-scale experiment to investigate whether non-trivial statistical laws emerge when continuous temporal dynamics are symbolized into a discrete vocabulary. This was achieved by applying K-Means clustering to a vast dataset of 38k time series patches from diverse domains, thereby quantizing continuous patterns into a vocabulary of ”Temporal Words.” The objective was to test whether the usage patterns of this data-driven vocabulary conform to the same statistical principles that govern human language.

### 2.1 The Vocabulary of Time Series

#### 2.1.1 Vocabulary Construction

Our central hypothesis posits that time series data harbors a deep statistical structure analogous to that of natural language. To empirically validate this claim, the primary task is to transform the continuous, high-dimensional time-series signals into a discrete, analyzable representation composed of symbols. The core of this transformation lies in constructing a ”Vocabulary of Time Series”. number of “tokens” or “words” in the time series.

Figure: Figure 3: Comprehensive analysis of clustering performance reveals a core trade-off governed by patch size ($P$). The visualizations (line plots, heatmaps, and radar chart) collectively demonstrate that smaller patches (e.g., $P=16$) excel at forming structurally distinct clusters (high Silhouette Score), whereas larger patches ($P\geq 64$) create vocabularies with greater linguistic plausibility (stronger conformity to Zipf’s Law). This shows that patch size is a fundamental design choice, balancing the structural clarity of “atomic” patterns against the semantic richness of a more language-like temporal vocabulary.
Refer to caption: x3.png

Nevertheless, this vocabulary is not formed from predefined functions or rules but is generated entirely in a data-driven manner. We envision that complex dynamic phenomena arise from the composition of a finite, reusable set of fundamental patterns or ”Temporal Motifs”. Therefore, the construction process for this vocabulary aims to automatically discover and distill a representative set of such pattern prototypes from vast and diverse time-series data.

Specifically, our construction pipeline is as follows:

- 1.
Patching: We segment the raw time series into continuous segments of length $P$ with a stride of $S$, referred to as ”patches”. Each patch serves as the fundamental unit for our analysis, encapsulating a segment of local dynamic information.
- 2.
Quantization: To map these continuous, high-dimensional patch vectors into a finite, discrete symbol space, we employ the technique of Vector Quantization. In this study, we select the K-Means clustering algorithm as our core quantization tool. This choice is predicated on two principal advantages:
(1) Intuitive Prototype Discovery: The K-Means algorithm partitions the data by finding $K$ ”centroids”. Each centroid is itself a vector of the same dimension as the input patches and can be intuitively interpreted as a standard, clean ”pattern prototype”.
(2) Computational Efficiency: We utilize its Mini-Batch variant, ensuring that the method can be efficiently scaled to process massive datasets, a critical requirement for training foundation models.

By executing K-Means clustering on the entire dataset of 38k patch samples, we obtain a ”codebook” or vocabulary $\mathcal{C}$ composed of $K$ centroids. Each entry in this vocabulary—that is, each centroid—constitutes a ”Temporal Word”, representing a fundamental dynamic pattern learned from the data. Subsequently, any given patch from the original dataset can be mapped to a specific index (ID) in this vocabulary by identifying its nearest centroid in Euclidean space.

Ultimately, a continuous time series is successfully transformed into an integer sequence of discrete ”token” IDs. This symbolic representation not only significantly compresses the data and filters out noise but, more critically, lays the groundwork for our subsequent statistical analysis. In the following sections, we will conduct an in-depth analysis of the frequency of use of this generated ”vocabulary of time” to test whether it truly exhibits the quasi-linguistic properties we hypothesize.

Following our experimental setup, we conducted a comprehensive analysis to evaluate the properties of the generated vocabularies. The results, summarized in the ”Multi-Patch Size Clustering Analysis” figure, reveal critical insights into the relationship between tokenization parameters and vocabulary quality. Our analysis of these results reveals a fundamental trade-off between the structural quality of the learned vocabulary and its statistical resemblance to natural language.

The experimental results present several clear trends:

- •
Structural Fidelity: Metrics for clustering quality consistently favor smaller patch sizes. ‘Patch 16‘ achieves the highest Silhouette Score and the lowest, or best, Davies-Bouldin Score. As patch size increases from 16 to 256, the maximum achievable Silhouette Score monotonically decreases. For all patch sizes, the optimal vocabulary size for maximizing clustering quality is consistently $K=16$.
- •
Linguistic Plausibility: The conformity to Zipf’s Law shows a more complex relationship. Larger patch sizes, such as $P=64$ and $P=256$, demonstrate the ability to achieve the lowest (best) deviation from an ideal Zipfian distribution. The optimal K for achieving this is higher than for clustering, typically falling at $K=32$ or $K=48$ for different patch sizes.

The opposing trends in these metrics point to a foundational trade-off in designing time-series vocabularies.

- 1.
Small patches ($P=16$) excel at creating a vocabulary of high structural fidelity. These short segments represent simple, ”atomic” patterns that are less varied and lower-dimensional. This makes it easier for the K-Means algorithm to partition them into well-defined, compact, and clearly separated clusters, resulting in superior scores on clustering metrics.
- 2.
Large patches ($P\geq 64$) are more adept at forming a vocabulary with high linguistic plausibility. These longer segments can capture more complete, semantically rich ”temporal motifs.” A vocabulary composed of these more complex patterns is more diverse and better mirrors the structure of natural language, where a vast number of specific, low-frequency words create a characteristic ”long-tail” distribution that conforms to Zipf’s Law.

A crucial observation from our analysis is that the cluster compactness, as measured by the Davies-Bouldin Score, is suboptimal across all tested configurations. The scores remain relatively high (where lower indicates better compactness), suggesting that the identified clusters are inherently diffuse rather than tightly concentrated.
This lack of high compactness is not interpreted as a failure of the clustering algorithm but rather as a reflection of an intrinsic property of the data itself. It indicates that the boundaries between different temporal motifs are not sharply defined. This observation aligns with our intuition that a single time-series patch is rarely a pure exemplar of one motif. For instance, a segment may exhibit a primary ”upward trend” while also containing secondary ”high-frequency volatility.” Consequently, its vector representation would naturally lie in the overlapping boundary regions between distinct clusters, reducing the measured compactness of both.
Ultimately, this empirical finding lends strong support to our ”distributional token” hypothesis. The observed looseness of the clusters can be seen as the physical manifestation of an underlying probabilistic representation, where each patch is not a discrete point but rather a distribution that may span multiple conceptual motifs.

The radar chart provides a holistic visualization of this trade-off. It clearly shows that smaller patch sizes (e.g., ‘Patch 16‘) dominate the axes related to clustering quality, while larger patch sizes are more competitive on the axis for Zipf Conformity. Given this finding, we will proceed with the subsequent experiments using $P=16$ and $K=32$ by default. However, it is important to note that this does not imply the chosen parameters are optimal; after all, unlike language models, we do not have a well-defined or fixed vocabulary for time series.

The choice of patch size is not a simple optimization problem but a fundamental design decision that dictates the nature of the learned vocabulary. A model designer must choose between a vocabulary of simple, structurally clear ”atomic” patterns (achieved with small patches) or a vocabulary of complex, semantically rich ”motifs” that exhibit more language-like statistical properties (achieved with larger patches). This choice directly impacts the subsequent learning paradigm of any foundation model built upon this tokenization scheme.

#### 2.1.2 Vocabulary Statistics

Figure: Figure 4: Comprehensive Statistical Analysis of the Temporal Vocabulary (P=16, K=32). This figure presents a multi-faceted analysis of a ”temporal vocabulary” constructed by applying K-Means clustering (K=32) to time series patches of length 16. (Top-Left) The PCA visualization reveals that patches form distinct yet overlapping clusters, visually representing the concept of separable but continuous temporal motifs. (Top-Center & Top-Right) The cluster size distribution exhibits a classic long-tail structure. When plotted on a log-log scale, this distribution demonstrates a striking conformity to Zipf’s Law (Deviation = 0.026), providing strong quantitative evidence for the quasi-linguistic nature of the data. (Bottom-Left) The performance radar chart offers a holistic assessment, showing a strong balance between traditional clustering quality metrics (e.g., Silhouette Score) and the vocabulary’s linguistic plausibility (Zipf Conformity). (Bottom-Center & Bottom-Right) The distribution of inter-cluster distances confirms a diverse vocabulary of patterns. Concurrently, the Within-Cluster Sum of Squares (WCSS) highlights the significant internal variance of certain motifs, supporting the ”distributional token” hypothesis where each token represents a family of related patterns rather than a single point. Collectively, these analyses provide a detailed statistical portrait, confirming that for the P=16, K=32 configuration, the tokenized time series data exhibits a robust, language-like structure.
Refer to caption: x4.png

Figure [4](https://arxiv.org/html/2507.00078v1#S2.F4) provides a comprehensive statistical analysis of the temporal vocabulary generated by applying K-Means clustering with parameters set to a patch size of $P=16$ and a vocabulary size of $K=32$. The results offer compelling, multi-faceted evidence for our central hypothesis: that tokenized time series data exhibits a robust, language-like structure.

##### Cluster Structure and Separability.

The PCA visualization (top-left panel) reveals the geometric distribution of the patch embeddings after being projected onto their first two principal components. The clusters, denoted by distinct colors, form visually coherent groups that are partially separable, indicating that the K-Means algorithm successfully identified meaningful, recurring patterns. However, the significant overlap between clusters provides initial support for our “distributional token” hypothesis, suggesting that temporal motifs are not discrete, isolated points but rather continuous regions in the latent space.

##### Zipfian Frequency Distribution.

The most striking finding is the vocabulary’s adherence to Zipf’s Law. The cluster size distribution (top-center panel) clearly shows a long-tail characteristic, where a few “temporal words” are exceedingly common, while the vast majority are rare. This observation is rigorously quantified in the log-log rank-frequency plot (top-right panel). The empirical data points (blue line) align remarkably well with the ideal Zipfian distribution (red dashed line), yielding a Zipf exponent of $1.025$ with a minimal deviation of $0.026$. This strong power-law signature is a hallmark of natural language and provides powerful empirical validation that complex temporal dynamics are composed from a vocabulary of reusable motifs governed by language-like statistical principles.

##### Holistic Performance and Vocabulary Diversity.

The performance radar chart (bottom-left) offers a synthesized view of the vocabulary’s quality, demonstrating a strong balance between structural fidelity (as measured by the Silhouette, CH, and DB scores) and linguistic plausibility (Zipf Conformity). This indicates that the chosen parameters produce a vocabulary that is both well-structured and statistically sound. Furthermore, the analysis of inter-cluster distances (bottom-center) shows a wide distribution, confirming that the learned vocabulary is diverse, consisting of distinct and well-differentiated temporal patterns.

##### Evidence for Distributional Tokens.

Finally, the Within-Cluster Sum of Squares (WCSS) plot (bottom-right) provides further evidence for the distributional nature of temporal tokens. The high WCSS values for several clusters (highlighted in red) are not indicative of poor clustering but rather reflect the high intrinsic variance of those specific motifs. This suggests that a single cluster centroid represents a family of similar, but not identical, temporal patterns (e.g., “a sharp rise” with varying slopes and noise levels). This observation reinforces the idea that a token is better understood as a probability distribution over a region of the latent space, rather than a single point vector.

In summary, the collective results presented in Figure [4](https://arxiv.org/html/2507.00078v1#S2.F4) establish a solid empirical foundation for viewing time series through a linguistic lens. The discovery of a robust, Zipf-like statistical structure, combined with evidence for distributional representations, provides a fundamental justification for the success of applying large language model paradigms to the time series domain.

### 2.2 The Quasi-Linguistic Properties of Time Series

Figure: Figure 5: Statistical Analysis of the Time Series Vocabulary. This figure presents a multi-dimensional analysis of the frequency distribution of ”tokens” derived from K-Means clustering on 38,000 time series patches (K=16 to 512). (Top-left) The log-log rank-frequency plot reveals a clear Zipf-like power-law distribution across all K values. (Top-right & Bottom-right) The heatmap and boxplots illustrate the distribution’s long-tail structure and its dynamic adaptation to varying K. (Middle) Quantitative analysis confirms that key metrics, such as the Zipf exponent $\alpha$ and the Gini coefficient, remain remarkably stable. Collectively, these results provide strong empirical evidence that time series data possesses an intrinsic, robust, language-like statistical structure.
Refer to caption: x5.png

#### 2.2.1 The Discovery of a Zipfian Distribution in the Temporal Lexicon

The experimental results offer decisive support for our theory. We discovered that the frequency distribution of these ”Temporal Words” consistently and robustly adheres to a Zipf-like law. As illustrated in the ‘Time Series Vocabulary Zipf’s Law Analysis’ plot, when the frequency of each token is plotted against its rank on a log-log scale, a distinct linear relationship emerges. This signature of a power-law distribution holds true across all tested vocabulary sizes, with K ranging from 16 to 256, demonstrating the universality of this phenomenon.

This finding is far more than a simple statistical observation; it provides a new and profound lens through which to understand time series data. The prevalence of Zipf’s law in a wide array of complex systems, from natural language to city populations, is widely considered a hallmark of systems built on principles of compositionality and evolution. In our context, its emergence strongly suggests that complex temporal dynamics are not merely sequences of independent numerical values. Instead, they behave like macroscopic phenomena generated from a finite, reusable vocabulary of underlying dynamic ”motifs,” which are combined and composed according to a form of ”grammar.”

This discovery yields two foundational insights. First, it provides a solid empirical basis for shifting the paradigm of time series analysis from numerical regression towards language modeling. The success of models like the Transformer in the time series domain has often been attributed solely to their powerful sequence processing capabilities. Our findings provide a more fundamental explanation rooted in the data’s intrinsic nature: these models are effective because, once time series are properly ”tokenized” (via patching and quantization), their statistical structure becomes isomorphic to that of natural language, the native domain for which these models were designed.

Second, it allows us to understand the information within time series in a more structured manner. The ”head” of the Zipfian distribution—the few, extremely frequent tokens—can be interpreted as the universal ”basic grammar” of dynamics, such as ”stability,” ”upward trends,” or ”seasonal patterns.” Conversely, the ”long tail,” comprising a vast number of low-frequency tokens, represents the rich, domain-specific, and often critical events or complex patterns. The power of a time series foundation model, therefore, lies not just in mastering the common grammar of the ”head,” but in its ability to comprehend and generate the rare and valuable knowledge encoded in the ”tail.”

#### 2.2.2 Robustness and Dynamic Adaptability of the Vocabulary Structure

While the existence of a Zipfian law is a critical first step, a deeper question concerns the stability of this structure. Does this quasi-linguistic property collapse or fundamentally change when the granularity of the vocabulary, represented by the core parameter K, is altered? By analyzing the dynamic morphology of the frequency distribution, we sought to answer this question and reveal the structure’s intrinsic robustness.

Our analysis, visualized in the ‘Frequency Distribution Density Heatmap’ and the ‘Frequency Distribution Statistical Box Plot,’ reveals a system that is not only stable but also adapts with remarkable grace and predictability. The primary adaptation is an elegant trade-off between representational richness and data sparsity. As K increases, the average frequency of any given ”Temporal Word” decreases, a fact made visible by the downward progression of the median in the boxplots. This is the well-behaved response of a system where a constant amount of information is being partitioned into a larger number of discrete states. It demonstrates that our vocabulary construction method is scalable and its behavior is interpretable.

More profound, however, is the invariance of the distribution’s core architecture. Despite the global shift in frequencies, the fundamental imbalance—characterized by a few ”superstar” motifs and a vast ”population” of common ones—remains unchanged. This is most powerfully illustrated by the boxplots. Across all values of K, we consistently observe a significant number of high-frequency outliers. In this context, these outliers are not statistical noise to be dismissed; they are the most important signal. They represent the foundational dynamic motifs that are so prevalent and fundamental that they are inevitably identified by the clustering algorithm, regardless of how many clusters it is asked to form. Their persistence validates that our methodology captures true, stable structures inherent in the data, rather than fleeting artifacts of the clustering process.

#### 2.2.3 Quantitative Confirmation of the Structure’s Intrinsic Nature

While visual inspection provides compelling intuition, a rigorous scientific claim demands objective, quantitative validation. We provide this decisive proof by analyzing key statistical invariants of the distribution: the fitted Zipf exponent ($\alpha$) and the Gini coefficient.

The analysis, quantified in the ‘Statistical Indicators’ bar chart, confirms the structure’s profound stability. The first key invariant, the Zipf exponent $\alpha$, which dictates the rate of frequency decay, remains relatively stable, fluctuating within the range of approximately 0.42 to 0.55 across the tested K values. This signifies that the fundamental ”grammatical rule” governing the relationship between common and rare patterns is a persistent property of this ”language.”

The second key invariant, the Gini coefficient, measures the inequality of the frequency distribution. It provides complementary and equally powerful evidence. The coefficient remains stable at a high value of approximately 0.6 across all K values tested. A high Gini coefficient is a direct mathematical signature of a system rich with information and structure, distinguishing it from random noise (which would have a Gini near zero).

The joint stability of these two invariants elevates our finding from a compelling analogy to a measurable statistical law. It proves, with mathematical certainty, that the quasi-linguistic structure we have uncovered is not an artifact of a specific algorithm or parameter choice, but is a profound and intrinsic property that emerges when time series data is viewed through a symbolic lens. This provides an unshakable quantitative foundation for the ”Language of Time” hypothesis and for the development of robust, general-purpose foundation models for time series.

### 2.3 The Grammar of Time Series

Figure: Figure 6: Comprehensive Grammatical Analysis of Temporal Motif Sequences. This figure visualizes the ”grammar” of motif sequences, revealing key principles: a strong state inertia shown in the transition matrix (top-left); language-like sparsity demonstrated by exponentially decaying n-gram coverage (top-right); and high macroscopic diversity from ”chunking,” supported by the broad complexity and entropy distributions (middle row). Collectively, these analyses provide a visual fingerprint of a non-trivial, discoverable grammar underlying time series data.
Refer to caption: x6.png

Having established that time series data can be tokenized into a robust ”vocabulary” of motifs exhibiting language-like frequency distributions (Sections 2.1, 2.2), we now address a more profound question: do these motifs combine randomly, or do they follow a discernible ”grammar”? A true language is defined not just by its words, but by the rules that govern their composition. To investigate this, we conducted a comprehensive grammatical analysis on the sequence of tokenized time series. The results, summarized in Figure [6](https://arxiv.org/html/2507.00078v1#S2.F6), reveal a clear, non-trivial grammar governing the ”language of time.”

Our analysis, visualized in Figure [6](https://arxiv.org/html/2507.00078v1#S2.F6), uncovers three fundamental grammatical principles:

##### The Principle of State Inertia.

Our primary discovery, clearly visible in the State Transition Probability Matrix (top-left panel), is the overwhelming dominance of self-transitions. The bright diagonal line indicates that once a particular temporal motif is established, it has a very high probability of persisting in the subsequent time step. This principle of state inertia is further corroborated by the analysis of the most frequent 2-grams (middle-right panel), where self-loops (e.g., a motif followed by itself) are the most common pairs. This is the simplest and most powerful rule of temporal grammar: dynamics are persistent.

Analogy to Natural Language: This principle is analogous to structures in language that maintain focus. For instance, a paragraph typically revolves around a core topic, keeping its ’state’ persistent. It is also akin to using a series of adjectives to describe a single noun (e.g., ”a long, dark, quiet road”), during which the subject of the description remains constant. Just as a sentence’s subject often endures across clauses, the temporal ’subject’ (i.e., the current dynamic pattern) tends to endure.

##### A Highly Structured and Sparse Language.

While motifs tend to repeat, their transitions to different motifs are far from random. The space of ”grammatically valid” motif combinations is extremely sparse. This is best illustrated by the N-gram Coverage Analysis (top-right panel), which shows that the coverage of possible n-grams decays exponentially as N increases. While 100% of 1-grams (single motifs) are observed, the coverage drops precipitously for 2-grams and higher, indicating that only a small fraction of all possible motif sequences are ”grammatically correct” or physically plausible.

Analogy to Natural Language: This is perhaps the most direct parallel to natural language grammar. Just as English syntax dictates that ’The cat sat’ is a valid phrase while ’Sat the cat’ is not, the grammar of time permits only a highly structured subset of all possible motif combinations. This is also akin to linguistic collocations, where we conventionally say ’heavy rain’ instead of ’large rain’. This sparsity provides strong proof for the existence of powerful, underlying compositional rules.

##### Macroscopic Diversity from Microscopic ”Chunking”.

At first glance, the dominance of self-loops might suggest that sequences are simple and monotonous. However, the Sequence Complexity and Entropy distributions (middle row) reveal a more nuanced reality: the sequences exhibit high macroscopic diversity, with broad distributions centered around non-trivial values. This apparent paradox is explained by a ”chunking” mechanism, where complex sequences are constructed by composing persistent chunks of motifs. A typical sequence is not uniformly simple or complex, but rather a concatenation of internally-stable segments, which generates high overall diversity and uniqueness. The Sequence Features Relationship plot (bottom-left) further reinforces this by showing a rich and varied interplay between the number of unique motifs used in a sequence and its resulting complexity and entropy.

Analogy to Natural Language: This ”chunking” mechanism perfectly mirrors the hierarchical structure of natural language. A complex sentence is not a random string of words but a structured composition of well-defined phrases and clauses (e.g., a noun phrase followed by a verb phrase). Similarly, a complex time series appears to be a composition of persistent ’motif phrases,’ concatenated to form a longer, meaningful ’temporal sentence.’ This process allows for immense expressive diversity while still adhering to a simpler set of local rules.

In summary, the collective evidence presented in Figure [6](https://arxiv.org/html/2507.00078v1#S2.F6) demonstrates that the “language of time” possesses not only a well-defined vocabulary but also a non-trivial, discoverable grammar. Our analysis further confirms that this “language of time” exhibits many syntactic patterns that closely mirror those seen in natural-language models. At the same time, because a time-series foundation model is an independent system, it also contains domain-specific syntactic constructs that do not map directly onto linguistic syntax; these unique rules deserve deeper investigation. This structure is precisely what allows foundation models to move beyond simple pattern matching and learn the underlying generative rules of temporal data, enabling effective forecasting and representation learning.

## 3 Theoretical Foundation

Our empirical findings suggest that time series, when viewed through the lens of patching and quantization, exhibit remarkable language-like statistical properties. To move beyond analogy and establish a rigorous basis for these observations, we now develop a hierarchical theoretical framework. This framework aims to answer three fundamental questions:
(1) Is it mathematically sound to represent continuous patches with a discrete vocabulary?
(2) Does this representation empower the model to learn and generalize effectively?
(3) Why is this patch-based representation inherently advantageous?

### 3.1 Feasibility and Structure of the Temporal Vocabulary

##### Fidelity of Representation.

First, we must establish that discretizing continuous, high-dimensional patches into a finite set of tokens is mathematically sound. The following theorem, based on covering number theory, guarantees that such a representation can be arbitrarily faithful.

###### Theorem 3.1 ( $\varepsilon$ - Covering Guarantees Bounded Information Loss) .

Let $0<\varepsilon<2\sqrt{P}$, where $P$ is the patch dimension. There exists a codebook $\mathcal{C}$ with a finite size $K$ such that for any patch vector $h$, its quantized representation $Q_{\mathcal{C}}(h)$ satisfies $d(h,Q_{\mathcal{C}}(h))\leq\varepsilon$.

Interpretation: This result confirms that we can construct a finite vocabulary that represents any continuous patch with a quantization error no larger than a predefined $\varepsilon$. This provides the theoretical cornerstone for tokenization, ensuring the process is fundamentally reliable. A detailed proof is provided in Appendix [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem1).

###### Theorem 3.1 ( ε 𝜀 \varepsilon italic_ε - Covering Guarantees Bounded Information Loss) .

##### Statistical Structure of the Vocabulary.

Having established that a vocabulary can be faithfully constructed, we now provide a theoretical explanation for the Zipf-like distribution observed in our empirical results (Section [2.2](https://arxiv.org/html/2507.00078v1#S2.SS2)). We model the generation of tokens using a Griffiths-Engen-McCloskey (GEM) distribution, a standard process for generating power-law phenomena.

###### Theorem 3.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .

Assume the probability distribution of tokens follows a two-parameter GEM distribution. The expected value of its ranked empirical frequency $f_{n}(r)$ (the frequency of the r-th most common token) satisfies a power-law relationship: $\mathbb{E}[f_{n}(r)]\asymp r^{-\beta}$.

Interpretation: This theorem demonstrates that if the underlying ”choice” of temporal motifs follows a plausible generative process, the resulting token frequencies will naturally exhibit the Zipf-like signature of natural language. This connects our empirical discovery to established statistical theory, solidifying the ”language of time” hypothesis. The full proof can be found in Appendix [A.2](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem2).

###### Theorem 3.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .

### 3.2 Representational Power and Generalization Guarantees

##### Expressiveness of Patch Representations.

A critical concern is whether patching might limit the model’s expressive power compared to processing raw data points. The following result shows that the opposite is true: patch-based representations can only enhance expressiveness.

###### Theorem 3.3 (Capacity Measure Monotonicity) .

The hypothesis space of a patch-based model, $\mathcal{H}_{\text{patch}}$, contains the hypothesis space of an equivalent pointwise model, $\mathcal{H}_{\text{point}}$ (i.e., $\mathcal{H}_{\text{point}}\subseteq\mathcal{H}_{\text{patch}}$). Consequently, any standard measure of model capacity (e.g., VC Dimension, Rademacher Complexity) for the patch-based model is greater than or equal to that of the pointwise model.

Interpretation: Patching does not constrain what a model can learn; it creates a richer representation space. This ensures that the performance gains from patching are not at the cost of reduced expressiveness. The proof is detailed in Appendix [A.2.1](https://arxiv.org/html/2507.00078v1#A1.SS2.SSS1).

###### Theorem 3.3 (Capacity Measure Monotonicity) .

##### Generalization on Dependent Data.

Time series data violates the standard i.i.d. assumption of learning theory, posing a challenge for guaranteeing generalization. We prove that our tokenization process preserves the underlying dependency structure, allowing us to establish a valid generalization bound.

###### Lemma 3.1 ( $\beta$ -Mixing Preservation) .

If the original time series $\{X_{t}\}$ is $\beta$-mixing (a common measure of temporal dependency), then the resulting token sequence $\{T_{m}\}$ is also $\beta$-mixing.

This preservation of dependency structure allows us to apply generalization bounds for non-i.i.d. sequences.

###### Theorem 3.4 (Dependence Generalisation Bound) .

For a learning algorithm with uniform stability $\varepsilon_{\text{stab}}$ trained on a $\beta$-mixing token sequence of length $n$, the generalization error is bounded with high probability: $G_{n}(A)\leq 2\varepsilon_{\text{stab}}+O(\frac{1}{\sqrt{n}})$.

Interpretation: Together, these results provide a crucial theoretical guarantee. They show that even though time series data is complex and dependent, the process of tokenization is ”safe” and does not break the mathematical assumptions needed to prove that the model can generalize from the training set to unseen data. See Appendix [A.4](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem4) for detailed proofs.

###### Lemma 3.1 ( β 𝛽 \beta italic_β -Mixing Preservation) .

###### Theorem 3.4 (Dependence Generalisation Bound) .

### 3.3 The Information-Theoretic Advantage of Patching

Finally, we address why patching is not just a valid representation, but an advantageous one. Using the Information Bottleneck principle, we show that patching acts as an effective denoising mechanism.

###### Theorem 3.5 (Patch Representation as an Effective Information Bottleneck) .

Patching and quantization transform the input $X$ into a compressed representation $Z_{\text{patch}}$. This process acts as an information bottleneck that preferentially discards noise (reducing the compression cost $I(X;Z_{\text{patch}})$) while preserving task-relevant information (maintaining the predictive power $I(Y;Z_{\text{patch}})$).

Interpretation: This theorem provides the fundamental justification for the robustness of patch-based models. Patching is not merely a segmentation technique; it is an intelligent form of information compression. By averaging out local variations and focusing on prototypical shapes, it naturally filters out high-frequency, task-irrelevant noise, leading to a cleaner signal for the downstream model and explaining the success of cross-domain transfer. A formal treatment is available in Appendix [A.4](https://arxiv.org/html/2507.00078v1#A1.SS4).

###### Theorem 3.5 (Patch Representation as an Effective Information Bottleneck) .

## 4 Conclusion

This paper resolves the paradox of why time series foundation models transfer so well across different domains. We propose that these models function like large language models, learning a universal language of ”temporal motifs” by representing time series patches as ”distributional tokens.” We provide strong empirical and theoretical evidence for this ”language of time” hypothesis. Empirically, we demonstrate for the first time that time series patches adhere to Zipf’s Law, a statistical signature of language, and uncover their compositional grammar. Theoretically, we build a complete analytical framework to validate the model’s representation, information compression, and generalization capabilities. Our work provides the first rigorous explanation for the success of time series foundation models and paves the way for building safer and more powerful temporal models.

## References

- Ansari et al. (2024)
Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al.
Chronos: Learning the language of time series.
*arXiv preprint arXiv:2403.07815*, 2024.
- Barbosa et al. (2018)
Hugo Barbosa, Marc Barthelemy, Gourab Ghoshal, Charlotte R James, Maxime Lenormand, Thomas Louail, Ronaldo Menezes, José J Ramasco, Filippo Simini, and Marcello Tomasini.
Human mobility: Models and applications.
*Physics Reports*, 734:1–74, 2018.
- Das et al. (2024a)
Abhimanyu Das, Matthew Faw, Rajat Sen, and Yichen Zhou.
In-context fine-tuning for time-series foundation models.
*arXiv preprint arXiv:2410.24087*, 2024a.
- Das et al. (2024b)
Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou.
A decoder-only foundation model for time-series forecasting.
In *Forty-first International Conference on Machine Learning*, 2024b.
- Ding et al. (2025)
Xueying Ding, Aakriti Mittal, and Achintya Gopal.
Delphyne: A pre-trained model for general and financial time series.
*arXiv preprint arXiv:2506.06288*, 2025.
- Fu et al. (2024)
Yingchun Fu, Zhe Zhu, Liangyun Liu, Wenfeng Zhan, Tao He, Huanfeng Shen, Jun Zhao, Yongxue Liu, Hongsheng Zhang, Zihan Liu, et al.
Remote sensing time series analysis: A review of data and applications.
*Journal of Remote Sensing*, 4:0285, 2024.
- Goswami et al. (2024)
Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski.
Moment: A family of open time-series foundation models.
*arXiv preprint arXiv:2402.03885*, 2024.
- He et al. (2023)
Huan He, Owen Queen, Teddy Koker, Consuelo Cuevas, Theodoros Tsiligkaridis, and Marinka Zitnik.
Domain adaptation for time series under feature and label shifts.
In *International conference on machine learning*, pp.  12746–12774. PMLR, 2023.
- Lonardi & Patel (2002)
JLEKS Lonardi and Pranav Patel.
Finding motifs in time series.
In *Proc. of the 2nd workshop on temporal data mining*, pp.  53–68, 2002.
- Montgomery et al. (2015)
Douglas C Montgomery, Cheryl L Jennings, and Murat Kulahci.
*Introduction to time series analysis and forecasting*.
John Wiley & Sons, 2015.
- Mueen (2014)
Abdullah Mueen.
Time series motif discovery: dimensions and applications.
*Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 4(2):152–159, 2014.
- Xie et al. (2023)
Yi Xie, Yun Xiong, Jiawei Zhang, Chao Chen, Yao Zhang, Jie Zhao, Yizhu Jiao, Jinjing Zhao, and Yangyong Zhu.
Temporal super-resolution traffic flow forecasting via continuous-time network dynamics.
*Knowledge and Information Systems*, 65(11):4687–4712, 2023.
- Ye et al. (2024)
Jiexia Ye, Weiqi Zhang, Ke Yi, Yongzi Yu, Ziyue Li, Jia Li, and Fugee Tsung.
A survey of time series foundation models: Generalizing time series representation with large language model.
*arXiv preprint arXiv:2405.02358*, 2024.
- (14)
Zhenwei Zhang, Jiawen Zhang, Shun Zheng, Yuantao Gu, and Jiang Bian.
Does cross-domain pre-training truly help time-series foundation models?
In *ICLR 2025 Workshop on Foundation Models in the Wild*.
- Zhao et al. (2024)
Shiji Zhao, Shao-Yuan Li, and Sheng-Jun Huang.
Nanoadapt: mitigating negative transfer in test time adaptation with extremely small batch sizes.
In *Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence*, pp.  5572–5580, 2024.
- Zheng & Huang (2020)
Jianhu Zheng and Mingfang Huang.
Traffic flow forecast through time series analysis based on deep learning.
*Ieee Access*, 8:82562–82570, 2020.

## Appendix A Theoretical Analysis

**Table 1: Summary of the Theoretical Chain Supporting Patch Quantization**
| Objective | Key Result | Core Conclusion | Interpretation |
| --- | --- | --- | --- |
| Finite dictionary approximation | Thm. [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem1)<br>($\varepsilon$ - Covering) | A finite codebook exists, guaranteeing the quantization error for any patch is strictly bounded by $\leq\!\varepsilon$. | Dense enough tokens faithfully represent continuous patches. |
| Zipf frequency property | Thm. [A.2](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem2)<br>(Zipf-like distribution) | Assuming a GEM process for token generation, the resulting rank–frequency relationship follows a power-law ($f(r)\!\propto\!r^{-\beta}$). | Patch “language” has a natural-language-style long tail. |
| Capacity non-decreasing | Lem. [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmlemma1)<br>(Subclass Relation)<br>Thm. [A.3](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem3)<br>(Capacity Monotonicity) | 1. The pointwise hypothesis space is a subset of the patch-based one ($\mathcal{H}_{\text{point}}\!\subseteq\!\mathcal{H}_{\text{patch}}$).<br>2. This implies the capacity (VC dim./Rademacher complexity) of the patch representation is never lower. | Patch representations cannot restrict expressiveness—only enlarge it. |
| Optimal ERM risk non-increasing | Cor. [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmcorollary1)<br>(Optimal-ERM Risk Non-Increase)<br>(A direct result of Lem. [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmlemma1)) | Because $\mathcal{H}_{\text{point}}\!\subseteq\!\mathcal{H}_{\text{patch}}$, the minimum achievable training error (ERM) in the patch space is no greater than in the pointwise space. | The best training loss with patches is no worse than pointwise. |
| Dependence & generalisation | Lem. [A.2](https://arxiv.org/html/2507.00078v1#A1.Thmlemma2)<br>($\beta$ - Mixing Preservation)<br>Thm. [A.4](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem4)<br>(Dependence Gen. Bound) | 1. Tokenization preserves the exponential $\beta$-mixing property of the original sequence.<br>2. This allows deriving stability-based generalisation bounds similar to the IID case. | Tokenisation keeps dependence assumptions, so generalisation theory still works. |
| Information-bottleneck advantage | Thm. [A.5](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem5)<br>(Information Bottleneck)<br>Thm. [A.6](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem6)<br>($\epsilon$ - MI Preservation) | 1. Patching acts as a denoising bottleneck by compressing the input.<br>2. The loss of task-relevant mutual information from quantization is controllably small, bounded by $O(\varepsilon)$. | Patches act as a denoising bottleneck—simpler inputs, task info retained. |

### A.1 ϵ italic-ϵ \epsilon italic_ϵ - Statistical Sufficiency and Zipf-like Long-Tail Frequency of Patch Tokens

#### A.1.1 ε 𝜀 \varepsilon italic_ε - Statistical Sufficiency

###### Theorem A.1 ( $\varepsilon$ - Covering Guarantees Bounded Information Loss) .

Let $0<\varepsilon<2\sqrt{P}$, where $P$ denotes the dimension of tokens. There exists a prototype set (or codebook) $\mathcal{C}\subset\mathcal{H}^{P}$ with a size $K$ bounded by

$$ $K\leq\left(1+\frac{2\sqrt{P}}{\varepsilon}\right)^{P-1},$ (1) $$

such that for any patch vector $h\in\mathcal{H}^{P}$, its quantized representation $Q_{\mathcal{C}}(h)=\arg\min_{c\in\mathcal{C}}d(h,c)$ satisfies

$$ $\forall h\in\mathcal{H}^{P},\quad d\big{(}h,Q_{\mathcal{C}}(h)\big{)}\leq\varepsilon.$ (2) $$

Therefore, the discrete token $T=Q_{\mathcal{C}}(H)$ represents $H$ with a guaranteed maximum error of $\varepsilon$. We can consider $T$ an $\varepsilon$-sufficient representation in the sense that the information loss, as measured by the metric $d$, is bounded.

###### Proof.

Embedding into the Unit Sphere.
We assume the patch space $\mathcal{H}^{P}$ is constituted by vectors $h\in\mathbb{R}^{P}$ that have been centered (i.e., $\sum_{i}h_{i}=0$) and normalized such that their Euclidean norm is constant, $\|h\|_{2}=\sqrt{P}$. The centering constraint ensures that $\mathcal{H}^{P}$ lies within a $(P-1)$-dimensional linear subspace. The normalization constraint means that all patch vectors lie on the surface of a hypersphere of radius $\sqrt{P}$ in that subspace, which is topologically equivalent to $S^{P-2}$.

To apply standard covering number results, we map this space to the unit sphere $S^{P-2}$ via the scaling transformation $h\mapsto h^{\prime}=h/\sqrt{P}$. This transformation is bi-Lipschitz. Specifically, for any two points $h_{1},h_{2}\in\mathcal{H}^{P}$, the distance in the new space is $d(h^{\prime}_{1},h^{\prime}_{2})=d(h_{1},h_{2})/\sqrt{P}$. To guarantee a distance of at most $\varepsilon$ in the original space, we require a covering of precision $\varepsilon^{\prime}=\varepsilon/\sqrt{P}$ in the unit sphere space.

Covering Number of the Sphere.
A well-known result from geometric functional analysis states that the size of a minimal $\varepsilon^{\prime}$-net for the $d$-dimensional unit sphere, $N(\varepsilon^{\prime},S^{d-1})$, is bounded by:

$$ $N(\varepsilon^{\prime},S^{d-1})\leq\left(1+\frac{2}{\varepsilon^{\prime}} \right)^{d}.$ (3) $$

In our case, the effective dimension is $d=P-1$. Substituting $d=P-1$ and the required precision $\varepsilon^{\prime}=\varepsilon/\sqrt{P}$, we obtain the bound on our codebook size $K$:

$$ $K=N(\varepsilon/\sqrt{P},S^{P-2})\leq\left(1+\frac{2}{\varepsilon/\sqrt{P}} \right)^{P-1}=\left(1+\frac{2\sqrt{P}}{\varepsilon}\right)^{P-1}.$ (4) $$

This bound guarantees the existence of such a codebook $\mathcal{C}$. This completes the proof. Q.E.D.
∎

###### Remark A.1 (On the Practical Construction of the Codebook) .

The proof above guarantees the existence of a suitable codebook. In practice, it can be constructed through various means. Geometrically, one could form a lattice in the $(P-1)$-dimensional subspace and project the nodes onto the sphere. Algorithmically, methods like k-means++ or Lloyd-Max, when run on a representative dataset of patches, can produce a codebook $\mathcal{C}$ that empirically satisfies the $d(h,Q_{\mathcal{C}}(h))\leq\varepsilon$ condition for all data points.

Discussion and Corollaries

The theoretical results yield three key implications:

Finite Dictionary Size.
The covering number provides an upper bound on the required dictionary size. For a small $\varepsilon$, the bound has the asymptotic behavior:

$$ $K(\varepsilon)=\mathcal{O}\left(\left(\frac{\sqrt{P}}{\varepsilon}\right)^{P-1 }\right)=\mathcal{O}(\varepsilon^{-(P-1)}).$ (5) $$

This shows that the number of required prototypes grows polynomially as a function of $1/\varepsilon$, with the degree of the polynomial determined by the intrinsic dimension of the data, $P-1$.

Bounded Quantization Error.
The lemma guarantees that for any vector $h$, the quantization error is bounded: $d(h,Q_{\mathcal{C}}(h))\leq\varepsilon$. This directly implies that the expected mean squared distortion $D=\mathbb{E}[d(h,Q_{\mathcal{C}}(h))^{2}]$ is also strictly bounded:

$$ $D\leq\varepsilon^{2}.$ (6) $$

This provides a direct and robust link between the covering precision $\varepsilon$ and the expected quantization error.

Bounded Information Loss in Downstream Tasks.
The lemma guarantees that quantizing a patch vector $H$ into a discrete token $T$ introduces an error that is strictly bounded by $\varepsilon$. Consequently, any downstream model that uses $T$ instead of $H$ operates with a precisely controlled level of input perturbation. For models or tasks that are robust to small input variations, this ensures that the tokenized representation $T$ preserves sufficient information to act as a reliable and efficient proxy for the original continuous data.

Intuitively, Lemma [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem1) shows that a limited, discrete, and controllable prototype set can approximate arbitrary real patches in time series.

###### Theorem A.1 ( ε 𝜀 \varepsilon italic_ε - Covering Guarantees Bounded Information Loss) .

###### Proof.

###### Remark A.1 (On the Practical Construction of the Codebook) .

#### A.1.2 Zipf-like Long-Tail Frequency

###### Theorem A.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .

Assume the probability distribution of tokens $\pi$ follows a two-parameter GEM (Griffiths-Engen-McCloskey) distribution, denoted $\pi\sim\text{GEM}(d,\theta)$, with parameters satisfying $0\leq d<1$ and $\theta>-d$. For an i.i.d. sequence of tokens $T_{1},T_{2},\dots\sim\pi$, the expected value of its ranked empirical frequency $f_{n}(r)$ (i.e., the frequency of the r-th most common token) satisfies a power-law relationship:

$$ $\mathbb{E}[f_{n}(r)]\asymp r^{-\beta},$ $$

where the power-law exponent $\beta$ depends on the parameter $d$:

- •
For $0<d<1$, the exponent is $\beta=1/d$.
- •
For $d=0$ (the Dirichlet Process case), the exponent is $\beta=2$.

Note: The symbol $\asymp$ denotes asymptotic equivalence, i.e., $\lim_{r\to\infty}\frac{\mathbb{E}[f_{n}(r)]}{r^{-\beta}}=C$ for some positive constant $C$.

###### Proof.

We establish this result through the connection between the GEM distribution and the Pitman-Yor process, which provides a framework for analyzing the ranked probabilities.

Connection to Pitman-Yor and Poisson-Dirichlet Distributions:
The probability distribution $\pi=(\pi_{1},\pi_{2},\dots)$ generated by the $\text{GEM}(d,\theta)$ process is equivalent to the distribution of weights in a Pitman-Yor process, denoted $\text{PY}(d,\theta)$. The set of ranked probabilities $\{\pi_{(1)},\pi_{(2)},\dots\}$ of this process follows the two-parameter Poisson-Dirichlet distribution, $\text{PD}(d,\theta)$. The asymptotic behavior of these ranked probabilities is well-studied.

Asymptotic Analysis for $d>0$:
The cornerstone result for the Pitman-Yor process, established in the work of Pitman and Yor, shows that for $0<d<1$, the expected ranked probabilities follow a power law. For large $r$, this is given by:

$$ $\mathbb{E}[\pi_{(r)}]\asymp r^{-1/d}.$ (7) $$

Asymptotic Analysis for $d=0$ (Dirichlet Process):
When $d=0$, the process degenerates to the Dirichlet Process, and the ranked probabilities follow the $\text{PD}(0,\theta)$ distribution. In this case, the asymptotic behavior changes. The expected ranked probabilities exhibit a different power-law decay, given by:

$$ $\mathbb{E}[\pi_{(r)}]\asymp r^{-2}.$ (8) $$

This result stems from the analysis of Ewens’s sampling formula, which describes the partition structure of the Dirichlet Process.

From Theoretical Probabilities to Empirical Frequencies:
For a sequence of $n$ i.i.d. samples from the distribution $\pi$, the sequence is exchangeable. By the strong law of large numbers for exchangeable sequences, the empirical frequency of the $r$-th most frequent token, $f_{n}(r)$, converges to the true ranked probability $\pi_{(r)}$ almost surely as $n\to\infty$.

$$ $\lim_{n\to\infty}f_{n}(r)=\pi_{(r)},$ (9) $$

almost surely holds.

Therefore, for large $n$, the expectation of the empirical frequency is well-approximated by the expectation of the true probability, $\mathbb{E}[f_{n}(r)]\approx\mathbb{E}[\pi_{(r)}]$. This allows us to apply the asymptotic results for $\mathbb{E}[\pi_{(r)}]$ directly to $\mathbb{E}[f_{n}(r)]$, establishing the power-law relationships as stated in the lemma. This completes the proof. Q.E.D.
∎

###### Remark A.2 (Connection to Zipf’s Law in Linguistics) .

Zipf’s law was first discovered in linguistics, where the power-law exponent $\beta$ is approximately 1. In our model, as the discount parameter $d\to 1^{-}$, we get $\beta=1/d\to 1$, which corresponds perfectly to the classic law. Empirical studies have shown that for real-world languages, the value of $d$ is typically between $0.7\text{--}0.8$, which leads to $\beta\approx 1.25\text{--}1.4$, in high agreement with linguistic observations.

This section’s theoretical analysis provides a rigorous foundation for tokenizing continuous patch data. In short, the two lemmas establish a complete theoretical chain.

First, Lemma [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem1) proves that any continuous patch can be represented by a token from a finite codebook with a guaranteed, bounded error ($\epsilon$-sufficiency). This confirms the feasibility and fidelity of the tokenization process. Second, Lemma [A.2](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem2) demonstrates that if the token generation process follows a GEM distribution, the resulting token frequencies will exhibit a Zipf-like power-law distribution, a key statistical signature of natural language.

Collectively, these results provide a solid theoretical basis for treating continuous signals as a ”language,” thereby validating the application of powerful sequence models like the Transformer.

###### Theorem A.2 (Zipf-like Long-Tail Distribution for Patch Tokens) .

###### Proof.

###### Remark A.2 (Connection to Zipf’s Law in Linguistics) .

### A.2 Non-decreased Representational Capacity and Non-increased Optimal-ERM Risk

#### A.2.1 Monotone Representational Capacity

###### Definition A.1 (Pointwise Hypothesis Space) .

Let $X\subseteq\mathbb{R}^{d}$ be the input space and $Y$ be the output space. The pointwise hypothesis space is defined as:

$$ $\mathcal{H}_{\text{point}}=\{h_{\theta}:X\to Y\mid h_{\theta}(x)=g_{\theta}(x) ,\theta\in\Theta\}$ (10) $$

where $g_{\theta}:\mathbb{R}^{d}\to Y$ is a parameterized function family.

###### Definition A.2 (Patch Hypothesis Space) .

Given a dictionary $\mathcal{C}=\{c_{1},c_{2},\ldots,c_{K}\}$ where each $c_{i}$ is a patch of length $P$, and a sliding window stride $S$, we define:

- •
Quantization function: $Q_{\mathcal{C}}:\mathbb{R}^{d}\to\mathcal{C}^{*}$, which segments the input sequence and maps it to the nearest patches in the dictionary
- •
Embedding function: $e:\mathcal{C}\to\mathbb{R}^{d^{\prime}}$, which maps patch tokens to the embedding space
- •
Reconstruction function: $\text{Reconstruct}:(\mathbb{R}^{d^{\prime}})^{*}\to\mathbb{R}^{d}$, which reconstructs patch sequences to the original dimension

The patch hypothesis space is defined as:

$$ $\begin{split}\mathcal{H}_{\text{patch}}=\Big{\{}&h_{\theta,\mathcal{C}}^{\text {patch}}:X\to Y\,\Big{|}\,h_{\theta,\mathcal{C}}^{\text{patch}}(x)\\ &=g_{\theta}\big{(}\text{Reconstruct}(\text{Embed}(Q_{\mathcal{C}}(x)))\big{)} ,\quad\theta\in\Theta\Big{\}}.\end{split}$ (11) $$

###### Assumption A.1 .

We assume the following conditions hold:

- 1.
$g_{\theta}$ is a continuous function for all $\theta\in\Theta$
- 2.
There exists an inverse reconstruction function such that under specific conditions, $\text{Reconstruct}(\text{Embed}(\cdot))$ can be the identity mapping
- 3.
The parameter space $\Theta$ remains consistent across both hypothesis spaces

###### Lemma A.1 (Idealized Subclass Relation) .

Let the dictionary $\mathcal{C}$ contain all length-1 patches, i.e., $\mathcal{C}\supseteq\{(x_{i})\mid x_{i}\in\mathbb{R},i=1,\ldots,d\}$, and set the sliding window stride $S=1$. Under appropriate choices of embedding and reconstruction functions, there exists:

$$ $\mathcal{H}_{\text{point}}\subseteq\mathcal{H}_{\text{patch}}$ (12) $$

###### Proof.

Construct the special embedding function.
For any length-1 patch $c=(a)\in\mathbb{R}$, define the embedding function as:

$$ $e(c)=a\in\mathbb{R}$ (13) $$

i.e., the identity embedding.

Verify the identity property of reconstruction.
When $S=1$ and all length-1 patches are in the dictionary, for any $x=(x_{1},x_{2},\ldots,x_{d})\in\mathbb{R}^{d}$:

- •
Quantization process: $Q_{\mathcal{C}}(x)=((x_{1}),(x_{2}),\ldots,(x_{d}))$
- •
Embedding process: $\text{Embed}(Q_{\mathcal{C}}(x))=(e((x_{1})),e((x_{2})),\ldots,e((x_{d})))=(x_
{1},x_{2},\ldots,x_{d})$
- •
Reconstruction process: $\text{Reconstruct}(\text{Embed}(Q_{\mathcal{C}}(x)))=x$

Establish functional equivalence.
For any $h_{\theta}^{\text{point}}\in\mathcal{H}_{\text{point}}$, there exists a corresponding $h_{\theta,\mathcal{C}}^{\text{patch}}\in\mathcal{H}_{\text{patch}}$ such that:

$$ $h_{\theta,\mathcal{C}}^{\text{patch}}(x)=g_{\theta}(\text{Reconstruct}(\text{ Embed}(Q_{\mathcal{C}}(x))))=g_{\theta}(x)=h_{\theta}^{\text{point}}(x)$ (14) $$

Therefore, $\mathcal{H}_{\text{point}}\subseteq\mathcal{H}_{\text{patch}}$. ∎∎

###### Theorem A.3 (Capacity Measure Monotonicity) .

Let $\mathcal{H}_{1}\subseteq\mathcal{H}_{2}$ be two hypothesis spaces. Then:

- 1.
VC Dimension Monotonicity: $\mathrm{VC}(\mathcal{H}_{1})\leq\mathrm{VC}(\mathcal{H}_{2})$
- 2.
Empirical Rademacher Complexity Monotonicity: $\widehat{\mathfrak{R}}_{n}(\mathcal{H}_{1})\leq\widehat{\mathfrak{R}}_{n}(
\mathcal{H}_{2})$

###### Proof.

VC Dimension:
Let $\mathcal{S}=\{x_{1},\ldots,x_{m}\}$ be any set shattered by $\mathcal{H}_{1}$. That is, there exist functions $h_{1},\ldots,h_{2^{m}}\in\mathcal{H}_{1}$ such that for each $A\subseteq\{1,\ldots,m\}$, there exists $h_{A}\in\mathcal{H}_{1}$ satisfying:

$$ $h_{A}(x_{i})=\begin{cases}1&\text{if }i\in A\\ 0&\text{if }i\notin A\end{cases}$ (15) $$

Since $\mathcal{H}_{1}\subseteq\mathcal{H}_{2}$, all these functions also belong to $\mathcal{H}_{2}$, hence $\mathcal{S}$ is also shattered by $\mathcal{H}_{2}$.
Therefore, $\mathrm{VC}(\mathcal{H}_{1})\leq\mathrm{VC}(\mathcal{H}_{2})$.

Rademacher Complexity Part:

$$ $\begin{split}\widehat{\mathfrak{R}}_{n}(\mathcal{H}_{1})&=\mathbb{E}_{\sigma} \left[\sup_{h\in\mathcal{H}_{1}}\frac{1}{n}\sum_{i=1}^{n}\sigma_{i}h(x_{i}) \right]\\ &\leq\mathbb{E}_{\sigma}\left[\sup_{h\in\mathcal{H}_{2}}\frac{1}{n}\sum_{i=1}^ {n}\sigma_{i}h(x_{i})\right]\\ &=\widehat{\mathfrak{R}}_{n}(\mathcal{H}_{2})\end{split}$ (16) $$

where the inequality follows from $\mathcal{H}_{1}\subseteq\mathcal{H}_{2}$, making the supremum over $\mathcal{H}_{2}$ at least as large as that over $\mathcal{H}_{1}$. ∎∎

###### Corollary A.1 (Capacity Monotonicity for Patch Methods) .

Combining Lemma [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmlemma1) and Theorem [A.3](https://arxiv.org/html/2507.00078v1#A1.Thmtheorem3), under the stated conditions:

- 1.
$\mathrm{VC}(\mathcal{H}_{\text{patch}})\geq\mathrm{VC}(\mathcal{H}_{\text{
point}})$
- 2.
$\widehat{\mathfrak{R}}_{n}(\mathcal{H}_{\text{patch}})\geq\widehat{\mathfrak{R
}}_{n}(\mathcal{H}_{\text{point}})$

###### Remark A.3 (Computational Complexity) .

Including all length-1 patches implies a dictionary size of $|\mathcal{C}|\geq|\text{range}(X)|$, which is infeasible for continuous input spaces. Practical applications require:

- •
Quantization strategies to limit dictionary size
- •
Approximation methods to preserve theoretical properties

###### Remark A.4 (Generalization Bounds) .

While patch methods possess higher representational capacity, this may lead to:

- •
Larger generalization error upper bounds
- •
Requirements for more training data to achieve comparable generalization performance

###### Remark A.5 (Practical Trade-offs) .

The theoretical capacity advantage must be balanced against:

- •
Computational efficiency
- •
Memory requirements
- •
Optimization difficulty

###### Proposition A.1 (Extension to Other Measures) .

The monotonicity results above extend to:

- •
Pseudo-dimension: For real-valued function classes
- •
Gaussian complexity: Using Gaussian random variables instead of Rademacher variables
- •
Local Rademacher complexity: Defined over subsets of function classes

The proof methodology follows similarly, based on the monotonicity of set inclusion relations.

###### Remark A.6 (Connection to PatchTST) .

The ”P=1” ablation study in PatchTST corresponds exactly to the setup described in Lemma [A.1](https://arxiv.org/html/2507.00078v1#A1.Thmlemma1), where the original sequence is treated as ”minimal patches.” This validates the practical relevance of our theoretical framework.

###### Definition A.1 (Pointwise Hypothesis Space) .

###### Definition A.2 (Patch Hypothesis Space) .

###### Assumption A.1 .

###### Lemma A.1 (Idealized Subclass Relation) .

###### Proof.

###### Theorem A.3 (Capacity Measure Monotonicity) .

###### Proof.

###### Corollary A.1 (Capacity Monotonicity for Patch Methods) .

###### Remark A.3 (Computational Complexity) .

###### Remark A.4 (Generalization Bounds) .

###### Remark A.5 (Practical Trade-offs) .

###### Proposition A.1 (Extension to Other Measures) .

###### Remark A.6 (Connection to PatchTST) .

#### A.2.2 Non-increased Optimal-ERM Risk

###### Corollary A.2 (Optimal-ERM Risk Non-Increase) .

For the same dataset $D$ and loss function $\ell$,

$$ $\min_{h\in\mathcal{H}_{\text{patch}}}\widehat{R}_{D}(h)\leq\min_{h\in\mathcal{ H}_{\text{point}}}\widehat{R}_{D}(h).$ (17) $$

###### Proof.

Define the optimal pointwise hypothesis.
Let

$$ $h^{\star}_{\text{point}}=\arg\min_{h\in\mathcal{H}_{\text{point}}}\widehat{R}_ {D}(h).$ (18) $$

Even if the minimum is not attained, an approximating sequence suffices.

Lift to the patch class.
By Lemma 1 ($\mathcal{H}_{\text{point}}\subseteq\mathcal{H}_{\text{patch}}$), we have $h^{\star}_{\text{point}}\in\mathcal{H}_{\text{patch}}$.

Compare minima over classes.
The minimum over a superset satisfies:

$$ $\min_{h\in\mathcal{H}_{\text{patch}}}\widehat{R}_{D}(h)\leq\widehat{R}_{D}(h^{ \star}_{\text{point}})=\min_{h\in\mathcal{H}_{\text{point}}}\widehat{R}_{D}(h).$ (19) $$

The minimal empirical risk in the larger patch class is no greater than that in the smaller pointwise class. Q.E.D.
∎

###### Corollary A.2 (Optimal-ERM Risk Non-Increase) .

###### Proof.

### A.3 A Rigorous Bound for Token Sequence Dependence

###### Definition A.3 (Patch Construction) .

Given a time series $\{X_{t}\}$, we define the patch sequence $\{Z_{m}\}$ as:

$$ $Z_{m}=\left(X_{(m-1)S+1},X_{(m-1)S+2},\ldots,X_{(m-1)S+P}\right)$ (20) $$

where $S>0$ is the stride and $P>0$ is the patch size.

###### Definition A.4 (Quantization) .

Through a deterministic quantization function $Q:\mathbb{R}^{P}\to\mathcal{T}$, where $\mathcal{T}$ is the token space, we obtain the token sequence $\{T_{m}\}$:

$$ $T_{m}=Q(Z_{m})$ (21) $$

###### Lemma A.2 (Patch $\beta$ -Mixing Preservation) .

Let $\{X_{t}\}$ be a $\beta$-mixing sequence with coefficients satisfying $\beta_{X}(k)\leq Ce^{-\rho k}$ for some constants $C,\rho>0$. The token sequence $\{T_{m}\}$ constructed as defined above remains $\beta$-mixing. Furthermore, when the non-overlapping condition $S\geq P$ holds, its $\beta$-mixing coefficients are bounded by:

$$ $\beta_{T}(k)\leq\beta_{X}(kS-P+1)$ (22) $$

###### Proof.

The proof proceeds in four steps.

$\sigma$-Algebra Setup.
To determine the $\beta$-mixing coefficient $\beta_{T}(k)$ for the token sequence, we consider the $\sigma$-algebras representing the past and future of the sequence $\{T_{m}\}$:

$$ $\displaystyle\mathcal{F}_{m}$ $\displaystyle=\sigma(T_{1},T_{2},\ldots,T_{m})$ (23) $\displaystyle\mathcal{G}_{m+k}$ $\displaystyle=\sigma(T_{m+k},T_{m+k+1},\ldots)$ (24) $$

Since each token $T_{j}$ is a deterministic function of the patch $Z_{j}=(X_{(j-1)S+1},\ldots,X_{(j-1)S+P})$, these $\sigma$-algebras are contained within the $\sigma$-algebras of the original sequence $\{X_{t}\}$. Specifically, the last data point influencing $\mathcal{F}_{m}$ is $X_{(m-1)S+P}$, and the first data point influencing $\mathcal{G}_{m+k}$ is $X_{(m+k-1)S+1}$. This gives us the tightest possible inclusions:

$$ $\displaystyle\mathcal{F}_{m}$ $\displaystyle\subseteq\sigma(X_{-\infty},\ldots,X_{(m-1)S+P})$ (25) $\displaystyle\mathcal{G}_{m+k}$ $\displaystyle\subseteq\sigma(X_{(m+k-1)S+1},\ldots,X_{\infty})$ (26) $$

Temporal Gap Analysis.
The temporal gap between the two $\sigma$-algebras of the underlying process in equation [25](https://arxiv.org/html/2507.00078v1#A1.E25) and equation [26](https://arxiv.org/html/2507.00078v1#A1.E26) is the difference between the first index of the future and the last index of the past:

$$ $\text{Gap}=\left((m+k-1)S+1\right)-\left((m-1)S+P\right)=kS-P+1$ (27) $$

Given the condition $S\geq P$ and $k\geq 1$, this gap is guaranteed to be positive, as $kS-P+1\geq S-P+1\geq 1$.

$\beta$-Mixing Inequality Derivation.
By the definition of the $\beta$-mixing coefficient for $\{T_{m}\}$, we have:

$$ $\beta_{T}(k)=\sup_{m}\sup_{\begin{subarray}{c}A\in\mathcal{F}_{m}\\ B\in\mathcal{G}_{m+k}\end{subarray}}|\mathbb{P}(A\cap B)-\mathbb{P}(A)\mathbb{ P}(B)|$ (28) $$

Since $T_{m}$ is a deterministic function of $X_{t}$, any events $A\in\mathcal{F}_{m}$ and $B\in\mathcal{G}_{m+k}$ correspond to preimage events in the appropriate $\sigma$-algebras of $\{X_{t}\}$. The dependence cannot be increased by this deterministic transformation. Therefore, the dependence between $A$ and $B$ is bounded by the dependence between their preimages, separated by the calculated gap:

$$ $\begin{split}|\mathbb{P}(A\cap B)-\mathbb{P}(A)\mathbb{P}(B)|&\leq\sup_{\begin {subarray}{c}A^{\prime}\in\sigma(X_{-\infty},\ldots,X_{(m-1)S+P})\\ B^{\prime}\in\sigma(X_{(m+k-1)S+1},\ldots)\end{subarray}}\\ &\quad|\mathbb{P}(A^{\prime}\cap B^{\prime})-\mathbb{P}(A^{\prime})\mathbb{P}( B^{\prime})|.\end{split}$ (29) $$

The right-hand side is precisely the definition of the $\beta$-mixing coefficient of the original sequence $\{X_{t}\}$ for a gap of $kS-P+1$. Thus,

$$ $\beta_{T}(k)\leq\beta_{X}(kS-P+1)$ (30) $$

Exponential Decay Preservation.
Given that $\beta_{X}(k)\leq Ce^{-\rho k}$, we can bound $\beta_{T}(k)$:

$$ $\beta_{T}(k)\leq\beta_{X}(kS-P+1)\leq Ce^{-\rho(kS-P+1)}$ (31) $$

We can rewrite this to show that $\{T_{m}\}$ also exhibits exponential decay:

$$ $Ce^{-\rho(kS-P+1)}=Ce^{-\rho(S-P+1)}e^{-\rho S(k-1)}=C^{\prime}e^{-\rho^{ \prime}(k-1)}$ (32) $$

where the new constants are $C^{\prime}=Ce^{-\rho(S-P+1)}$ and $\rho^{\prime}=\rho S$. This confirms that the exponential decay property is preserved.
∎

###### Remark A.7 (Non-overlapping Condition) .

The condition $S\geq P$ is crucial for this clean derivation. It ensures that the patches of the original time series used to generate different tokens do not overlap and, more formally, guarantees a positive temporal gap ($kS-P+1\geq 1$) for all $k\geq 1$. This simplifies the temporal gap analysis significantly. This is a common setup in applications like Vision Transformers (ViT).

###### Remark A.8 (Overlapping Case) .

When $S<P$, the patches overlap, and the analysis becomes more complex as dependencies from shared data points must be accounted for. A more refined analysis, beyond the scope of this proof, could yield a bound such as:

$$ $\beta_{T}(k)\leq\max\{P-S+1,1\}\cdot\beta_{X}(\max\{(k-1)S-P+1,1\})$ (33) $$

###### Remark A.9 (Quantization Independence) .

This result holds for any deterministic quantization function $Q$. The specific choice of tokenizer or quantization method (e.g., k-means clustering, VQ-VAE) does not affect the validity of the bound, making it broadly applicable.

###### Theorem A.4 (Dependence Generalisation Bound) .

Let an algorithm $A$ have uniform stability $\varepsilon_{\text{stab}}$. Let the data sequence $T_{1:n}=\{Z_{1},\dots,Z_{n}\}$ be drawn from a stochastic process satisfying $\beta$-mixing, with mixing coefficients that satisfy $\sum_{k\geq 1}\beta(k)=B<\infty$. Let the loss function $loss(\cdot,\cdot)$ be bounded, and let $\sigma^{2}$ be an upper bound on its variance.

Then, for any $\delta\in(0,1)$, with probability at least $1-\delta$, the following inequality holds:

$$ $G_{n}(A(T_{1:n}))\leq 2\varepsilon_{\text{stab}}+\sqrt{\frac{2\sigma^{2}(1+4B) \ln(2/\delta)}{n}}$ (34) $$

(Note: The constant $(1+4B)$ comes from tighter concentration inequalities for $\beta$-mixing sequences, such as variants of McDiarmid’s or Bernstein’s inequalities. The specific constant depends on the underlying concentration inequality being invoked.)

###### Proof.

Let $h=A(T_{1:n})$ denote the hypothesis (model) trained on the training set $T_{1:n}$. The generalization error is defined as the difference between the true risk and the empirical risk:

$$ $\begin{split}G_{n}(h)&=R(h)-R_{\text{emp}}(h)\\ &=\mathbb{E}_{Z\sim\mathcal{D}}[\mathrm{loss}(h,Z)]-\frac{1}{n}\sum_{i=1}^{n} \mathrm{loss}(h,Z_{i}).\end{split}$ (35) $$

Our goal is to provide a high-probability upper bound for $G_{n}(h)$. We decompose the error into two parts: a Bias Term and a Concentration Term.

$$ $G_{n}(h)=\underbrace{\left(R(h)-\mathbb{E}[R_{emp}(h)]\right)}_{\text{Bias Term}}+\underbrace{\left(\mathbb{E}[R_{emp}(h)]-R_{emp}(h)\right)}_{\text{ Concentration Term}}$ (36) $$

By the triangle inequality, we can bound the two terms separately:

$$ $G_{n}(h)\leq\left|R(h)-\mathbb{E}[R_{emp}(h)]\right|+\left|\mathbb{E}[R_{emp}( h)]-R_{emp}(h)\right|$ (37) $$

Bounding the Bias Term
We first bound the term $\left|R(h)-\mathbb{E}[R_{emp}(h)]\right|$. The core of this step is to leverage the uniform stability of the algorithm. Through a classic symmetrization argument, which involves introducing a ”ghost sample” drawn independently from the same distribution, it can be shown that uniform stability implies a bound on the gap between the true risk and the expected empirical risk:

$$ $\left|\mathbb{E}[R(h)]-\mathbb{E}[R_{emp}(h)]\right|\leq 2\varepsilon_{\text{ stab}}$ (38) $$

This bound is deterministic; it does not depend on a particular sample but only on the algorithm’s stability property. It quantifies the systematic bias introduced because the algorithm uses the same data for both training and evaluation.

Bounding the Concentration Term
Next, we bound the second term, $\left|R_{emp}(h)-\mathbb{E}[R_{emp}(h)]\right|$, which represents the deviation of the random variable $R_{emp}(h)$ from its expected value.

$$ $\begin{split}\left|R_{\text{emp}}(h)-\mathbb{E}[R_{\text{emp}}(h)]\right|&= \left|\frac{1}{n}\sum_{i=1}^{n}\mathrm{loss}(h,Z_{i})-\mathbb{E}\left[\frac{1} {n}\sum_{i=1}^{n}\mathrm{loss}(h,Z_{i})\right]\right|.\end{split}$ (39) $$

Here, the randomness comes from the training data $T_{1:n}$. Since the sequence $\{Z_{i}\}$ is $\beta$-mixing, the sequence of random variables $\{loss(A(T_{1:n}),Z_{i})\}$ is also a dependent sequence.

We can apply a concentration inequality designed for $\beta$-mixing sequences (e.g., a variant of Bernstein’s or Hoeffding’s inequality). For any $\gamma>0$, such an inequality takes the form:

$$ $\Pr\left[\left|R_{emp}(h)-\mathbb{E}[R_{emp}(h)]\right|\geq\gamma\right]\leq 2 \exp\left(-\frac{n\gamma^{2}}{C(\sigma^{2},B)}\right)$ (40) $$

where $C(\sigma^{2},B)$ is a constant that depends on the variance upper bound $\sigma^{2}$ and the sum of mixing coefficients $B$. A common form is $C(\sigma^{2},B)=2\sigma^{2}(1+4B)$. Thus, we have:

$$ $\Pr\left[\left|R_{emp}(h)-\mathbb{E}[R_{emp}(h)]\right|\geq\gamma\right]\leq 2 \exp\left(-\frac{n\gamma^{2}}{2\sigma^{2}(1+4B)}\right)$ (41) $$

Combining the Bounds
Now, we combine the results. We want the total error to be bounded with high probability, at least $1-\delta$. From the concentration inequality in Step 2, we set the probability upper bound to $\delta$:

$$ $\delta=2\exp\left(-\frac{n\gamma^{2}}{2\sigma^{2}(1+4B)}\right)$ (42) $$

Solving for $\gamma$, we get the bound on the concentration term:

$$ $\gamma=\sqrt{\frac{2\sigma^{2}(1+4B)\ln(2/\delta)}{n}}$ (43) $$

This means that, with probability at least $1-\delta$, we have:

$$ $\left|R_{emp}(h)-\mathbb{E}[R_{emp}(h)]\right|\leq\sqrt{\frac{2\sigma^{2}(1+4B )\ln(2/\delta)}{n}}$ (44) $$

Combining this high-probability bound with the deterministic bound from Step 1, we obtain the final result:

$$ $\begin{split}G_{n}(h)&\leq\left|R(h)-\mathbb{E}[R_{\text{emp}}(h)]\right|+ \left|R_{\text{emp}}(h)-\mathbb{E}[R_{\text{emp}}(h)]\right|\\ &\leq 2\varepsilon_{\text{stab}}+\sqrt{\frac{2\sigma^{2}(1+4B)\log(2/\delta)}{ n}}.\end{split}$ (45) $$

This inequality holds with probability at least $1-\delta$.
∎

###### Definition A.3 (Patch Construction) .

###### Definition A.4 (Quantization) .

###### Lemma A.2 (Patch β 𝛽 \beta italic_β -Mixing Preservation) .

###### Proof.

###### Remark A.7 (Non-overlapping Condition) .

###### Remark A.8 (Overlapping Case) .

###### Remark A.9 (Quantization Independence) .

###### Theorem A.4 (Dependence Generalisation Bound) .

###### Proof.

### A.4 Non-Decreasing Task-Relevant Mutual Information

###### Theorem A.5 (Patch Representation as an Effective Information Bottleneck) .

Let $X$ be an input signal, $Y$ be the task label, $Z_{\text{point}}$ be the pointwise representation of $X$, and $Z_{\text{patch}}$ be the patch-based quantized representation. Under the following conditions:

- 1.
The input signal can be decomposed as $X=S+N$, where $S$ is the task-relevant signal and $N$ is independent noise with $S\perp N$.
- 2.
The noise is weakly informative about the task: $I(N;Y)\leq\epsilon\cdot I(S;Y)$ for some small $\epsilon>0$.
- 3.
The patching operation has a denoising effect: $\frac{H(N|Z_{\text{patch}})}{H(N)}>\frac{H(S|Z_{\text{patch}})}{H(S)}$.

Then there exists a range $\beta\in[\beta_{\min},\beta_{\max}]$ such that:

$$ $\mathcal{L}_{IB}(Z_{\text{patch}})<\mathcal{L}_{IB}(Z_{\text{point}})$ (46) $$

where $\mathcal{L}_{IB}(Z)=I(X;Z)-\beta\cdot I(Y;Z)$ is the Information Bottleneck Lagrangian.

###### Proof.

We provide a constructive proof by analyzing the difference in Lagrangian values.

Decomposition of Mutual Information.
Since $X=S+N$ with $S\perp N$, and $Z_{\text{point}}\approx X$, we have:

$$ $\displaystyle I(X;Z_{\text{point}})$ $\displaystyle=I(S+N;Z_{\text{point}})\approx H(S)+H(N)$ (47) $\displaystyle I(Y;Z_{\text{point}})$ $\displaystyle=I(Y;S+N)=I(Y;S)+I(Y;N)\leq I(Y;S)(1+\epsilon)$ (48) $$

where the second line uses the independence of $S$ and $N$, and condition 2.

Analysis of Patch Representation.
For the patch representation, by the data processing inequality:

$$ $\displaystyle I(X;Z_{\text{patch}})$ $\displaystyle=H(X)-H(X|Z_{\text{patch}})$ (49) $\displaystyle=H(S)+H(N)-H(S|Z_{\text{patch}})-H(N|Z_{\text{patch}})$ (50) $$

By condition 3 (denoising effect), let $\alpha_{S}=\frac{H(S|Z_{\text{patch}})}{H(S)}$ and $\alpha_{N}=\frac{H(N|Z_{\text{patch}})}{H(N)}$ with $\alpha_{N}>\alpha_{S}$. Then:

$$ $I(X;Z_{\text{patch}})=(1-\alpha_{S})H(S)+(1-\alpha_{N})H(N)$ (51) $$

Task-Relevant Information Preservation.
Since the patching operation primarily affects the noise component:

$$ $\displaystyle I(Y;Z_{\text{patch}})$ $\displaystyle\geq I(Y;S|Z_{\text{patch}})$ (52) $\displaystyle\geq(1-\delta)I(Y;S)$ (53) $$

where $\delta>0$ is a small constant representing the information loss due to quantization of the signal component.

Comparison of Lagrangians.
The difference in Lagrangian values is:

$$ $\displaystyle\Delta\mathcal{L}$ $\displaystyle=\mathcal{L}_{IB}(Z_{\text{point}})-\mathcal{L}_{IB}(Z_{\text{ patch}})$ (54) $\displaystyle=[I(X;Z_{\text{point}})-I(X;Z_{\text{patch}})]-\beta[I(Y;Z_{\text {point}})-I(Y;Z_{\text{patch}})]$ (55) $\displaystyle\geq\alpha_{S}H(S)+\alpha_{N}H(N)-\beta[\epsilon+\delta]I(Y;S)$ (56) $$

Existence of Optimal $\beta$.
For $\Delta\mathcal{L}>0$, we need:

$$ $\beta<\frac{\alpha_{S}H(S)+\alpha_{N}H(N)}{[\epsilon+\delta]I(Y;S)}$ (57) $$

Since $\alpha_{N}>\alpha_{S}$ and typically $H(N)$ is substantial in real signals, the numerator is positive and significant. Given that $\epsilon$ and $\delta$ are small, there exists a non-trivial range of $\beta$ values, specifically:

$$ $\beta\in\left(0,\frac{\alpha_{S}H(S)+\alpha_{N}H(N)}{[\epsilon+\delta]I(Y;S)}\right)$ (58) $$

for which $Z_{\text{patch}}$ achieves a better (lower) Lagrangian value than $Z_{\text{point}}$.
∎

###### Remark A.10 .

This result formalizes the intuition that patch-based representations excel when:

- •
The input contains significant noise ($H(N)$ is large)
- •
The noise is largely task-irrelevant ($\epsilon$ is small)
- •
The patching operation effectively denoises ($\alpha_{N}>\alpha_{S}$)

The optimal $\beta$ range depends on the signal-to-noise characteristics and the denoising effectiveness of the patching operation.

###### Theorem A.6 ( $\varepsilon$ -MI Preservation via Lipschitz Continuity) .

Let $Z_{\text{pt}}$ be the continuous (pointwise) representation and $Z_{\varepsilon}=Q_{\varepsilon}(Z_{\text{pt}})$ be its quantized version, satisfying $\|Z_{\text{pt}}-Z_{\varepsilon}\|_{2}\leq\varepsilon$. Let the model be a probabilistic classifier where the conditional probability $P(Y|Z)$ is generated from a logit function $f(Z)$ followed by a softmax. Assume the logit function $f:\mathbb{R}^{P}\to\mathbb{R}^{|Y|}$ is $L_{f}$-Lipschitz continuous.

Then, the loss in mutual information is bounded:

$$ $I(Y;Z_{\varepsilon})\geq I(Y;Z_{\text{pt}})-C\cdot\varepsilon$ (59) $$

where $C$ is a constant dependent on the model’s Lipschitz constant and the number of task classes, for instance, $C=L_{f}\log(|Y|-1)$ under certain tight bounding assumptions.

Underlying Assumptions:

- •
Bounded Quantization Error: There exists a fixed $\varepsilon>0$ such that for any $Z_{\text{pt}}$, we have $\|Z_{\text{pt}}-Q_{\varepsilon}(Z_{\text{pt}})\|_{2}\leq\varepsilon$.
- •
Probabilistic Model: The model’s conditional probability distribution $P(Y|Z)$ is generated by a softmax applied to a logit function, i.e., $P(Y|Z)=\mathrm{softmax}(f(Z))$.
- •
Model Smoothness: The logit function $f$ is $L_{f}$-Lipschitz continuous with respect to the $L_{2}$ norm. This is a common assumption for robust models.

###### Proof.

The proof proceeds by bounding the change in conditional entropy, which arises from the quantization error, through a chain of Lipschitz continuity arguments.

Mutual Information Difference Decomposition.
We begin with the standard definition of mutual information, $I(Y;Z)=H(Y)-H(Y|Z)$. The difference can be expressed precisely as:

$$ $I(Y;Z_{\text{pt}})-I(Y;Z_{\varepsilon})=H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}} )=\mathbb{E}[H(Y|Z_{\varepsilon})]-\mathbb{E}[H(Y|Z_{\text{pt}})]$ (60) $$

Our goal is to find an upper bound for the right-hand side, which requires bounding the term $|H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})|$.

Bounding the Change in Conditional Entropy.
We establish a “continuity propagation chain” from the input representation $Z$ to the conditional entropy $H(Y|Z)$.

- (a)
From Input to Logits: By the Lipschitz assumption on the logit function $f$, the quantization error $\varepsilon$ bounds the change in the logits:
$\|f(Z_{\varepsilon})-f(Z_{\text{pt}})\|_{2}\leq L_{f}\|Z_{\varepsilon}-Z_{
\text{pt}}\|_{2}\leq L_{f}\varepsilon.$
(61)
- (b)
From Logits to Probabilities (TV Distance): The softmax function is also Lipschitz. It can be shown that the Total Variation (TV) distance between two output probability distributions is bounded by the difference in their input logits.
$\mathrm{TV}(P_{Y|Z_{\varepsilon}},P_{Y|Z_{\text{pt}}})\leq\frac{1}{2}\|f(Z_{
\varepsilon})-f(Z_{\text{pt}})\|_{1}\leq\frac{\sqrt{|Y|}}{2}\|f(Z_{\varepsilon
})-f(Z_{\text{pt}})\|_{2}\leq\frac{\sqrt{|Y|}}{2}L_{f}\varepsilon.$
(62)
This shows that a small perturbation in $Z$ leads to a proportionally small change in the conditional probability distribution.
- (c)
From Probabilities to Entropy: The entropy function $H(P)=-\sum p_{i}\log p_{i}$ is Lipschitz continuous over the probability simplex. Its Lipschitz constant, $L_{H}$, with respect to the TV distance (or L1 norm), can be bounded, e.g., by $L_{H}\leq\log(|Y|-1)$. Thus, the change in entropy is bounded by the change in the probability distribution:
$|H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})|=|H(P_{Y|Z_{\varepsilon}})-H(P_{Y|Z_{
\text{t}}})|\leq L_{H}\cdot\mathrm{TV}(P_{Y|Z_{\varepsilon}},P_{Y|Z_{\text{pt}
}}).$
(63)

Combining the Bounds.
By chaining the inequalities from equation [62](https://arxiv.org/html/2507.00078v1#A1.E62) and equation [63](https://arxiv.org/html/2507.00078v1#A1.E63), we get a direct bound on the change in entropy for any given point:

$$ $|H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})|\leq\log(|Y|-1)\cdot\frac{\sqrt{|Y|}} {2}L_{f}\varepsilon.$ (64) $$

Let’s define the constant $C=L_{f}\log(|Y|-1)\frac{\sqrt{|Y|}}{2}$ (or a tighter version thereof). We have $|H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})|\leq C\cdot\varepsilon$.

Returning to the mutual information difference in equation [60](https://arxiv.org/html/2507.00078v1#A1.E60), we take the expectation over all possible values. By linearity of expectation and Jensen’s inequality:

$$ $\begin{split}\left|I(Y;Z_{\text{pt}})-I(Y;Z_{\varepsilon})\right|&=\left| \mathbb{E}\left[H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})\right]\right|\\ &\leq\mathbb{E}\left[\left|H(Y|Z_{\varepsilon})-H(Y|Z_{\text{pt}})\right| \right]\\ &\leq\mathbb{E}[C\cdot\varepsilon]=C\cdot\varepsilon.\end{split}$ (65) $$

This yields the final result, $I(Y;Z_{\text{pt}})-I(Y;Z_{\varepsilon})\leq C\cdot\varepsilon$, which can be rewritten as:

$$ $I(Y;Z_{\varepsilon})\geq I(Y;Z_{\text{pt}})-C\cdot\varepsilon.$ (66) $$

∎

###### Theorem A.5 (Patch Representation as an Effective Information Bottleneck) .

###### Proof.

###### Remark A.10 .

###### Theorem A.6 ( ε 𝜀 \varepsilon italic_ε -MI Preservation via Lipschitz Continuity) .

###### Proof.

## Appendix B Datasets Overview

This study utilizes a comprehensive collection of 19 time series datasets spanning multiple domains, totaling 31,479,451 data points across 1,758,768 temporal observations. The datasets encompass various temporal resolutions from minute-level to annual scales, providing diverse patterns for time series analysis and forecasting tasks.

**Table 2: Overview of Time Series Datasets**
| Dataset | Shape (L×C) | Domain | Description |
| --- | --- | --- | --- |
| Air Quality | 9,357 × 14 | Environmental | Hourly air quality measurements including CO, benzene, NOx, and meteorological variables |
| Electricity Demand | 230,736 × 6 | Energy | Electricity consumption across Australian states (NSW, VIC, QUN, SA, TAS) with temporal patterns |
| WTH | 35,064 × 13 | Environmental | Comprehensive weather dataset with temperature, humidity, pressure, and wind measurements |
| Wind Power | 493,144 × 2 | Energy | High-frequency (1-minute) wind power generation data for renewable energy analysis |
| ETTh1 | 17,420 × 8 | Energy | Electricity Transformer Temperature dataset with hourly readings from power grid infrastructure |
| ETTh2 | 17,420 × 8 | Energy | Secondary electricity transformer temperature dataset with complementary power grid measurements |
| Electricity | 26,304 × 322 | Energy | Large-scale electricity consumption dataset covering 321 consumers over extended time period |
| Exchange Rate | 7,588 × 9 | Financial | Daily foreign exchange rates for multiple currency pairs in international markets |
| Traffic | 17,544 × 863 | Transportation | Highway traffic flow measurements from 862 sensors monitoring vehicle occupancy rates |
| River Flow | 23,741 × 2 | Environmental | Daily river discharge measurements for hydrological modeling and water resource management |
| TCPC | 52,416 × 9 | Energy | Temperature-correlated power consumption with environmental factors and zonal energy usage |
| Energy | 19,735 × 27 | Energy | Building energy consumption with appliance usage, lighting, and multi-zone temperature/humidity data |
| Weather | 52,696 × 22 | Environmental | Extended meteorological dataset with atmospheric pressure, solar radiation, and precipitation data |
| Sunspot | 73,924 × 2 | Astronomical | Solar activity measurements tracking sunspot numbers for space weather analysis |
| National Illness | 966 × 8 | Healthcare | Weekly influenza-like illness surveillance data across age groups and healthcare providers |
| Metro | 48,204 × 2 | Transportation | Urban metro system passenger traffic volume with temporal ridership patterns |
| ETTm1 | 69,680 × 8 | Energy | Minute-resolution electricity transformer temperature data for fine-grained power grid monitoring |
| Solar Power | 493,149 × 2 | Energy | High-frequency (1-minute) solar power generation data for photovoltaic system analysis |
| ETTm2 | 69,680 × 8 | Energy | Secondary minute-resolution transformer dataset providing additional power infrastructure insights |

## Appendix C Detailed Dataset Descriptions

### C.1 Environmental Domain Datasets

Air Quality Dataset: Contains hourly measurements of atmospheric pollutants and meteorological conditions collected from urban monitoring stations. Key variables include carbon monoxide (CO), benzene (C6H6), nitrogen oxides (NOx), and various sensor readings for pollution monitoring, alongside temperature, relative humidity, and absolute humidity measurements.

WTH (Weather) Dataset: Provides comprehensive meteorological observations including dry and wet bulb temperatures in both Fahrenheit and Celsius, dew point measurements, relative humidity, wind speed and direction, atmospheric pressure readings, and visibility conditions.

River Flow Dataset: Records daily streamflow measurements essential for hydrological modeling, flood prediction, and water resource management. The time series captures seasonal variations and extreme events in riverine systems.

Extended Weather Dataset: Features detailed atmospheric measurements including barometric pressure, potential temperature, vapor pressure components, specific humidity, water vapor concentration, air density, wind velocities, precipitation data, and solar radiation parameters.

### C.2 Energy Domain Datasets

Electricity Demand Dataset: Captures electricity consumption patterns across five Australian states, providing insights into regional energy usage, demand forecasting, and grid management strategies.

Wind Power Dataset: High-resolution (1-minute interval) measurements of wind power generation, crucial for renewable energy integration, grid stability analysis, and short-term power forecasting applications.

Solar Power Dataset: Minute-level solar photovoltaic power output data enabling fine-grained analysis of solar energy patterns, cloud intermittency effects, and renewable energy variability studies.

ETT (Electricity Transformer Temperature) Datasets: Four complementary datasets (ETTh1, ETTh2, ETTm1, ETTm2) monitoring transformer temperatures at hourly and minute resolutions. These datasets are fundamental for power grid health monitoring, predictive maintenance, and electrical infrastructure management.

Large-scale Electricity Dataset: Encompasses consumption data from 321 individual consumers, providing a comprehensive view of distributed electricity usage patterns suitable for demand response analysis and consumer behavior modeling.

TCPC (Temperature-Correlated Power Consumption): Integrates environmental factors with power consumption across multiple zones, including temperature, humidity, wind speed, and diffuse radiation measurements alongside zonal energy usage data.

Building Energy Dataset: Detailed energy consumption monitoring of residential appliances and lighting systems, complemented by multi-zone temperature and humidity sensors, outdoor weather conditions, and building environmental parameters.

### C.3 Transportation Domain Datasets

Traffic Dataset: Comprehensive highway traffic monitoring system covering 862 sensor locations, measuring vehicle occupancy rates and traffic flow patterns essential for intelligent transportation systems and congestion management.

Metro Dataset: Urban public transportation ridership data capturing passenger traffic volumes in metropolitan transit systems, valuable for public transportation planning and urban mobility analysis.

### C.4 Financial Domain Datasets

Exchange Rate Dataset: Daily foreign exchange rate fluctuations for multiple international currency pairs, providing data for financial market analysis, currency risk assessment, and economic forecasting models.

### C.5 Healthcare Domain Datasets

National Illness Dataset: Weekly surveillance data tracking influenza-like illness (ILI) prevalence across different age demographics and healthcare provider networks, supporting epidemiological research and public health monitoring.

### C.6 Astronomical Domain Datasets

Sunspot Dataset: Long-term solar activity observations recording sunspot numbers, essential for space weather prediction, satellite operations planning, and understanding solar-terrestrial interactions.

## Appendix D Dataset Statistics Summary

The complete dataset collection comprises:

- •
Total temporal observations: 1,758,768 time points
- •
Total data points: 31,479,451 (L × C)
- •
Temporal resolutions: 1-minute to weekly intervals
- •
Domain coverage: 7 distinct application areas
- •
Dimensionality range: 2 to 863 features per dataset
- •
Estimated storage requirement: 240.2 MB

The datasets provide extensive coverage across critical infrastructure sectors, environmental monitoring systems, and socio-economic indicators, making them suitable for comprehensive time series analysis, multivariate forecasting, and cross-domain pattern recognition research.

## Appendix E Qualitative Analysis of the Learned Temporal Vocabulary

To qualitatively understand the vocabulary discovered by our data-driven approach, Figure [7](https://arxiv.org/html/2507.00078v1#A5.F7) visualizes the complete set of 32 cluster centroids, or “temporal motifs”, learned from the dataset. The motifs are sorted in descending order of their frequency of occurrence (denoted by $n$), providing a clear view into the structural composition of the “language of time”.

##### A Hierarchy from Simple States to Complex Events.

A striking feature revealed in Figure [7](https://arxiv.org/html/2507.00078v1#A5.F7) is the emergent hierarchy of pattern complexity. The most frequent motifs, displayed in the top row, represent simple and fundamental states. For instance, Cluster 18 ($n=4452$) corresponds to a near-zero stable signal, while Cluster 21 ($n=2738$) and Cluster 1 ($n=2694$) represent high and medium constant values, respectively. These high-frequency patterns can be interpreted as the “grammatical” or functional components of the temporal language, akin to articles or prepositions in natural language, forming the stable background upon which more complex dynamics unfold.

Conversely, as we proceed to motifs with lower frequencies (middle and bottom rows), the patterns exhibit significantly greater complexity and convey more specific dynamic information. We can clearly identify distinct archetypes corresponding to fundamental temporal behaviors:

- •
Trends and Slopes: Gentle upward (e.g., Cluster 2) and downward (e.g., Cluster 25) trends.
- •
Troughs and Peaks: U-shaped valleys (e.g., Cluster 10, 13) and bell-shaped crests (e.g., Cluster 28).
- •
Sharp Transitions: Rapid state changes, such as sharp rising edges (e.g., Cluster 16), S-shaped transitions (e.g., Cluster 17), and step-like functions (e.g., Cluster 22).

These rarer, more complex motifs act as the “semantic” core of the vocabulary, analogous to content-rich nouns and verbs that describe specific, meaningful events within the time series.

##### Qualitative Validation of the Linguistic Analogy.

The structure of this learned vocabulary provides strong qualitative validation for our central hypothesis. The inverse relationship between pattern complexity and frequency—whereby simple, foundational patterns are ubiquitous and complex, event-specific patterns are rare—aligns perfectly with the quantitative findings of Zipf’s Law presented in our earlier analysis. The ability to automatically discover such a rich, interpretable, and comprehensive lexicon from raw data demonstrates that complex time series dynamics are indeed compositional. This confirms that a finite set of reusable motifs forms the basis of observed signals, providing a solid foundation for treating time series analysis as a language modeling task.

Figure: Figure 7: The Learned Vocabulary of Temporal Motifs: Visualizing the 32 Cluster Centers. This figure displays the 32 cluster centers, or ’temporal motifs,’ learned by the K-Means algorithm (K=32) from time series patches of length 16. Each plot represents a single prototypical pattern. The plots are sorted in descending order based on their frequency of occurrence (cluster size, denoted by $n$), from the most common (Cluster 18, top-left) to the rarest (Cluster 7, bottom-right).
Refer to caption: x7.png