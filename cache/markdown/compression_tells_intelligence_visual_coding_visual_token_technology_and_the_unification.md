## Contents
- Index Terms:
- 1 Introduction
- 2 Classical Visual Coding and Codecs
  - 2.1 Related Core Techniques
  - 2.2 Architectures of Visual Coding
    - 2.2.1 Traditional Codec
    - 2.2.2 Neural Codec
  - 2.3 Image Codec
    - 2.3.1 Traditional Image Codec
    - 2.3.2 Learned Image Codec
  - 2.4 Video Codec
    - 2.4.1 Traditional Video Codec
    - 2.4.2 Learned Video Codec
  - 2.5 Semantic Codec
    - 2.5.1 Human-Perception-Oriented Coding
    - 2.5.2 Machine-Vision-Oriented Coding
- 3 VISUAL TOKEN TECHNOLOGY OF MLLMs
  - 3.1 Overview
    - 3.1.1 Visual Tokenization
    - 3.1.2 Visual Token Compression
    - 3.1.3 Cross-Modal Token Fusion and Reasoning
  - 3.2 Architectures of visual tokenizers
    - 3.2.1 Continuous tokenizers
    - 3.2.2 Discrete tokenizers
  - 3.3 Generation Task
  - 3.4 Understanding Task
    - 3.4.1 Motivation
    - 3.4.2 Compact Tokenization and Compression for Visual Understanding
  - 3.5 Unified Tokenizer
    - 3.5.1 Task-Specific Tokenizer: Understanding vs. Generation
    - 3.5.2 Dual-Branch Cooperative Framework
    - 3.5.3 Single Unified Tokenizers
    - 3.5.4 Toward the Ideal Unified Tokenizer
- 4 Bridging Visual Coding and Visual Tokens: A Unified Perspective
  - 4.1 Unified Formulation
  - 4.2 Information Theory Aspect: Shannon Entropy vs. Semantic Entropy
    - 4.2.1 Unified Perspective
    - 4.2.2 Visual Coding: Minimizing Shannon Entropy
    - 4.2.3 MLLM Tokens: Minimizing Semantic Entropy
  - 4.3 Functionality Aspect: Redundancy Reduction vs. Context Modeling
    - 4.3.1 Unified Perspective
    - 4.3.2 Visual Coding: Redundancy Reduction
    - 4.3.3 MLLM Tokens: Context Modeling
  - 4.4 Optimization Aspect: R-D Trade-off vs. Information Bottleneck
    - 4.4.1 Unified Perspective
    - 4.4.2 Visual Coding: Rate-Distortion (R-D) Trade-off
    - 4.4.3 MLLM Tokens: Information Bottleneck
  - 4.5 Objective Aspect: Human Eye Fidelity vs. Machine Task Analysis
    - 4.5.1 Unified Perspective
    - 4.5.2 Visual Coding: Human Eye Fidelity
    - 4.5.3 MLLM Tokens: Machine Task Analysis
  - 4.6 How Visual Coding Principles Can Refine Token Technology
  - 4.7 How Token Technology Can Refresh Codecs
- 5 Application and Outlook
  - 5.1 Next-generation Token Applications
    - 5.1.1 Token Technology in AIGC
    - 5.1.2 Token Technology in Embodied AI
    - 5.1.3 Categorization and Transferability of Visual Tokenizers
  - 5.2 Next-generation Codec Applications
    - 5.2.1 Immersive Media
    - 5.2.2 MLLMs for Codec
    - 5.2.3 Video Coding for Machine (VCM)
  - 5.3 Unified Communication System in The LLM Era
    - 5.3.1 Traditional Communication
    - 5.3.2 Semantic Communication
    - 5.3.3 Token Communication
- 6 Conclusion
- Acknowledgments
- References

## Abstract

Abstract “Compression Tells Intelligence”, is supported by research in artificial intelligence, particularly concerning (multimodal) large language models (LLMs/MLLMs), where compression efficiency often correlates with improved model performance and capabilities. For compression, classical visual coding based on traditional information theory has developed over decades, achieving great success with numerous international industrial standards widely applied in multimedia (e.g., image/video) systems. Except that, the recent emergingvisual token technology of generative multi-modal large models also shares a similar fundamental objective like visual coding: maximizing semantic information fidelity during the representation learning while minimizing computational cost. Therefore, this paper provides a comprehensive overview of two dominant technique families first – Visual Coding and
Vision Token Technology – then we further unify them from the aspect of optimization, discussing the essence of compression efficiency and model performance trade-off behind. Next, based on the proposed unified formulation bridging visual coding andvisual token technology, we synthesize bidirectional insights of themselves and forecast the next-gen visual codec and token techniques.
Last but not least, we experimentally show a large potential of the task-oriented token developments in the more practical tasks like multimodal LLMs (MLLMs), AI-generated content (AIGC), and embodied AI, as well as shedding light on the future possibility of standardizing a general token technology like the traditional codecs (e.g., H.264/265) with high efficiency for a wide range of intelligent tasks in a unified and effective manner.

###### Index Terms:

## 1 Introduction

Acompelling principle is emerging: “Compression Tells Intelligence” [72]. This perspective holds that the essence of intelligence is the ability to form compact and effective representations of the world by identifying, modeling, and exploiting patterns within data. The recent success of Large Language Models (LLMs) [7, 114] provides strong validation for this concept. Their extraordinary capabilities in reasoning, generation, and in-context learning stem directly from their ability to compress vast linguistic data into powerful internal representations. *As a result, compression efficiency has evolved from a simple engineering metric for storage and bandwidth into a fundamental benchmark for a model’s depth of understanding and intelligence.*

This core philosophy naturally extends to the visual domain, where it has inspired two distinct, but strongly related, lines of technological development. The first is Classical Visual Coding [167, 206, 16, 15, 9, 30]. Grounded in information theory, this field has a long history of success, producing international standards from JPEG [206] to H.265/HEVC [16]. These technologies excel at minimizing statistical redundancy to achieve the highest possible pixel-level fidelity, and they form the foundation of our modern multimedia ecosystem. The second line of development is the recently emerging Visual Token Technology [168, 251, 179, 3], which has emerged alongside generative AI and Multimodal Large Language Models (MLLMs) [7, 123, 29]. Unlike classical coding, the primary goal of visual tokens is not the perfect reconstruction of pixels, but rather the extraction of crucial semantic information for downstream tasks like visual question answering or image generation. Despite their different approaches, both classical coding and vision technology share the same objective: *to find an optimal balance between information fidelity and computational cost.*

Despite this shared goal, these two technical families have evolved almost entirely independently. They are pursued by different academic communities (Signal Processing vs. Machine Learning), are based on different theoretical principles (Information Theory vs. Representation Learning), and are evaluated by different criteria (e.g., visual quality vs. downstream task accuracy). This divergence extends to the very purpose of compression itself. Classical coding primarily aims to reduce data size for efficient storage and transmission, thus saving bandwidth. In contrast, visual token technology seeks to create a compact sequence of representations to reduce the computational cost of learning processing by large-scale models like Transformers. This separation has triggered a significant gap. Classical codecs, optimized to minimize bit-rate against signal fidelity, offer unparalleled compression efficiency but their representations are not inherently designed for direct use in AI model architectures. Conversely, visual tokens are explicitly designed to produce compact feature sets that reduce computational load and improve model performance, yet they currently lack the theoretical rigor and compression rates of traditional methods. *We argue that bridging this gap is essential. A unified framework would allow to understand the fundamental trade-off between compression efficiency and model performance more deeply, paving the way for the next generation of visual intelligence.*

As shown in Fig. [1](https://arxiv.org/html/2601.20742v1#S1.F1) to bridge this gap and foster innovation between the fields, this paper makes the following key contributions and organizes our paper as follows:

- •
Section II&III: We provide the first systematic review that connects the fields of classical visual coding and emerging visual token technologies, outlining their histories, core principles, and key techniques.
- •
Section IV&V: We propose a theoretical framework that unifies the goals of both visual coding and visual token technology from different perspectives. Based on the unified framework, we distill key insights that allow each field to re-formula and improve the other, forecasting the next-gen visual coding and visual token technology.
- •
Section VI: We demonstrate the significant potential of compression technology, particularly focusing on the fast-developing visual token skills rather than well-standardized visual coding, on system-level real-world applications, including MLLMs, AIGC, and Embodied AI.

Figure: Figure 1: The overall organization of this paper.
Refer to caption: x1.png

## 2 Classical Visual Coding and Codecs

Classical visual coding [167, 130, 150, 104] seeks to create compact representations of visual data, minimizing the required bits while preserving essential information. This core pursuit facilitates efficient storage and transmission across digital platforms. All realizations share the same three technique primitives: transformation for decorrelation/energy compaction, quantization for discretization and rate control, and entropy coding for lossless compression of syntax symbols. This section provides an overview of classical visual coding, starting with the fundamental techniques, followed by different architectures, specific codec instances for images and videos, and finally, the emerging area of semantic coding.

### 2.1 Related Core Techniques

The foundational principles of nearly all visual coding systems, both traditional and learned, revolve around three core techniques. Transformation is employed to decorrelate the visual data and compact its energy into a smaller set of coefficients [206, 130, 30]. Common transforms include the Discrete Cosine Transform (DCT) in JPEG and many video codecs [74, 107], and more recently, learned non-linear transforms using autoencoders in neural codecs. Quantization is the process of reducing the precision of the transformed coefficients, which is the primary source of lossy compression [35, 201]. This step is crucial for controlling the bitrate. Entropy coding [151, 88, 140], such as Huffman coding or arithmetic coding, is the final stage, where the quantized symbols are losslessly compressed by assigning shorter codes to more probable symbols.

### 2.2 Architectures of Visual Coding

#### 2.2.1 Traditional Codec

Traditional codecs, like those famous standards of JPEG, JPEG 2000, HEVC, VVC, etc., [167, 6, 16, 15] are built upon hand-crafted modules that are individually optimized. They typically follow a block-based hybrid coding framework, especially for video. This architecture involves prediction (either spatial for intra-frames or temporal for inter-frames), transformation of the residual, quantization, and entropy coding.

#### 2.2.2 Neural Codec

Learned codecs [130, 104, 140], referred to as neural codecs, replace the hand-crafted components of traditional codecs with deep neural networks. These architectures are trained end-to-end, typically using an autoencoder framework for the transform and learned priors for entropy modeling. This data-driven approach allows for more powerful and adaptive modeling of complex visual data.

### 2.3 Image Codec

Figure: Figure 2: A taxonomy of modern video coding paradigms, categorized by their different optimization objectives. The left branch represents traditional and neural codecs optimized for pixel fidelity (e.g., JPEG [206], PNG [13], HEVC [16], VVC [15], DVC [137], DCVC [107], etc). The right branch focuses on coding for human perception (e.g., PerCo [19], Diffeic [116], DiffC [198], MS-ILLM [154]) and coding for machine tasks (e.g., Channel Selection [128], TransTIC [28], Adapt-ICMH [105]).
Refer to caption: x2.png

#### 2.3.1 Traditional Image Codec

JPEG is the canonical lossy image standard: images are divided into $8{\times}8$ blocks, transformed by a DCT, coefficients are zig-zag scanned, run-length coded, quantized, and entropy-coded via Huffman/arithmetic coding [206].
JPEG2000 improves upon JPEG by using a Discrete Wavelet Transform (DWT), which provides better compression performance and features like scalability and region-of-interest coding.
For lossless coding, PNG applies predictive filtering followed by DEFLATE [61]. These systems exemplify the classical transform–quantize–entropy pipeline.

#### 2.3.2 Learned Image Codec

Learned image codecs retain the same three primitives but replace hand-crafted parts with data-driven ones: an autoencoder provides the transform; (soft) quantization or vector quantization discretizes latents; and a learned prior—commonly a hyperprior and/or an autoregressive or attention-based context model—predicts symbol probabilities for the entropy coder [9, 150, 30]. Foundational works proposed hyperprior models that leverage side information for more accurate entropy estimation of the latent codes [9, 150]. Subsequent research introduced improved network architectures and transforms [30]. Recent systems (e.g., ELIC and successors) push state-of-the-art rate–distortion, often rivaling or surpassing VVC on standard datasets [66, 130]. These architectures were trained end-to-end to optimize for metrics like PSNR or MS-SSIM. Moreover, even though scalar quantization is known to be suboptimal in the literature, due to the high computational complexity of classical vector quantization schemes in information theory and the difficulty of estimating the rate-distortion function, learned compression algorithms based on neural networks [235, 96] have been studied.
Such a learned neural compressor can even approximately recover the optimal vector quantization performance at reasonable complexity [97].

### 2.4 Video Codec

#### 2.4.1 Traditional Video Codec

Modern block-based hybrids (e.g., HEVC/H.265 [16] and VVC/H.266) [15] extend the image pipeline with motion-compensated prediction (temporal), rich intra prediction (spatial), hierarchical block/tree partitioning (e.g., CTU/QTMT), in-loop filtering (deblocking, SAO), and context-adaptive binary arithmetic coding (CABAC) [16, 15]. Transform choice is typically integerized DCT/DST variants; quantization uses rate–distortion optimized decisions; entropy coding leverages context models tied to local syntax structure. These standards have pushed the rate-distortion (R-D) performance frontier under metrics like PSNR.

#### 2.4.2 Learned Video Codec

Learned video codecs (e.g., the DCVC series [107, 74]) model motion and residuals directly in the *feature/latent* domain via learned warping/conditioning, with recurrent or GOP-structured inference; the entropy term becomes the cross-entropy of quantized latents under conditional priors, enabling strong rate savings with real-time throughput on GPUs [74]. Similarly, in video, learned approaches have demonstrated superior performance over traditional standards while maintaining practical inference speeds [74].

### 2.5 Semantic Codec

Historically, the definition of ”essential information” has been tightly coupled with mathematical, pixel-level fidelity. However, the field is undergoing a significant transformation, with optimization objectives evolving from simple pixel-level accuracy to encompass more sophisticated goals tailored for human perception and machine-vision tasks. This has led to the development of semantic codecs, as shown in Fig [2](https://arxiv.org/html/2601.20742v1#S2.F2).

#### 2.5.1 Human-Perception-Oriented Coding

While pixel-based metrics are computationally convenient, they often correlate imperfectly with subjective quality perceived by human observers. This mismatch motivated a perceptual optimization paradigm, where the objective shifts from minimizing mathematical distortion to maximizing visual realism and appeal.

This paradigm is closely tied to generative models, which can synthesize natural-looking textures and details that pixel-wise losses tend to suppress. GAN-based approaches [1] were among the first to demonstrate this advantage, producing reconstructions that are often subjectively preferred over PSNR-optimized counterparts, even when PSNR is lower [148]. More recently, diffusion models have pushed perceptual compression further, achieving state-of-the-art performance and generating high-fidelity, visually pleasing images even at very low bitrates [146, 116, 154, 198, 19]. These methods explicitly prioritize plausible synthesis over exact reconstruction, representing a clear departure from pixel-fidelity-oriented optimization.

#### 2.5.2 Machine-Vision-Oriented Coding

The most recent and transformative paradigm shift in visual coding is driven by the proliferation of machine-centric applications. In this context, the ultimate consumer of the visual data is not a human, but an AI model performing a task like classification, detection, or segmentation. Consequently, the optimization objective moves away from both pixel fidelity and human perception, focusing instead on preserving the semantic information essential for machine tasks [28].

The goal becomes maximizing task accuracy for a given bitrate, leading to a rate-accuracy trade-off. Semantic codecs are designed to identify and allocate more bits to features critical for machine analysis while aggressively compressing irrelevant background information [103, 129]. Some approaches have demonstrated the benefit of jointly optimizing the compression model and the downstream task network, further improving machine task performance [106]. To accommodate diverse use cases, scalable bitstreams have been developed that can flexibly serve both human and machine needs from a single compressed representation [125, 79]. This evolution has culminated in industry-led standardization initiatives, such as MPEG’s Video Coding for Machines (VCM) [260] and JPEG AI [6], which aim to create a unified framework that recognizes both perceptual and semantic goals in next-generation codecs.

## 3 VISUAL TOKEN TECHNOLOGY OF MLLMs

Figure: Figure 3: Pipeline of (visual) token technology, typically used in the mainstream (multi-modal) large language models (LLMs/MLLMs). Visual inputs are first converted into *visual tokens* by a visual tokenizer, which may be either *continuous* (patchify + linear projection with positional encoding, as in CLIP/SigLIP/DINOv2) or *discrete* (latent encoding and codebook quantization, as in VQ-VAE/VQ-GAN), thereby forming transformer-ready sequences. A subsequent *visual token compression* stage (e.g., attention-, similarity-, query-, pooling-, or RL-based) reduces the visual tokens to a small budget that, together with text tokens, feeds the *token reasoning* module for cross-modal fusion and inference. Arrows indicate the data flow from tokenization to compression, then reasoning.
Refer to caption: x3.png

### 3.1 Overview

We organize *visual token technologies* into three stages of the multimodal pipeline (Fig. [3](https://arxiv.org/html/2601.20742v1#S3.F3)): 1). visual tokenization; 2).visual token compression; and 3).cross-modal fusion and reasoning. Tokenization is either *continuous*, mapping images/videos to patch- or region-level embeddings for attention backbones, or *discrete*, quantizing latents into codebook “words” that form compact symbolic sequences amenable to generative and autoregressive modeling. Because visual tokens usually dominate sequence length, compression reduces $N$ to $K$ ($K\!\ll\!N$), improving latency/throughput, reducing memory and KV-cache, and expanding spatial–temporal context under fixed compute. Importantly, $K$ is an explicit *interface constraint*: compressors expose at most $K$ tokens, and downstream connectors, query bottlenecks, attention patterns, and decoders are designed and evaluated under this fixed allowance. In Sec. [3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2), we systematize compressors by *objective*, *mechanism*, *training regime*, *location*, *guidance*, and *schedule*, and detail complexity and memory trade-offs. With fixed $K$, cross-modal reasoning scales as $\mathcal{O}(MK)$ per layer for $M$ text tokens. We highlight three operating modes: *understanding* (image/video $\rightarrow$ text) via lightweight connectors or learned queries, *generation* (text $\rightarrow$ image/video or editing) via AR or hybrid AR–diffusion decoders conditioned on $K$ tokens, and *unified* models that read and emit visual tokens in an interleaved sequence. This framing links upstream tokenization to downstream reasoning through the compression budget, motivating the design choices and evaluation criteria developed next.

**TABLE I: Representative *visual token compression* methods positioned along six axes (compact labels defined above).**
| Method (abbr.) | Goal | Mechanism | Training | Location | Guidance | Schedule |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-OCR [214] | Long. | Trans. | E2E | Bridge | Vision | Static |
| VoCo-LLaMA [242] | Accel+Mem/KV | Query | Post. | LM | Vision | Static |
| DynamicViT [170] | Accel. | Attn. | E2E | Enc. | Vision | Dynamic |
| FastV [26] | Accel+Mem/KV | Attn. | TF | LM | Text | One-shot |
| IVTP [70] | Accel. | Attn. | Post. | LM | Text | Dynamic |
| VisionZip [231] | Accel. | Attn. | Post. | Bridge | Vision | One-shot |
| PruneVid [71] | Accel+Long | Sim. | TF | Hybrid | Text | Dynamic |
| HoliTom [182] | Accel. | Sim+Attn | TF | Hybrid. | Vision | Dynamic |
| RL4EViT [136] | Accel. | RL | RL | Enc. | Vision | Dynamic |
| VisionThink [232] | Accel. | RL | RL | Enc. | Text | Dynamic |
| VisPruner [258] | Accel. | Attn+Sim | TF | LM | Vision | One-shot |
| DivPrune [3] | Accel. | Sim | TF | LM | Vision | One-shot |
| LLaVA-UHD [63] | Accel+Long | Trans+Query | E2E | Bridge | Vision | Static |

#### 3.1.1 Visual Tokenization

Visual tokenization converts images/videos into transformer-compatible token sequences [205]. Methods largely fall into two families. *Continuous* tokenizers partition inputs into patches (or regions) and map each unit to an embedding via linear projection with positional encoding, as in ViT-style backbones and representation learners such as CLIP [168], SigLIP [202], and DINOv2 [158]. *Discrete* (codebook-based) tokenizers encode inputs into latents and quantize them into codebook indices, as in VQ-VAE [204] and VQ-GAN [48]; the resulting sequences are typically modeled with autoregressive or diffusion priors. Continuous tokenizers dominate perception/understanding pipelines and supply the main compressible visual inputs to LVLMs [210], whereas discrete tokenizers provide learned visual vocabularies central to high-fidelity generation. From an information-theoretic view, both can be cast as learned source coding: a transform stage (e.g., convolutional [157] or linear [200] encoders), a quantization/selection stage (e.g., codebooks [261] or structured resampling [2]), and a probabilistic modeling stage (e.g., LLMs [149] or diffusion models [229]). This lens links token formation to downstream fusion and generation, supporting more unified evaluation across understanding- and generation-centric systems.

#### 3.1.2 Visual Token Compression

Figure: Figure 4: Six–axis view of visual token compression. The center *Goal* (acceleration, memory/KV reduction, long-context) is realized by choices along five orthogonal axes: *Methodology* (attention, similarity, RL, query, transformation), *Training pattern* (TF, Post, E2E, RL), *Location* (encoder, bridge, LM/KV, hybrid), *Guidance* (vision, text, hybrid), and *Schedule* (static, one-shot, dynamic, progressive). Arrows indicate how these factors compose into a concrete compression policy under a fixed visual budget $K$; the taxonomy matches Sec. [3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2) and Table [I](https://arxiv.org/html/2601.20742v1#S3.T1).
Refer to caption: x4.png

Problem setup & scope.
In LVLMs [63, 122, 211], visual tokens often dominate the multimodal sequence. For an $H{\times}W$ image with patch size $p{\times}p$, $N_{\text{img}}=\frac{HW}{p^{2}}$; for a $T$-frame video, $N_{\text{vid}}=T\cdot\frac{HW}{p^{2}}$. This growth rapidly exhausts attention and KV-cache budgets, becoming a primary latency/memory bottleneck. *Visual token compression* maps an original set of $N$ visual tokens to a smaller, task-faithful set of $K$ ($K\!\ll\!N$), reducing per-layer cross-attention from $\mathcal{O}(MN)$ to $\mathcal{O}(MK)$ (for $M$ active text tokens) and vision self-attention from $\mathcal{O}(N^{2})$ to $\mathcal{O}(K^{2})$, while retaining the information needed for fusion and reasoning.

Goal (why compress).
We emphasize three objectives. *Accel.* reduces wall-clock time and FLOPs by shrinking the active visual sequence (e.g., merging/pruning), improving prefill cost (e.g., ToMe [10]). *Mem/KV* targets peak VRAM and KV-cache footprint via early, decode-consistent reduction (e.g., TopV [228]), remaining compatible with FlashAttention [38]/KV paging. *Long.* extends admissible spatial–temporal context under fixed resources, e.g., fixed per-frame budgets in long-video LVLMs (LLaMA-VID [114]).

Mechanism (how to compress).
We group techniques into five families that trade a large visual token set for a compact representation. *a. Similarity-based.* Merge redundant tokens to preserve coverage with fewer representatives: canonical ToMe-style merging [10], late-stage merging (FOLDER [208]), diversity-driven subset selection (DivPrune [3]), and stage-wise fusion (AuroraCap [22]); video variants combine spatial merging with temporal redundancy control (Chat-UniVi [78], FastVID [183], DynTok [254]). *b. Attention-based.* Use attention-derived salience to keep informative tokens and prune the rest, including encoder-side hierarchical schemes (VisPruner [258], MustDrop [133], VScan [252], HiReD [5], GlobalCom^2 [134]) and LVLM-side, layer-aware schedules or learned thresholds (PyramidDrop [225], VTW [120], FitPrune [240], ST^3 [271], ATP-LLaVA [241], ZipVL [67]), often preserving VQA/Caption quality with small $K$. *c. Query-based.* Replace dense tokens with a fixed-$K$ bottleneck that reads vision features via cross-attention and forwards only summaries to the LM, as in Flamingo [2] and BLIP-2 [108], with extensions such as InstructBLIP [37], mPLUG-Owl [239], MiniGPT-4 [270], Victor [215], and extreme pre-fusion interfaces (LLaVA-Mini [259]); video instantiations constrain tokens per frame or distill into LM-internal codes (BLIP-3-Video [174], Long-VMNet [64]). With architectural $K$, fusion cost is near-constant as resolution or $T$ grows. *d. Transformation-based.* Reduce tokens at ingress by changing the sampling before fusion (downsampling, pyramids, learned pooling), e.g., pooling-based front-ends (LLaVA-OneVision [100], DeCo [237]), multi-granular pooling (M^3 [18]), lightweight conv abstractors (Honeybee C-Abstractor [21], MobileVLM LDP [33]), and tiling with compression heads for UHD inputs (NVLM [36]). *e. RL-based.* Formulate compression as budgeted decision-making with rewards balancing fidelity and efficiency: multi-agent layer-wise pruning (RL4EViT [136]), event-triggered streaming controllers (MARC [218]), and instance-adaptive policies for difficult inputs (VisionThink [232]).

Training (whether to train). *TF* methods are plug-and-play (PruMerge [179], SparseVLM [262], LLaVA-Scissor [189]). *Post* tunes only lightweight selectors/bridges while freezing backbones (TokenPacker [113], VideoChat-Flash [144]). *E2E* co-optimizes intake/projectors so the model natively operates at small $K$ (LLaVA-NeXT [122], VideoLLaMA-2 [264], TimeChat [171]). *RL* learns budget-aware policies with explicit efficiency–accuracy rewards (MARC [218], RL4EViT [136]).

Location $\mathcal{L}$ (where to act).
*Enc.* compresses before fusion (PatchMerger [172]); *Bridge* retokenizes to a compact interface (Kosmos2-style connectors [160], Perceiver-IO resampling [73]); *LM* prunes inside the decoder (SparseVLM [262]); *KV* directly controls cache growth (VL-Cache [203], LOOK-M [207]); *Hybrid* coordinates stages for large reductions (VideoChat-style systems [145]).

Guidance (who guides).
*Vision guidance* uses bottom-up cues (vid-TLDR [31], VisionDrop [255]); *Text guidance* conditions selection on instructions (LVPruning [192], CTFP [142]); *Hybrid* combines both (PTP [118], TCR [86]).

Schedule (when compress).
*Static* uses fixed ratios/budgets (TCR [86], vid-TLDR [31]);
*One-shot* prunes once at prefill for decode-consistent savings (LLaVA-Scissor [189]);
*Dynamic* adapts budgets per input/prompt (LVPruning [192], Dynamic-VLM [132]);
*Progressive* sparsifies across layers/depths (CoViPAL [226], FEATHER [46]).

#### 3.1.3 Cross-Modal Token Fusion and Reasoning

Problem framing.
With a compressed visual budget of $K$ tokens, fusion must present these tokens to the LLM so cross-modal interactions scale as $\mathcal{O}(MK)$ per layer, where $M$ is the active text length. Architectures mainly differ in (i) how visual tokens enter the decoder (connectors, learned query bottlenecks, or two-stream encoders) and (ii) how the decoder reasons over mixed evidence (attention, grounded pointers, or program/tool execution).

Understanding (image/video $\rightarrow$ text).
Two-stream encoders (ViLBERT [138], LXMERT [194]) established cross-attention blueprints for vision–language alignment. Modern LVLMs either (a) project vision features into the LLM token space with lightweight connectors (e.g., LLaVA [123, 124]) or (b) aggregate dense features via learned query bottlenecks before the LLM (Flamingo [2], BLIP-2 [108]). For video, pre-projection alignment can stabilize fusion under small per-frame budgets (Video-LLaVA [119]). Grounded pointers such as location tokens (Kosmos-2 [160]) make spatial references explicit, while program/tool routes (ViperGPT [193]) externalize long reasoning when attention alone is insufficient.

Generation (text $\rightarrow$ image/video, or editing).
Autoregressive (AR) generators interleave text and visual tokens in a single next-token stream (CM3LeOn [246]), supporting prompting, infilling, and editing. Hybrid AR+diffusion designs retain one Transformer trunk but attach diffusion-style heads for higher-fidelity synthesis (Show-o [223]), allowing a shared backbone across understanding and generation. Strong visual priors can further improve conditional generation when the model is constrained to an informative budget of $K$ tokens (Emu [191]).

Unified understanding & generation (U&G in one model).
Unified models adopt a shared token space/objective so one network both *reads* and *emits* visual tokens. Purely autoregressive formulations perform next-token prediction over interleaved text and quantized visual tokens (VILA-U [219], Chameleon [196]). Scaling decoder-only pretraining on interleaved corpora further improves both capabilities and induces stronger multimodal reasoning (BAGEL [41]). Hybrid designs combine autoregressive decoding with flow matching or diffusion-style generation under a shared Transformer trunk (Show-o2 [224]). Inference attends jointly to $K$ visual tokens and $M$ text tokens, enabling grounded instruction following and multi-step reasoning without separate understanding/generation models.

### 3.2 Architectures of visual tokenizers

Visual tokenizers determine the mechanism by which raw visual signals are converted into latent representations that are suitable for downstream processing, including both generation and understanding tasks. Architecturally, they convert a spatially structured signal into a sequence of tokens that align with the transformer or diffusion interface. Current designs fall into two categories, *continuous* and *discrete* tokenizers, each characterized by distinct design philosophie, objectives, and downstream compatibilities.

#### 3.2.1 Continuous tokenizers

Continuous tokenizers, also known as vision encoder, are designed to convert visual patches into continuous embedding vectors via vision transformers and linear projections, incorporating positional encodings. These embeddings remain differentiable and support gradient-based optimization throughout multimodal pretraining. During the visual understanding era, they have become the predominant choice in MLLMs for image and video understanding, including representative approaches like the vision encoders of LLaVA[123]. The tokenizer produces high-dimensional latent sequences whose embeddings are compatible with those of text embeddings for direct ingestion by the LLM.

Continuous visual tokenizers form the architecture of nearly all modern MLLMs. Their design inherits from ViT[43] and contrastive representation learners such as CLIP[168] and SigLIP[202]. A typical tokenizer consists of three stages: (i) patchification, splitting the image into non-overlapping patches of size $P\times P$; (ii) projection, flattening and linearly mapping each patch into a $d$ dimensional embedding; and (iii) positional encoding, injecting spatial or temporal information. The resulting sequence of embeddings $X_{V}\in\mathbb{R}^{N\times d}$, where $N=H\times W/P^{2}$.

In MLLMs, the tokenizer is typically initialized from a pre-trained visual encoder (e.g., CLIP ViT-L/14) then frozen or fine-tuned. Continuous tokens are aligned with text embeddings thus semantically rich. Representative approaches include LLaVA[123], which relies on a frozen CLIP encoder and a subsequent projection module for cross-modal alignment; BLIP-2[108], which builds its vision tower upon a frozen CLIP encoder with Q-Former; and Qwen2-VL[210], which introduces a dynamic-resolution vision transformer for flexible visual encoding. In the video domain, methods such as VideoLLaMA[253] and InternVideo[212] extend image-based encoders with spatio-temporal patching and using a projection modules to maintain alignment across frames.
Continuous tokenizers are differentiable and easily integrated into multimodal backbones, but they may produce redundancy at high resolution and lack explicit interpretability.

#### 3.2.2 Discrete tokenizers

Discrete tokenizers, in contrast, encode visual inputs into indices of a learned codebook, thereby producing a symbolic token sequence. This quantization process such as in VQ-VAE[204], VQ-GAN[48], bridges perception and generation: each index corresponds to a visual word learned from the data. In diffusion or autoregressive generative models, such discrete tokenizers serve as visual vocabulary or compressesor. In MLLMs, discrete tokenization is less commonly used for understanding tasks, but is typically adopted in unified modeling frameworks that represent both images and text as discrete sequences under a shared embedding space.

Discrete visual tokenizers initially designed for generative models, have recently gained attention in multimodal understanding and unified modeling. Their core idea is to represent an image $x$ by a sequence of quantized codes $\{z_{i}\}_{i=1}^{N}$, where each $z_{i}\in\{1,\dots,K\}$ indexes a learned codebook $\mathcal{C}\in\mathbb{R}^{K\times d}$. The encoder $E_{\phi}$ maps $x$ to latent features, which are then quantized to the nearest codebook entry:

$$ $z_{i}=\arg\min_{k}\|E_{\phi}(x)_{i}-\mathcal{C}_{k}\|_{2}^{2},\quad\hat{x}=D_{\psi}(\mathcal{C}_{z_{1}},\ldots,\mathcal{C}_{z_{N}}).$ $$

This pipeline, introduced by VQ-VAE and refined by VQ-GAN, forms the foundation of many discrete vision language systems.

In MLLMs, discrete tokenizers treats images as symbolic sequences as the text do. For instance, CM3[246], MAGVIT2[245] unify text and images modalities by mapping images into discrete tokens and jointly training a transformer to autoregress over text–image sequences. This enables bidirectional tasks such as captioning, visual question answering, and text-to-image generation within a single language backbone. Compared to continuous tokenizers, discrete tokenizers demonstrate superior compatibility with LLM architecture, produce more compact representation, but require non-differentiable quantization steps and codebook maintenance. They also tend to lose fine-grained details critical for dense reasoning.
*Overall*, continuous tokenizer is more lossless and suitable for understanding tasks, dominate current MLLM architectures for reasoning. Discrete tokenizers is more efficient and compatible for unified generative frameworks. The boundary between them is increasingly blurred as hybrid systems adopt quantized continuous latents or learnable patch embeddings jointly optimized with language supervision.

**TABLE II: Representative visual tokenizers, classified by the model architectures and target tasks they were designed for.**
| Method | Continuous / Discrete | MLLM / Diffusion | Generation / Understanding |
| --- | --- | --- | --- |
| CLIP[168] | Continuous | MLLM | Understanding |
| SigLIP[202] | Continuous | MLLM | Understanding |
| BLIP-2[108] | Continuous | MLLM | Understanding |
| LLaVA[123] | Continuous | MLLM | Understanding |
| Qwen2-VL[7] | Continuous | MLLM | Understanding |
| VideoLLaMA[253] | Continuous | MLLM | Understanding |
| InternVideo[212] | Continuous | MLLM | Understanding |
| VQ-VAE[204] | Discrete | Diffusion / MLLM | Generation |
| VQ-GAN[48] | Discrete | Diffusion / MLLM | Generation |
| DALL$\cdot$E[169] | Discrete | MLLM | Generation |
| MaskGIT[23] | Discrete | MLLM | Generation |
| MAGVIT2[245] | Discrete | MLLM | Both Gen & Understanding |
| CM3[246] | Discrete | MLLM | Both Gen & Understanding |
| LDM[173] | Continuous | Diffusion | Generation |
| REPA[248] | Continuous | Diffusion | Generation |
| RAE[267] | Continuous | Diffusion | Generation |

### 3.3 Generation Task

Figure: Figure 5: Comparison of Discrete vs. Continuous Image Tokenization Paradigms. (A) Discrete Tokenization: The input image is encoded into dense vectors and then quantized using a learnable codebook to produce discrete indices ($z_{q}$). These tokens are processed by a discrete prior model (e.g., AR [247] Transformer [205]). (B) Continuous Tokenization: The encoder maps the image directly to continuous latent variables ($z$) without quantization. These latents are modeled by a continuous prior, such as a Diffusion Model [68] or Flow Matching [121]. Both frameworks utilize conditioning inputs (e.g., text) and a decoder for image reconstruction.
Refer to caption: x5.png

To situate the discussion, Fig. [5](https://arxiv.org/html/2601.20742v1#S3.F5) compares the two dominant paradigms. Discrete tokenization (Fig. [5](https://arxiv.org/html/2601.20742v1#S3.F5)A) maps encoder features to finite codebook indices, suitable for autoregressive or masked priors. Continuous tokenization (Fig. [5](https://arxiv.org/html/2601.20742v1#S3.F5)B) maps images to smooth latent spaces, enabling diffusion or flow-based priors. This dichotomy underpins modern generative pipelines.

Discrete tokenization quantizes images into index grids. VQ-VAE enabled visual sequence modeling [204], while VQGAN enhanced fidelity via adversarial objectives [48]. These naturally pair with discrete priors: DALL·E employs autoregressive transformers for open-vocabulary generation [169], whereas MaskGIT uses masked parallel decoding to improve efficiency [23]. Recent advances like TiTok and VAR further optimize throughput for high resolutions [247, 199]. While discrete formulations offer entropy coding, they suffer from codebook pathologies and sequence inflation at high resolutions, requiring hierarchical mitigations.

Continuous tokenization replaces quantization with smooth, expressive latents. RAEs simplify training by pairing frozen pretrained encoders with lightweight decoders [267], and REPA improves DiT-style models via feature alignment [248]. Stable Diffusion performs denoising in perceptually regularized latent spaces [173], supported by accelerated samplers. The differentiable nature of continuous latents facilitates gradient-based editing and parameter-efficient adaptation. However, the absence of quantization complicates exact bit-level accounting, and sampling costs can dominate without accelerated objectives.

Across both families, controllability is achieved via cross-modal connectors or structural branches, which enforce geometry and layout [256], while video extensions utilize spatiotemporal grids and consistency losses to maintain coherence. The choice represents a trade-off: discrete tokenization offers compactness and probabilistic clarity at the cost of quantization artifacts, whereas continuous tokenization prioritizes smoothness and editability at the expense of sampling complexity. Hybrid designs are increasingly adopted to combine the structural advantages of discrete compressibility with the alignment benefits of continuous spaces.

### 3.4 Understanding Task

Figure: Figure 6: Visual token technologies for understanding tasks. (a) Continuous visual tokenization [43, 168, 202]represents visual inputs using compact latent embeddings. (b) Discrete visual tokenization [204, 244, 245, 56] encodes images and videos into compact symbolic tokens. Integrated token compression [265, 209, 55] complements both by controlling token quantity for scalable multimodal reasoning.
Refer to caption: x6.png

#### 3.4.1 Motivation

Visual images and videos contain rich spatial and temporal information but also substantial redundancy, making direct pixel-level modeling inefficient. To address this, recent work focuses on compact visual token representations that transform raw pixels into low-dimensional latent tokens, preserving essential semantics and dynamics while discarding irrelevant details. As shown in figure [6](https://arxiv.org/html/2601.20742v1#S3.F6), an encoder typically converts inputs into spatial or spatio-temporal tokens for downstream understanding. Within this paradigm, two challenges arise: defining effective representational units and controlling token quantity for scalable reasoning. Together, these enable efficient visual understanding based on compact, semantically meaningful tokens rather than dense pixels.

#### 3.4.2 Compact Tokenization and Compression for Visual Understanding

The development of compact visual tokens forms a continuum from representation learning to efficient reasoning.
Early visual transformers such as ViT [43] and CLIP [168] represented images as fixed-size patch embeddings, which captured spatial structure but produced hundreds of redundant tokens per frame. Subsequent research introduced learned tokenizers—such as VQ-VAE [204], VQGAN [48], and MAGVIT [244]—that replaced raw embeddings with discrete codebook indices, offering compact symbolic representations compatible with transformer and language-model-based architectures. These methods achieved strong perceptual compression while preserving object and texture semantics. Continuous and hybrid designs, such as CV-VAE [265], OmniTokenizer [209], and BSQ-ViT [266], further improved reconstruction fidelity and cross-modal alignment by encoding fine-grained visual context into continuous latent spaces. For visual understanding, these compact image tokens act as semantic building blocks, capturing scene layout and entity relationships in a format readily interpretable by multimodal reasoning models.

Extending spatial compression into the temporal dimension introduces new challenges—motion coherence, temporal redundancy, and causal consistency. Early video extensions of image tokenizers simply applied 2D encoders frame by frame, resulting in redundant and temporally inconsistent tokens. To overcome this, models such as TATS [56], MAGVIT-V2 [245], and CogVideoX [236] employ spatio-temporal quantization, learning shared codebooks or latent VAEs that jointly model spatial appearance and temporal dynamics. Diffusion-oriented systems like OpenSora [268], CV-VAE [265], and HunyuanVideo [85] further compress videos into continuous latent sequences suitable for generative or reasoning backbones. These tokenizers aim for semantic sufficiency—capturing key entities, motions, and interactions—rather than pixel-perfect detail, which is redundant for understanding tasks. Hybrid tokenizers (e.g., LinVT [55], TVC [269]) explicitly balance reconstruction quality with token compactness by combining discrete quantization with continuous temporal compression, providing representations that are both efficient and expressive across modalities.

Even with efficient tokenizers, the total number of tokens in long or high-resolution videos often exceeds the processing limits of large transformers. To mitigate this, compression mechanisms are increasingly integrated into tokenization pipelines. Transformation-based methods (e.g., PLLaVA [227], VideoLLaMA-2 [264]) employ learnable pooling or convolution layers to summarize local regions or temporal segments into fewer tokens while preserving coarse semantics. Similarity-based clustering (e.g., Chat-UniVi [78], FastVID [183], HoliTom [182]) merges highly correlated frame or patch tokens based on feature proximity, reducing redundancy while maintaining contextual continuity. Attention-guided pruning (e.g., VisionZip [231], FastV [26]) removes low-saliency tokens using attention scores or importance maps, while query-driven selection (e.g., Token Turing Machines [175], BLIP-3-Video [174], LongVU [184]) selectively retains tokens relevant to a textual or task-driven query. In practice, these techniques are often jointly optimized with the encoder, producing adaptive compression that responds to content and task complexity.

Across benchmarks [249, 109, 52], studies show that retaining only 25–35% of tokens preserves over 95% of reasoning accuracy, indicating that most pixel-level details are redundant for semantic understanding. Current research therefore focuses on task-aware and adaptive tokenization, dynamically allocating representational capacity to semantically important regions or temporal segments.

### 3.5 Unified Tokenizer

Figure: Figure 7: Overview of task-specific tokenizers (understanding-oriented [168, 251, 11] and generation-oriented [204, 48]), dual-branch cooperative frameworks [196, 223], and single unified tokenizers [219, 143, 139]. This taxonomy illustrates the evolution from separated representations toward unified visual tokenization.
Refer to caption: x7.png

To contextualize the evolution of visual tokenization, Fig. [7](https://arxiv.org/html/2601.20742v1#S3.F7) summarizes three major paradigms:
(1) task-specific tokenizers designed separately for understanding or generation,
(2) dual-branch cooperative frameworks that combine both types of representations, and
(3) unified tokenizers that aim to bridge semantic alignment and pixel-level reconstruction within a shared latent space.
Next, we discuss each category in detail.

#### 3.5.1 Task-Specific Tokenizer: Understanding vs. Generation

As discussed above, most existing visual tokenizers are task-specific.
One family, represented by understanding-oriented tokenizers (usually in continuous form) such as CLIP [168], SigLIP [251, 202], Perception Encoder [11], and DINO[20, 158], is trained through image–text alignment and excels in multimodal reasoning tasks such as VQA[4] and image captioning[188]. However, the lack of Generation supervision leads to clear bottlenecks in image generation and editing[27].
Another family, represented by generation-oriented tokenizers (usually in discrete form) such as VQ-VAE[204] and VQ-GAN[48], enables high-fidelity image synthesis, generation, and editing. Nevertheless, their latent spaces are not semantically aligned with language, which limits their generalization to cross-modal understanding tasks. In addition, these models often require large-scale joint training data to align the latent space with downstream models.

#### 3.5.2 Dual-Branch Cooperative Framework

As multimodal large models continue to evolve toward unified understanding and generation, this dichotomy has become increasingly restrictive.
Recent works demonstrate two opposing tendencies.
Models such as chameleon[196] and Show-o[223] adopt generation-oriented tokenizers that achieve strong reconstruction fidelity and detailed generation but lack semantic alignment and controllability in multimodal reasoning.
In contrast, Models such as BLIP3-o[24] and EMU2[190] adopt understanding-oriented tokenizers. Some require the large language model to directly predict continuous visual embeddings that interact with a diffusion module, while others quantize semantic features through a codebook mechanism to support generation. Despite these efforts, such methods still suffer from representation distortion, modality mismatch, and semantic drift.

Beyond these two extremes, an emerging line of research explores dual-branch cooperative frameworks that jointly leverage both understanding- and generation-oriented tokenizers within a unified model.
Representative works such as Janus [217] and BAGEL [41] adopt a hybrid design, where a understanding-oriented tokenizer and a generation-oriented tokenizer are employed in parallel.
While this design generally delivers stable performance, but several limitations remain.
First, maintaining two different types of tokenizers simultaneously greatly increases the number of visual tokens in the input sequence, leading to higher inference latency and memory overhead.
Moreover, since the two branches are often pretrained with different objectives, their latent distributions may gradually diverge, resulting in semantic–visual inconsistency.
Consequently, recent studies have begun to explore a new unified paradigm: employing a single Unified Tokenizer that achieves both semantic alignment and detail-preserving reconstruction within a shared latent space.

#### 3.5.3 Single Unified Tokenizers

As one of the earliest attempts toward a unified visual tokenizer, VILA-U[219] builds upon the VQ-VAE framework and introduces contrastive learning between discrete visual tokens and text tokens, thereby enabling both visual understanding and generation within a single model.
UniTok[143] points out that this joint training paradigm is difficult to stabilize, as the two objectives often interfere with each other, leading to minimal improvement in understanding but substantial degradation in generation. Further analysis reveals that the issue does not stem from conflicting tasks, but rather from the limited expressiveness of the discrete token space, which fails to capture the semantics required for understanding. To address this, UniTok proposes a Multi-Codebook Quantization Mechanism that expands the codebook capacity and dimensionality to enhance the representational power of discrete features.
TokenFlow[166] builds upon the features obtained from understanding- and generation-oriented tokenizers, and employs a Shared Mapping to project them into a Semantic Codebook and a Pixel Codebook, respectively. While this design facilitates cross-modal alignment, the shared mapping may not yield the optimal correspondence for either semantic abstraction or texture fidelity.
DualToken[187] introduces a hierarchical design within a single model: shallow layers are responsible for predicting pixel tokens, while deeper layers predict semantic tokens. During interaction with the LLM, the two token types are concatenated along the embedding dimension; during decoding, a Pixel Head and a Semantic Head are used for generation and understanding, respectively.
Finally, AToken[139] identifies that many previous methods suffer from architectural inconsistency and modality-specific limitations. It proposes a fully Transformer-based unified tokenizer applicable to images, videos, and 3D scenes. By leveraging 4D Rotary Position Embedding (4D RoPE), AToken achieves both semantic understanding and high-fidelity reconstruction within a shared latent space, marking a key step toward true multimodal unification.

#### 3.5.4 Toward the Ideal Unified Tokenizer

An ideal unified visual tokenizer should achieve an intrinsic unification of understanding and generation within a shared latent space, balancing semantic alignment with high-fidelity reconstruction.
Rather than maintaining the dichotomy between semantic-oriented and reconstruction-oriented tokenizers, it should encode visual content into representations that are simultaneously interpretable to language models and reversible to pixel-level details.

From a system perspective, such a tokenizer is expected to simultaneously satisfy several key properties: it must enable understanding–generation compatibility, supporting both high-level semantic reasoning and precise visual synthesis within a shared representational space; it should be modality-extensible, featuring a modular and flexible architecture that can be seamlessly extended to new modalities such as video, 3D, audio, or action; it needs to ensure semantic consistency and reversibility, so that encoding and decoding preserve stable semantic correspondences and maintain alignment between understanding and generation; and it must achieve efficiency and compactness, avoiding the redundancy and latency inherent in dual-branch designs through compact token representations and shared computation. In essence, an ideal unified tokenizer is not a simple fusion of understanding and generation, but a semantically reversible, modality-agnostic, and structurally efficient representation mechanism that lays the foundation for truly unified multimodal intelligence.

## 4 Bridging Visual Coding and Visual Tokens: A Unified Perspective

In this section, we bridge the gap between classical visual coding techniques and the visual token mechanisms employed in Multimodal Large Language Models (MLLMs). Despite their disparate origins—visual coding rooted in signal processing and compression standards, and MLLM tokens emerging from generative AI architectures—both paradigms share the common goal of representing visual information efficiently while preserving essential fidelity. We unify their principles through four key aspects: information theory, functionality, optimization, and objectives. This unification not only highlights intrinsic connections but also paves the way for cross-domain innovations in token technology.

Figure: Figure 8: Analogy and comparison between Classical Visual Coding (top) and Visual Token Technology of MLLMs (bottom). Despite distinct operational modules of these two technologies, both paradigms align under a shared functional workflow: transforming raw inputs into latent representations (Representation Transformation), compressing information by discarding non-essential details (Redundancy Reduction), and capturing dependencies for downstream reconstruction or reasoning (Context Modeling).
Refer to caption: x8.png

To intuitively illustrate this connection, Fig. [8](https://arxiv.org/html/2601.20742v1#S4.F8) presents a parallel view of the two paradigms. We map the distinct modules of classical coding (Transform, Quantization, Predictive & Entropy Encoding) and MLLM processing (Tokenization, Token Compression, Token Reasoning) onto three shared functional stages: Representation Transformation, Redundancy Reduction, and Context Modeling.
In the classical view (top), raw pixels are transformed and quantized to remove statistical redundancy, resulting in a compact bitstream. Similarly, in the MLLM view (bottom), visual patches are tokenized and compressed to filter out semantic redundancy, forming a sequence of tokens ready for reasoning. This visual juxtaposition underscores that while the outputs differ—bitstreams for human perception versus semantic vectors for machine reasoning—the underlying structural logic remains remarkably consistent.

### 4.1 Unified Formulation

Here, we first try to understand *Visual Token Technology* within the Multi-Modal Large Language Model (MLLM) pipeline [123, 270, 7] through the lens of the Information Bottleneck (IB) principle [59, 9, 150]. The core physical idea is that an optimal visual tokenizer must act as a strategic compressor similar to *Visual Coding*, discarding irrelevant pixel-level details while zealously preserving the semantic information essential for downstream tasks like visual question answering.

Let $X$ denote the raw visual input and $Z$ represent the compressed visual tokens. The original information bottleneck objective can be formulated as:

$$ $\min_{p(z|x)}I(X;Z)-\beta I(Z;Y),$ (1) $$

where $I(\cdot;\cdot)$ denotes mutual information, $Y$ is the target task, and $\beta$ controls the trade-off between compression and preservation. The fundamental IB objective, $\min I(X;Z)-\beta I(Z;Y)$, formalizes this trade-off: the first term $I(X;Z)$ represents the compression rate, penalizing the number of bits used to describe $X$ via $Z$, while the second term $I(Z;Y)$ is the relevance, rewarding the preservation of information about $Y$. The Lagrange multiplier $\beta$ controls the balance between these two competing goals; a high $\beta$ favors more descriptive but less compressed tokens.

Based on the classic theory above, we can reform the process of Visual Tokenization as a compression problem. Specifically, the raw visual input $X$ is information-rich but highly redundant and often contains nuisances like texture and lighting variations. The goal is to find a compressed representation $Z$ (the visual tokens) that is maximally informative about the target task $Y$. So, we view the visual tokenization process $f_{\theta}:X\rightarrow Z$ as an information bottleneck:

$$ $Z=f_{\theta}(X)=\arg\min_{Z}\underbrace{\mathcal{L}_{\text{comp}}}_{\text{Compression}}+\lambda\underbrace{\mathcal{L}_{\text{task}}}_{\text{Task Preservation}}$ (2) $$

Specifically:

$$ $\displaystyle\mathcal{L}_{\text{comp}}$ $\displaystyle=\mathbb{E}_{x\sim p_{\text{data}}}[\|x-g_{\phi}(f_{\theta}(x))\|^{2}],$ (3) $\displaystyle\mathcal{L}_{\text{task}}$ $\displaystyle=-\mathbb{E}_{(x,y)\sim p_{\text{data}}}[\log p_{\psi}(y|f_{\theta}(x))],$ (4) $$

where $g_{\phi}$ is a reconstruction decoder and $p_{\psi}$ is the task predictor. The visual tokenization function $f_{\theta}$ is thus conceptualized as an optimizer of this objective, practically achieved by minimizing a combination of a compression loss $\mathcal{L}_{\text{comp}}$ and a task preservation loss $\mathcal{L}_{\text{task}}$. The compression loss, often realized as a reconstruction error, ensures the tokens do not stray too far from the input data manifold, while the task loss, typically a cross-entropy loss, forces the tokens to be discriminative for the ultimate goal.

In this way, the optimal token representation $Z^{*}$ needs satisfy:

$$ $p^{*}(z|x)=\frac{p^{*}(z)}{K(x,\beta)}\exp\left(-\beta\mathbb{E}_{y\sim p(y|x)}[D_{\text{KL}}(p(y|x)\|p(y|z))]\right),$ (5) $$

where $K(x,\beta)$ is the normalization partition function. $p^{*}(z|x)$ reveals that the probability of a token $z$ given an input $x$ is proportional to its prior probability $p^{*}(z)$ re-weighted by an exponential factor of how well the token-induced conditional distribution $p(y|z)$ matches the true data distribution $p(y|x)$, with $\beta$ acting as the tuning parameter for this matching fidelity.

Moreover, to quantify the effectiveness of the tokenizer, the token efficiency ratio $\eta_{\text{token}}$ can be defined as the amount of task-relevant information per bit of compression, providing a single metric to evaluate different tokenization schemes:

$$ $\eta_{\text{token}}=\frac{I(Z;Y)}{I(X;Z)}=\frac{\mathbb{E}_{z,y}[\log\frac{p(y|z)}{p(y)}]}{\mathbb{E}_{x,z}[\log\frac{p(z|x)}{p(z)}]},$ (6) $$

which connects directly to the classical rate-distortion (R-D) theory, where the function $R(D)$ defines the fundamental limit of compression (the rate $R$) for a given maximum allowable distortion $D$ in reconstructing the input or its semantics.

Thus, for visual tokenizers, the rate-distortion trade-off follows:

$$ $R(D)=\min_{p(z|x):\mathbb{E}[\Delta(X,g(Z))]\leq D}I(X;Z),$ (7) $$

where $\Delta$ is the distortion measure and $D$ is the maximum allowable distortion.

For hierarchical tokenization with scales $s_{1}<s_{2}<\cdots<s_{k}$, the information preservation is decomposed across hierarchical scales; the mutual information $I(X;Y)$ is approximated by a weighted sum of the information at each scale $I(Z_{s_{i}};Y)$ minus the redundant information shared between consecutive scales $I(Z_{s_{i}};Z_{s_{i+1}})$, ensuring that each level of the hierarchy captures unique and complementary semantic information, thereby achieving a more efficient and powerful visual representation for the MLLM:

$$ $I(X;Y)\approx\sum_{i=1}^{k}\alpha_{i}I(Z_{s_{i}};Y)-\sum_{i=1}^{k-1}\beta_{i}I(Z_{s_{i}};Z_{s_{i+1}}),$ (8) $$

where $\alpha_{i}$ and $\beta_{i}$ control information flow between scales.

In this way, we understand and formulate the popular visual token technology from the IB aspect, revealing its nature close to visual coding that pursues an information trade-off. Based on it, in the following section, we discuss visual coding and visual token technology in details from more specific perspectives, respectively.

### 4.2 Information Theory Aspect: Shannon Entropy vs. Semantic Entropy

#### 4.2.1 Unified Perspective

From an information theory standpoint, both visual coding and MLLM tokenization can be seen as processes of entropy minimization. The core difference lies in the level of abstraction at which entropy is measured. Classical coding operates on the statistical properties of the signal (Shannon Entropy [9, 130]), while MLLM tokenization operates on the meaning or conceptual information conveyed by the signal (Semantic Entropy [90, 180]). In essence, tokenization is a semantic extension of classical coding, shifting the focus from compressing pixels to compressing concepts.

#### 4.2.2 Visual Coding: Minimizing Shannon Entropy

In classical visual coding, the primary goal is to represent raw pixel data with the fewest bits possible. This is fundamentally governed by Shannon entropy, $H(X)$, which quantifies the theoretical lower bound for lossless compression based on the statistical redundancy of the source signal $X$. The objective is to design encoders that approach this limit by removing statistical correlations, thereby minimizing the bit-rate required for transmission or storage.

#### 4.2.3 MLLM Tokens: Minimizing Semantic Entropy

MLLM tokenization is concerned with preserving *meaning*, not exact pixel values. This motivates the adoption of semantic entropy [90, 180], $H_{s}(\tilde{U})$, which measures the uncertainty over a set of semantic equivalence classes. By collapsing signals that are syntactically different but semantically identical (e.g., two different images of a ”cat on a mat”), semantic entropy is inherently lower than Shannon entropy ($H_{s}\leq H$). MLLM encoders act as *semantic filters*, discarding high-entropy, pixel-level details while retaining low-entropy, semantically decisive features. Thus, tokenization can be interpreted as compression guided not by Shannon entropy but by semantic entropy.

### 4.3 Functionality Aspect: Redundancy Reduction vs. Context Modeling

#### 4.3.1 Unified Perspective

Functionally, both approaches achieve compression by building a probabilistic model of the visual data to exploit its structure. Classical coding explicitly models and removes statistical redundancy through fixed transforms and entropy coders. MLLM tokenization implicitly learns to model the deep semantic context and dependencies within the data through learned, high-capacity architectures like the transformer, which performs a sophisticated form of context-aware redundancy removal.

#### 4.3.2 Visual Coding: Redundancy Reduction

Classical visual coding relies on explicit techniques for redundancy reduction. This typically involves a pipeline of decorrelation (e.g., Discrete Cosine Transform in JPEG [206]), which reduces spatial redundancy; quantization, which discards perceptually insignificant information; and entropy coding (e.g., Huffman or Arithmetic coding), which assigns shorter codes to more probable symbols. The functionality is to systematically strip away statistical redundancies present at the pixel level.

#### 4.3.3 MLLM Tokens: Context Modeling

MLLM tokenization achieves compression through powerful context modeling [180, 72]. Vision transformers learn to represent an image as a sequence of tokens and use self-attention mechanisms to model the complex interdependencies between them. This process is analogous to next-token prediction in language models, where the model learns the conditional probability $P(\text{token}_{t}|\text{token}_{<t})$. By capturing the high-level semantic context, the model can form a compact representation that implicitly discards irrelevant details, effectively performing semantic compression.

### 4.4 Optimization Aspect: R-D Trade-off vs. Information Bottleneck

#### 4.4.1 Unified Perspective

At their core, both domains solve an optimization problem that balances the compactness of the representation (rate) with its faithfulness to the original source (distortion). This can be universally formulated using the rate-distortion Lagrangian $\mathcal{L}=R+\lambda D$, where $\lambda$ controls the trade-off. The key distinction arises from how ”rate” and ”distortion” are defined. Moreover, these two problems are already tightly connected in information theory: the information bottleneck problem has a solution [59] that exactly coincides with the single-letter rate–distortion formula for the remote source coding problem [42, 216] under a logarithmic distortion function [34].

#### 4.4.2 Visual Coding: Rate-Distortion (R-D) Trade-off

The classic Rate-Distortion (R-D) trade-off in visual coding is defined as:

$$ $R(D)=\min I(X;Y)\quad\text{s.t.}\quad\mathbb{E}[d(X,Y)]\leq D,$ (9) $$

where the rate ($R$) is measured in bits, and the distortion ($D$) is measured by perceptual metrics like Mean Squared Error (MSE) or SSIM. The goal is to find an encoding that uses the minimum number of bits for a given level of visual fidelity.

#### 4.4.3 MLLM Tokens: Information Bottleneck

MLLM tokenization can be framed as an Information Bottleneck problem, which is a form of semantic rate-distortion optimization:

$$ $R_{s}(D_{s})=\min I(\tilde{X};\tilde{Y})\quad\text{s.t.}\quad\mathbb{E}[d_{s}(\tilde{X},\tilde{Y})]\leq D_{s},$ (10) $$

Here, the ”rate” ($R_{s}$) is operationalized by the number of tokens ($N$) or the computational complexity they induce (e.g., $\mathcal{O}(N^{2})$), as this directly relates to the semantic code length and computational cost. The ”distortion” ($D_{s}$) is semantic, measured by task performance metrics like classification accuracy or caption quality. The optimization seeks the most compact set of tokens that preserves the necessary semantic information for downstream tasks.

### 4.5 Objective Aspect: Human Eye Fidelity vs. Machine Task Analysis

#### 4.5.1 Unified Perspective

The ultimate objective of any compression scheme is to preserve the fidelity of the information for its intended ”user.” Both visual coding and MLLM tokenization are optimized for a specific user, but the nature of this user differs fundamentally. This leads to distinct definitions of what constitutes acceptable information loss.

#### 4.5.2 Visual Coding: Human Eye Fidelity

Classical visual coding is designed for human consumption. Therefore, its primary objective is to maintain high human eye fidelity. The distortion metrics (e.g., PSNR, SSIM, VMAF) are engineered to correlate with the human visual system’s perception of quality. The goal is to create a compressed representation that is perceptually indistinguishable, or nearly so, from the original to a human observer.

#### 4.5.3 MLLM Tokens: Machine Task Analysis

In contrast, MLLM tokens are generated for machine consumption. The objective is not perceptual quality but successful machine task analysis. The fidelity of the tokenized representation is measured by its utility in downstream tasks, such as image classification, object detection, or visual question answering. Therefore, the system is optimized to preserve task-relevant semantic features, even if this comes at the cost of pixel-level accuracy that would be noticeable to a human. This bridging framework sets the stage for subsequent discussions on multimodal tokens and their applications in communication and embodied AI.

**TABLE III: Comparison between classical visual coding and MLLM tokenization under the unified framework.**
| Aspect | Classical Visual Coding | MLLM Tokenization |
| --- | --- | --- |
| 1. Information Theory | Minimize Shannon Entropy (statistical uncertainty) | Minimize Semantic Entropy (conceptual uncertainty) |
| 2. Functionality | Explicit Redundancy Reduction (e.g., DCT, entropy coding) | Learned Context Modeling (e.g., self-attention in transformers) |
| 3. Optimization | Rate-Distortion (R-D) Trade-off (Rate in bits, Distortion in perceptual error) | Information Bottleneck (Rate in tokens/compute, Distortion in task error) |
| 4. Objective | Preserve Human Eye Fidelity (for human viewers) | Enable Machine Task Analysis (for machine algorithms) |

### 4.6 How Visual Coding Principles Can Refine Token Technology

The maturity of classical visual coding provides a rich set of optimization tools that can be directly adapted to mitigate inefficiencies in current visual tokenizers. By casting token generation as signal compression, we can inject structural priors and principled rate control into the tokenization pipeline.

Structural Decorrelation and Transformation
Current tokenizers often treat image patches as independent units or rely solely on self-attention to find correlations. Classical coding suggests that transforming signals into a decorrelated domain significantly enhances compressibility. Inspired by this, recent works have explored operating in the frequency domain via discrete transforms to compact energy before tokenization [50, 161]. Furthermore, borrowing the concept of Inter/Intra-frame coding from video standards, temporal redundancy can be explicitly modeled. For instance, logic similar to Group of Pictures (GOP) structures can be applied to token streams, separating information-rich “key-tokens” from predictable “motion-tokens,” thereby initializing a far more efficient representation for video inputs [53].

Entropy-Aware Token Management
Standard ViTs produce a fixed number of tokens regardless of content complexity, a stark contrast to the variable-bitrate nature of efficient codecs. Principles from entropy coding, such as Run-Length Encoding (RLE), serve as a blueprint for merging consecutive, redundant tokens in semantic space [32]. Moving beyond simple heuristics, the rigorous rate-control philosophy—optimizing the trade-off between bit consumption and distortion—can be adapted into “information-preserving guided selection.” This involves pruning or retaining tokens based on their marginal contribution to the total semantic information, effectively applying rate-distortion optimization (RDO) to the token budget [195].

Complexity-Adaptive Representation
Underpinning optimal compression is the principle of Minimum Description Length (MDL), a computable proxy for Kolmogorov Complexity. Applying this to MLLMs advocates for variable-length tokenization mechanisms [45]. Instead of a uniform grid, the tokenizer should dynamically allocate fewer symbols to simple, low-frequency regions and more symbols to complex, high-frequency details. This mirrors the quantization parameter (QP) adaptation in codecs, ensuring that the token count scales linearly with the semantic density of the input.

Discretization via Vector Quantization
While continuous embeddings dominate understanding tasks, the stability of storage and transmission benefits from the discrete nature of digital signals. Vector Quantization (VQ) acts as the bridge, mapping continuous latent spaces to discrete codebooks. This process not only aligns with the symbolic nature of language models but has been proven to stabilize generative tasks and reduce representation costs by enforcing a compact, learned vocabulary [57].

Figure: Figure 9: Adaptive quadtree partitioning driven by information density. Low-information regions remain coarse; high-information regions are split more finely. Red boxes denote retained dense tokens.
Refer to caption: x9.png

**TABLE IV: LLaVA-v1.5 [123] (7B) under three visual-token budgets. We compare QPID with FastV [26] and PruMerge [179] on six benchmarks at 25%/12.5%/6.25% retention (144/72/36 tokens). “Vanilla” uses all 576 tokens. The last column reports accuracy as a percentage of the full-token average; best in bold.**
| LLaVA-v1.5-7B Results |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Method | MME | SciQA | VQA^T | POPE | SeedB | VizW | Avg |
| Vanilla | 1512.53 | 53.76 | 57.62 | 85.5 | 65.79 | 53.21 | 100.0% |
| Ratio = 25% (144 tokens) |  |  |  |  |  |  |  |
| FastV | 1396.66 | 52.79 | 51.10 | 73.7 | 61.94 | 50.93 | 92.55% |
| PruMerge | 1416.52 | 53.53 | 55.03 | 81.6 | 62.39 | 52.04 | 96.13% |
| QPID | 1415.67 | 54.60 | 55.24 | 85.8 | 62.19 | 52.33 | 96.82% |
| Ratio = 12.5% (72 tokens) |  |  |  |  |  |  |  |
| FastV | 1301.68 | 52.96 | 48.65 | 62.7 | 56.40 | 50.97 | 85.51% |
| PruMerge | 1348.89 | 53.05 | 54.12 | 74.7 | 58.48 | 53.00 | 91.72% |
| QPID | 1353.52 | 53.59 | 54.34 | 84.3 | 59.76 | 53.42 | 94.17% |
| Ratio = 6.25% (36 tokens) |  |  |  |  |  |  |  |
| FastV | 1054.14 | 51.73 | 45.97 | 43.3 | 49.03 | 50.07 | 74.64% |
| PruMerge | 1261.41 | 52.86 | 52.98 | 67.5 | 54.09 | 52.61 | 86.64% |
| QPID | 1308.56 | 52.89 | 53.01 | 77.5 | 56.32 | 53.81 | 90.22% |

Figure: Figure 10: Ablation of visual‐token pruning on LLaVA-v1.5-7B [123]. We plot MME-Perception [51] (left) and ScienceQA [177] accuracy (right) vs. actual TFLOPs under three token‐retention rates (25%, 12.5%, 6.25%). QPID (ID+QP) consistently leads across all budgets.
Refer to caption: x10.png

Case Study I: Quadtree Partitioning–based Visual Token Pruning Considering Information Density. We replace attention score heuristics with an information-theoretic, structure-aware pipeline. First, an *entropy-based information density* criterion selects a compact, non–redundant subset of visual tokens, distributing them across the scene rather than clustering in a few high–attention regions; this task–agnostic scoring suppresses background redundancy and preserves diverse cues. Second, an adaptive *quadtree partitioning* allocator refines spatial granularity where information is high while keeping homogeneous areas as large leaves, so more tokens are assigned to semantically rich zones without losing global layout. The four–panel visualization in Fig. [9](https://arxiv.org/html/2601.20742v1#S4.F9) illustrates this behavior: low information regions remain coarse, high information regions are split more finely, and red boxes mark retained dense tokens. Quantitatively, Table [IV](https://arxiv.org/html/2601.20742v1#S4.T4) (LLaVA-v1.5-7B [123]) shows that QPID attains the best overall accuracy at all three budgets: at 25% tokens it maintains 96.82% of the full-token average and leads on four of six benchmarks; at 12.5% it reaches 94.17% and is best on all tasks; and at the extreme 6.25% (36 tokens) it still preserves 90.22%, widening the margin over prior methods as the budget tightens. The ablations in Fig. [10](https://arxiv.org/html/2601.20742v1#S4.F10) further isolate contributions under matched compute: on both MME-Perception [51] (left) and ScienceQA [177] accuracy (right), the full QPID (ID+QP) curve consistently lies above its variants across the 25%/12.5%/6.25% settings; removing information-density scoring (WO/ID) produces the largest drop, while omitting quadtree partitioning (WO/QP) also degrades results. Together, these findings indicate that entropy-driven selection and adaptive quadtree allocation are jointly responsible for stable accuracy at very small token budgets and deliver favorable accuracy–compute trade-offs for multimodal inference.

### 4.7 How Token Technology Can Refresh Codecs

Conversely, the rise of MLLMs and visual token Technology introduces a semantic dimension to the traditional signal processing field. The powerful reasoning and predictive capabilities of these models are transforming codecs from pixel-matching engines into semantic-aware intelligence systems.

Semantic-Guided Rate Allocation.
Traditional codecs struggle to distinguish between statistically complex noise and semantically important details. MLLMs can serve as a perceptual “brain” for the codec, analyzing the scene to generate semantic importance maps. These maps guide the encoder to allocate high bitrates to critical regions—such as text or human faces—while aggressively compressing irrelevant backgrounds, thus optimizing the bitstream for downstream machine vision utility rather than purely human perceptual metrics [131, 103].

Feature-Domain Compression.
As the consumers of visual data shift from humans to machines, the optimal compression target shifts from pixels to intermediate representations. A new paradigm of “Token Coding” is emerging, where the bitstream directly encapsulates semantic tokens rather than reconstructed pixels. This approach is particularly valuable for edge-cloud systems; by placing the tokenizer at the edge, one can transmit compact feature tokens [54, 165] or specialized machine-oriented bitstreams [81, 99], significantly reducing bandwidth requirements while preserving the performance of cloud-based MLLMs.

Universal Probabilistic Modeling.
At its core, compression is about prediction: the better one can predict the next symbol, the fewer bits are needed to encode it. MLLMs, trained on big data, have emerged as powerful general-purpose predictors. Their ability to model long-range dependencies and complex patterns allows them to function as universal compressors [117, 39]. By treating raw data bytes as tokens, MLLMs have demonstrated the potential to surpass specialized engineering codecs (like PNG) in compression ratios, hinting at a future where intelligence and compression are unified under a single probabilistic framework.

Figure: Figure 11: The framework of the Coding Paradigm Tailored to MLLMs (CoTAM) [126]. This paradigm utilizes the CLIP token-level prior to help improve the performance on compressed images.
Refer to caption: x11.png

Case Study II: A Coding Paradigm Tailored to MLLMs. Recent research [126] challenges the traditional decoupling of coding and machine perception by proposing CoTAM, a codec explicitly tailored for MLLMs. Instead of treating the downstream model as a black box, this approach analyzes the internal information flow of the vision encoder (e.g., CLIP), identifying a distinct ”three-stage” processing pattern: preliminary screening, local extraction, and global semantic integration. The study reveals a critical insight: compression distortion does not affect all tokens uniformly; it disproportionately disrupts the ”cross-level” features where low-level structural details are synthesized into high-level semantics.
As shown in Fig. [11](https://arxiv.org/html/2601.20742v1#S4.F11), guided by this token-level intelligence, CoTAM introduces a Shallow CLIP-Guided mechanism. It extracts attention maps from the shallow layers of the vision encoder to generate a semantic importance map, which directly controls the quantization step in the image codec to allocate more bits to semantically rich regions. Furthermore, it employs a multi-level fidelity decoder to align the reconstructed signal with the MLLM’s feature hierarchy. By achieving up to  36% bitrate savings on six benchmarks (MME [51], TextVQA [186], POPE [115], SeedBench [101], VQAv2 [60], MMMU [250], and MMBench [135], as shown in Fig. [12](https://arxiv.org/html/2601.20742v1#S4.F12)) for comparable MLLM performance, CoTAM exemplifies the potential of ”Compression Tells Intelligence”: leveraging the model’s own token attention mechanisms to optimize the fundamental rate-distortion trade-off in signal coding.

Figure: Figure 12: The results of CoTAM [126]. By utilizing token-level CLIP guidance, compared with recent codecs (ELIC [66], DCAE [140], Bridge [81], Adapt-ICMH [105]), it achieves better performance on MLLM tasks.
Refer to caption: x12.png

## 5 Application and Outlook

### 5.1 Next-generation Token Applications

#### 5.1.1 Token Technology in AIGC

Token technology is increasingly central in the AIGC era, where models must map high-dimensional continuous signals to compact representations for efficient generation. Across modalities, tokenization is converging toward a shared goal: *compact, semantically rich, and generative-friendly* representations. Text tokenization provides the discrete modeling blueprint[89], image tokenization extends it to perceptual semantics[48], and video tokenization largely inherits and adapts image techniques[244].

Image tokenization. Image generation like controllable synthesis and editing demands representations that preserve semantics while remaining efficient. Autoregressive systems rely on discrete tokens as the interface between images and LLM modeling[169]. Diffusion models originally introduced latent tokenization to reduce spatial resolution and accelerate training/inference[173]. As both paradigms scale, they increasingly converge: AR models are bottlenecked by long visual token sequences and thus seek compact-yet-detailed tokenizers[26], while diffusion models push toward more semantically aligned latent spaces to improve global coherence and controllability[248]. This convergence suggests tokenization is a key leverage point where coding principles and learned compression jointly shape modern AIGC.

Video tokenization. Current video generation pipelines often combine frame-wise tokenization with temporal attention and redundancy reduction[84, 245], but a principled temporal tokenizer remains less mature. A promising path is to re-introduce classical video-coding insights (motion prediction, hierarchical redundancy removal) into learned token pipelines[127].

In summary, tokenization across image, video is becoming increasingly unified around compact and semantically aligned representations[139], with image tokenization acting as the primary driver and video tokenization as a fast-growing frontier.

#### 5.1.2 Token Technology in Embodied AI

Embodied AI increasingly uses end-to-end foundation models that unify perception, language, and control. Here, tokenization acts as a machine-native compression interface, converting high-dimensional sensory streams (and optionally actions) into compact sequences for LLM-style backbones, improving data efficiency for long-horizon reasoning and real-time control.

Perception and context compression.
For manipulation, VLA systems often use continuous visual tokens from ViT features (RT-2[14], OpenVLA[82]), with OpenVLA combining semantically aligned tokens via SigLIP[251] and geometry-rich tokens via DINOv2[158]. Discrete tokenization via VQ-style codebooks enables scalable world modeling and next-token rollout (Genie[17]), while object-centric sparsity reduces redundancy by encoding salient entities (VIMA[77]). To handle long temporal contexts under transformer complexity, systems apply dynamic token pruning for real-time efficiency (LightVLA[76], FAST[161]) and compress historical interactions into highly compact memory tokens for retrieval (MemoryVLA[185]).

Action tokenization.
Recent models also compress continuous control into discrete action tokens, aligning robot outputs with the sequence modeling interface. RT-2[14] uses explicit quantization, while learned codebooks mitigate multi-modal “average action” effects (VQ-BeT[95] built on VQ-VAE[204]), enabling action generation as discrete token prediction.

#### 5.1.3 Categorization and Transferability of Visual Tokenizers

Building upon the taxonomy in Section [3.5](https://arxiv.org/html/2601.20742v1#S3.SS5), we revisit tokenizer families from the lens of transferability. Understanding-oriented tokenizers (CLIP-ViT[168], SigLIP[251], Perception Encoder[11]) emphasize semantic abstraction and cross-modal alignment, making them reusable across architectures with lightweight adaptation. Generation-oriented tokenizers (VQ-VAE[204], VQGAN[48]) prioritize high-fidelity reconstruction, but heterogeneous codebooks and objectives can hinder portability. Unified tokenizers (e.g., Show-o2[224]) attempt to encode semantics and details in one space, yet principled mechanisms to balance objectives and ensure cross-model portability remain underexplored. Overall, understanding-oriented tokenizers are typically more transferable, while generation-oriented tokenizers offer stronger perceptual fidelity but face practical transfer challenges.

### 5.2 Next-generation Codec Applications

#### 5.2.1 Immersive Media

NeRF-based codecs. Neural radiance fields (NeRF) have rapidly evolved from pure view-synthesis models into neural codecs for static and dynamic 3D content. Instead of transmitting per-frame pixels, NeRF-based methods encode a compact radiance field whose parameters are optimized for novel-view rendering, and then quantize and entropy-code these parameters as the bitstream. For static scenes, NeRFCodec [111] is a representative end-to-end design: it treats NeRF feature planes as latent images, reuses a pretrained 2D neural image codec, and learns lightweight scene-specific encoder/decoder heads under a joint rendering and rate–distortion objective, thereby achieving high-quality novel-view synthesis from bitstreams on the order of a few hundred kilobytes. For dynamic content, several works explicitly cast NeRF as a volumetric video codec. VRVVC [69] further introduces a tri-plane residual representation together with learnable quantization and compact entropy models, enabling variable-rate volumetric video compression with a single network and competitive rate–distortion performance across a wide bitrate range. Streaming radiance fields [110] demonstrate that explicit-grid radiance fields can also be updated over time and transmitted via model-difference coding, paving a path toward online NeRF-style streaming. Conceptually, these systems can be regarded as NeRF-based token codecs: structured radiance-field tokens (grid cells, tri-plane coefficients, residual fields, latent planes) become the basic symbols, and codec design focuses on their parameterization, on bit allocation between geometry and appearance, and on integration with conventional streaming infrastructures.

#### 5.2.2 MLLMs for Codec

With multimodal LLMs as receivers, codec objectives increasingly shift from classical rate–distortion (RD)[65] toward *rate–task performance* (RT): under a bit budget, maximize downstream utility while keeping latency and memory bounded. Two directions are emerging. (i) *MLLM-aware codecs* optimize representations for machine receivers, including end-to-end task losses[92], unified human/machine coding with multimodal supervision[243], and compression tailored to VLM decoders with explicit RT trade-offs[99]. (ii) *(M)LLMs as priors/decoders* leverage generative sequence models for compression: theory connects language modeling and compression[40], and recent systems demonstrate LLM-assisted lossless image coding via visual prompting[44] or language-space prediction[25]. These trends motivate evaluating codecs by RT curves (bitrate vs. task performance) and system metrics (decoding latency, KV-cache footprint), not RD alone.

#### 5.2.3 Video Coding for Machine (VCM)

VCM targets scenarios where the consumer is a machine (detector/tracker/MLLM), and the bitstream may carry pixels, intermediate features, or semantic descriptors. MPEG exploratory work formalizes tracks, common test conditions, and evaluation protocols that distinguish signal-domain and feature-domain pipelines[152, 153, 93]. Representative directions include task-aware signal coding[257], intermediate feature compression with notable RD/complexity advantages[83, 94], and semantic-level coding/collaborative analytics summarized in surveys[234]. Emerging theory further studies RT limits and rate–accuracy bounds for analysis tasks[8], while standardization efforts continue to converge on test protocols and metrics[263].

### 5.3 Unified Communication System in The LLM Era

Modern intelligent receivers motivate a representation-centric view of communication, driven by *what* is communicated and *why*. We distinguish systems by (i) communication unit (bits/semantics/tokens), (ii) objective (signal fidelity/task utility/model-conditioned utility under compute/memory budgets), and (iii) receiver interface (reconstruction/semantic inference/foundation-model conditioning).

#### 5.3.1 Traditional Communication

Classical communication optimizes bit recoverability: source coding removes redundancy toward entropy, channel coding protects bits under noise, and separation motivates independent design[181, 155]. Distortion is defined in signal space (e.g., MSE/PSNR)[49], and deep Joint Source-Channel Coding (JSCC) variants largely remain reconstruction-oriented[12, 91, 230]. Finite-blocklength theory clarifies the gap to Shannon limits under short packets[162, 147, 87] and extends to one-shot and multiuser settings[238, 102, 178], explaining why reconstruction-driven RD pipelines can mismatch modern machine receivers.

#### 5.3.2 Semantic Communication

Semantic communication shifts to task utility by transmitting only what is needed for a task. Task-oriented JSCC directly optimizes downstream losses over noisy channels[221, 12, 222]. Surveys and tutorials formalize the rate–task viewpoint and evaluation beyond RD[233, 141, 62]. Foundation models (multimodal LLMs, diffusion priors) can serve as semantic front-ends for summarization and regeneration[75, 213], though generalization across tasks/channels and short-blocklength overheads remain challenges; robust task-oriented training is an active direction[159].

#### 5.3.3 Token Communication

Machine-native coordination via learned tokens.
Token communication advances semantic communication by adopting model-consumable tokens as the transmitted interface[80, 197]. A sensor agent maps observations into compact continuous embeddings or discrete codebook indices for direct consumption, aligning with split inference and feature transmission to reduce latency and on-device compute[47]. Learned machine-language tokens can be substantially more transmission-efficient than natural language while remaining task-sufficient[220].

Robustness can be addressed via Joint Token & Channel Coding (JTCC) that injects channel impairments during training[220], or via analog mappings that transmit token vectors directly (“over-the-air tokens”) leveraging deep JSCC and over-the-air computation[91, 230, 58, 176, 156]. Recovered tokens can condition an LLM through soft-prefix prompting without fine-tuning[112, 98]. Recent token-domain multi-access designs further explore contextual prediction to mitigate collisions and improve bandwidth efficiency[163, 164].

## 6 Conclusion

Guided by the principle that “Compression Tells Intelligence,” this paper unifies classical visual coding and emerging visual token technology under a shared view of efficiency–fidelity trade-offs. We connect the two through a common framework spanning information measures (Shannon vs. semantic), functional roles (redundancy reduction vs. context modeling), optimization criteria (R–D vs. information bottleneck), and objectives (human fidelity vs. machine utility), etc.
This unification yields bidirectional insights: coding principles (e.g., decorrelation and entropy-aware rate control) can improve token systems, while token-based semantic modeling motivates next-generation codecs optimized for machine tasks. We also discuss the potential impacts of the token techniques on MLLMs, AIGC, and even embodied AI, and outline the next generation of visual coding technology.
Future work includes unified tokenizers balancing semantic alignment and reconstructive fidelity, token communication across platforms, and extending the framework to emerging modalities such as 3D and 4D information.

## Acknowledgments

This work was supported in part by NSFC 62302246 and ZJNSFC under Grant LQ23F010008, and supported by High Performance Computing Center at Eastern Institute of Technology, Ningbo, and Ningbo Institute of Digital Twin.

## References

- [1]
E. Agustsson, M. Tschannen, F. Mentzer, R. Timofte, and L. V. Gool (2019)
Generative adversarial networks for extreme learned image compression.
In Proceedings of the IEEE/CVF international conference on computer vision,
pp. 221–231.
Cited by: [§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [2]
J. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. (2022)
Flamingo: a visual language model for few-shot learning.
nips.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [3]
S. R. Alvar, G. Singh, M. Akbari, and Y. Zhang (2025)
Divprune: diversity-based visual token pruning for large multimodal models.
In cvpr,
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.13.1.1.1).
- [4]
S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh (2015)
Vqa: visual question answering.
In Proceedings of the IEEE international conference on computer vision,
pp. 2425–2433.
Cited by: [§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1).
- [5]
K. H. I. Arif, J. Yoon, D. S. Nikolopoulos, H. Vandierendonck, D. John, and B. Ji (2025)
HiRED: attention-guided token dropping for efficient inference of high-resolution vision-language models.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39, pp. 1773–1781.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [6]
J. Ascenso and E. Upenik (2021)
White paper on jpeg ai scope and framework.
ISO/IEC JTC 1.
Cited by: [§2.2.1](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS1.p1.1),
[§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [7]
S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, et al. (2025)
Qwen2. 5-vl technical report.
preprint arXiv:2502.13923.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p1.1),
[§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.7.1),
[§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1).
- [8]
I. V. Bajić (2025)
Rate-accuracy bounds in visual coding for machines.
arXiv preprint arXiv:2505.14980.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [9]
J. Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston (2018)
Variational image compression with a scale hyperprior.
preprint arXiv:1802.01436.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1),
[§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1),
[§4.2.1](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS1.p1.1).
- [10]
D. Bolya, C. Fu, X. Dai, P. Zhang, C. Feichtenhofer, and J. Hoffman (2023)
Token merging: your vit but faster.
In ICLR,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p2.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [11]
D. Bolya, P. Huang, P. Sun, J. H. Cho, A. Madotto, C. Wei, T. Ma, J. Zhi, J. Rajasegaran, H. Rasheed, et al. (2025)
Perception encoder: the best visual embeddings are not at the output of the network.
preprint arXiv:2504.13181.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [12]
E. Bourtsoulatze, D. B. Kurka, and D. Gündüz (2019)
Deep joint source-channel coding for wireless image transmission.
IEEE Transactions on Cognitive Communications and Networking 5 (3), pp. 567–579.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1),
[§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [13]
T. Boutell (1997)
Png (portable network graphics) specification version 1.0.
Technical report
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2).
- [14]
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, P. Florence, C. Fu, M. G. Arenas, K. Gopalakrishnan, K. Han, K. Hausman, A. Herzog, J. Hsu, B. Ichter, A. Irpan, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, L. Lee, T. E. Lee, S. Levine, Y. Lu, H. Michalewski, I. Mordatch, K. Pertsch, K. Rao, K. Reymann, M. Ryoo, G. Salazar, P. Sanketi, P. Sermanet, J. Singh, A. Singh, R. Soricut, H. Tran, V. Vanhoucke, Q. Vuong, A. Wahid, S. Welker, P. Wohlhart, J. Wu, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich (2023)
RT-2: vision-language-action models transfer web knowledge to robotic control.
External Links: 2307.15818,
[Link](https://arxiv.org/abs/2307.15818)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1),
[§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p3.1).
- [15]
B. Bross, Y. Wang, Y. Ye, S. Liu, J. Chen, G. J. Sullivan, and J. Ohm (2021)
Overview of the versatile video coding (vvc) standard and its applications.
IEEE Transactions on Circuits and Systems for Video Technology 31 (10), pp. 3736–3764.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.2.1](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS1.p1.1),
[§2.4.1](https://arxiv.org/html/2601.20742v1#S2.SS4.SSS1.p1.1).
- [16]
B. Bross (2013)
High efficiency video coding (hevc) text specification draft 10 (for fdis & last call).
In Joint Collaborative Team on Video Coding (JCT-VC) of ITU-T SG 16 WP 3 and ISO/IEC JTC 1/SC 29/WG 11, 12th Meeting, Geneva,(Jan. 2013),
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.2.1](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS1.p1.1),
[§2.4.1](https://arxiv.org/html/2601.20742v1#S2.SS4.SSS1.p1.1).
- [17]
J. Bruce, M. Dennis, A. Edwards, J. Parker-Holder, Y. Shi, E. Hughes, M. Lai, A. Mavalankar, R. Steigerwald, C. Apps, Y. Aytar, S. Bechtle, F. Behbahani, S. Chan, N. Heess, L. Gonzalez, S. Osindero, S. Ozair, S. Reed, J. Zhang, K. Zolna, J. Clune, N. de Freitas, S. Singh, and T. Rocktäschel (2024)
Genie: generative interactive environments.
External Links: 2402.15391,
[Link](https://arxiv.org/abs/2402.15391)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [18]
M. Cai, J. Yang, J. Gao, and Y. J. Lee (2024)
Matryoshka multimodal models.
In NeurIPS Workshop,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [19]
M. Careil, M. J. Muckley, J. Verbeek, and S. Lathuilière (2024)
Towards image compression with perfect realism at ultra-low bitrates.
In The Twelfth International Conference on Learning Representations,
External Links: [Link](https://openreview.net/forum?id=ktdETU9JBg)
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [20]
M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin (2021)
Emerging properties in self-supervised vision transformers.
In Proceedings of the IEEE/CVF international conference on computer vision,
pp. 9650–9660.
Cited by: [§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1).
- [21]
J. Cha, W. Kang, J. Mun, and B. Roh (2024)
Honeybee: locality-enhanced projector for multimodal llm.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 13817–13827.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [22]
W. Chai, E. Song, Y. Du, C. Meng, V. Madhavan, O. Bar-Tal, J. Hwang, S. Xie, and C. D. Manning (2025)
Auroracap: efficient, performant video detailed captioning and a new benchmark.
In iclr,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [23]
H. Chang, H. Z. Yu, V. Vasudevan, W. T. Freeman, D. K. Liu, B. Catanzaro, I. Essa, M. Halber, and M. Sandler (2022)
MaskGIT: masked generative image transformer.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.12.1).
- [24]
J. Chen, Z. Xu, X. Pan, Y. Hu, C. Qin, T. Goldstein, L. Huang, T. Zhou, S. Xie, S. Savarese, et al. (2025)
Blip3-o: a family of fully open unified multimodal models-architecture, training and dataset.
preprint arXiv:2505.09568.
Cited by: [§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p1.1).
- [25]
K. Chen, P. Zhang, H. Liu, J. Liu, Y. Liu, J. Huang, S. Wang, H. Yan, and H. Li (2024)
Large language models for lossless image compression: next-pixel prediction in language space is all you need.
arXiv preprint arXiv:2411.12448.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [26]
L. Chen, H. Zhao, T. Liu, S. Bai, J. Lin, C. Zhou, and B. Chang (2024)
An image is worth 1/2 tokens after layer 2: plug-and-play inference acceleration for large vision-language models.
In ECCV,
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.5.1.1.1),
[TABLE IV](https://arxiv.org/html/2601.20742v1#S4.T4),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p2.1).
- [27]
X. Chen, Z. Zhang, H. Zhang, Y. Zhou, S. Y. Kim, Q. Liu, Y. Li, J. Zhang, N. Zhao, Y. Wang, et al. (2025)
Unireal: universal image generation and editing via learning real-world dynamics.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 12501–12511.
Cited by: [§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1).
- [28]
Y. Chen, Y. Weng, C. Kao, C. Chien, W. Chiu, and W. Peng (2023)
Transtic: transferring transformer-based image compression from human perception to machine perception.
In Proceedings of the IEEE/CVF International Conference on Computer Vision,
pp. 23297–23307.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p1.1).
- [29]
Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, M. Zhong, Q. Zhang, X. Zhu, L. Lu, et al. (2024)
Internvl: scaling up vision foundation models and aligning for generic visual-linguistic tasks.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 24185–24198.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1).
- [30]
Z. Cheng, H. Sun, M. Takeuchi, and J. Katto (2020)
Learned image compression with discretized gaussian mixture likelihoods and attention modules.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 7939–7948.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1).
- [31]
J. Choi, S. Lee, J. Chu, M. Choi, and H. J. Kim (2024)
Vid-tldr: training-free token merging for light-weight video transformer.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [32]
R. Choudhury, G. Zhu, S. Liu, K. Niinuma, K. Kitani, and L. Jeni (2024)
Don’t look twice: faster video transformers with run-length tokenization.
Advances in Neural Information Processing Systems 37, pp. 28127–28149.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p3.1).
- [33]
X. Chu, L. Qiao, X. Lin, S. Xu, Y. Yang, Y. Hu, F. Wei, X. Zhang, B. Zhang, X. Wei, et al. (2023)
Mobilevlm: a fast, strong and open vision language assistant for mobile devices.
preprint arXiv:2312.16886.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [34]
T. A. Courtade and T. Weissman (2013)
Multiterminal source coding under logarithmic loss.
IEEE Transactions on Information Theory 60 (1), pp. 740–761.
Cited by: [§4.4.1](https://arxiv.org/html/2601.20742v1#S4.SS4.SSS1.p1.2).
- [35]
Z. Cui, J. Wang, S. Gao, T. Guo, Y. Feng, and B. Bai (2021)
Asymmetric gained deep image compression with continuous rate adaptation.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 10532–10541.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1).
- [36]
W. Dai, N. Lee, B. Wang, Z. Yang, Z. Liu, J. Barker, T. Rintamaki, M. Shoeybi, B. Catanzaro, and W. Ping (2024)
Nvlm: open frontier-class multimodal llms.
arXiv preprint arXiv:2409.11402.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [37]
W. Dai, J. Li, D. Li, A. Tiong, J. Zhao, W. Wang, B. Li, P. N. Fung, and S. Hoi (2023)
Instructblip: towards general-purpose vision-language models with instruction tuning.
Advances in neural information processing systems 36, pp. 49250–49267.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [38]
T. Dao (2024)
FlashAttention-2: faster attention with better parallelism and work partitioning.
In International Conference on Learning Representations (ICLR),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p2.1).
- [39]
G. Deletang, A. Ruoss, P. Duquenne, E. Catt, T. Genewein, C. Mattern, J. Grau-Moya, L. K. Wenliang, M. Aitchison, L. Orseau, et al.
Language modeling is compression.
In The Twelfth International Conference on Learning Representations,
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p4.1).
- [40]
G. Delétang, A. Ruoss, P. Duquenne, E. Catt, T. Genewein, C. Mattern, J. Grau-Moya, L. K. Wenliang, M. Aitchison, L. Orseau, et al. (2023)
Language modeling is compression.
arXiv preprint arXiv:2309.10668.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [41]
C. Deng, D. Zhu, K. Li, C. Gou, F. Li, Z. Wang, S. Zhong, W. Yu, X. Nie, Z. Song, et al. (2025)
Emerging properties in unified multimodal pretraining.
arXiv preprint arXiv:2505.14683.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p4.2),
[§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p2.1).
- [42]
R. Dobrushin and B. Tsybakov (1962)
Information transmission with additional noise.
IRE Transactions on Information Theory 8 (5), pp. 293–304.
Cited by: [§4.4.1](https://arxiv.org/html/2601.20742v1#S4.SS4.SSS1.p1.2).
- [43]
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby (2020-10)
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
arXiv e-prints, pp. arXiv:2010.11929.
External Links: [Document](https://dx.doi.org/10.48550/arXiv.2010.11929),
2010.11929
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p2.4),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1).
- [44]
J. Du, C. Zhou, N. Cao, G. Chen, Y. Chen, Z. Cheng, L. Song, G. Lu, and W. Zhang (2025)
Large language model for lossless image compression with visual prompts.
arXiv preprint arXiv:2502.16163.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [45]
S. Duggal, S. Byun, W. T. Freeman, A. Torralba, and P. Isola (2025)
Single-pass adaptive image tokenization for minimum program search.
preprint arXiv:2507.07995.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p4.1).
- [46]
S. Endo, T. Kuwabara, K. Yamaguchi, T. Takikawa, and S. Saito (2025)
FEATHER the throttle: revisiting token pruning inside language decoders.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [47]
A. E. Eshratifar, A. Esmaili, and M. Pedram (2019)
Bottlenet: a deep learning architecture for intelligent mobile cloud computing services.
In 2019 IEEE/ACM International Symposium on Low Power Electronics and Design (ISLPED),
pp. 1–6.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p1.1).
- [48]
P. Esser, R. Rombach, and B. Ommer (2021)
Taming transformers for high-resolution image synthesis.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 12873–12883.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.2.2](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS2.p1.1),
[§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.11.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p1.1),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [49]
F. A. Fardo, V. H. Conforto, F. C. De Oliveira, and P. S. Rodrigues (2016)
A formal evaluation of psnr as quality measurement parameter for image segmentation algorithms.
arXiv preprint arXiv:1605.07116.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [50]
H. Feng, Q. Liu, H. Liu, J. Tang, W. Zhou, H. Li, and C. Huang (2024)
Docpedia: unleashing the power of large multimodal model in the frequency domain for versatile document understanding.
Science China Information Sciences 67 (12), pp. 220106.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p2.1).
- [51]
C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng, K. Li, X. Sun, and R. Ji (2023)
MME: a comprehensive evaluation benchmark for multimodal large language models.
ArXiv abs/2306.13394.
External Links: [Link](https://api.semanticscholar.org/CorpusID:259243928)
Cited by: [Figure 10](https://arxiv.org/html/2601.20742v1#S4.F10),
[§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p6.1),
[§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [52]
C. Fu, Y. Dai, Y. Luo, L. Li, S. Ren, R. Zhang, Z. Wang, C. Zhou, Y. Shen, M. Zhang, P. Chen, Y. Li, S. Lin, S. Zhao, K. Li, T. Xu, X. Zheng, E. Chen, C. Shan, R. He, and X. Sun (2025)
Video-mme: the first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis.
External Links: 2405.21075,
[Link](https://arxiv.org/abs/2405.21075)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p4.1).
- [53]
U. Gadot, A. Shocher, S. Mannor, G. Chechik, and A. Hallak (2025)
RL-rc-dot: a block-level rl agent for task-aware video compression.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 12533–12542.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p2.1).
- [54]
C. Gao, Y. Ma, Q. Chen, Y. Xu, D. Liu, and W. Lin (2024)
Feature coding in the era of large models: dataset, test conditions, and benchmark.
preprint arXiv:2412.04307.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p3.1).
- [55]
L. Gao, Y. Zhong, Y. Zeng, H. Tan, D. Li, and Z. Zhao (2024)
Linvt: empower your image-level large language model to understand videos.
arXiv preprint arXiv:2412.05185.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [56]
S. Ge, T. Hayes, H. Yang, X. Yin, G. Pang, D. Jacobs, J. Huang, and D. Parikh (2022)
Long video generation with time-agnostic vqgan and time-sensitive transformer.
In European Conference on Computer Vision,
pp. 102–118.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [57]
Z. Geng, Y. Wang, Y. Ma, C. Li, Y. Rao, S. Gu, Z. Zhong, Q. Lu, H. Hu, X. Zhang, et al. (2025)
X-omni: reinforcement learning makes discrete autoregressive image generative models great again.
arXiv preprint arXiv:2507.22058.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p5.1).
- [58]
M. Goldenbaum, H. Boche, and S. Stańczak (2013)
Harnessing interference for analog function computation in wireless sensor networks.
IEEE Transactions on Signal Processing 61 (20), pp. 4893–4906.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [59]
Z. Goldfeld and Y. Polyanskiy (2020)
The information bottleneck problem and its applications in machine learning.
IEEE Journal on Selected Areas in Information Theory 1 (1), pp. 19–38.
Cited by: [§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1),
[§4.4.1](https://arxiv.org/html/2601.20742v1#S4.SS4.SSS1.p1.2).
- [60]
Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh (2017)
Making the v in vqa matter: elevating the role of image understanding in visual question answering.
In Proceedings of the IEEE conference on computer vision and pattern recognition,
pp. 6904–6913.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [61]
P. N. Graphics (2003)
Specification information technology-computer graphics and image processing-portable network graphics (png): functional specification.
ISO/IEC 15948.
Cited by: [§2.3.1](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS1.p1.1).
- [62]
D. Gündüz, M. A. Wigger, T. Tung, P. Zhang, and Y. Xiao (2024)
Joint source–channel coding: fundamentals and recent progress in practical designs.
Proceedings of the IEEE.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [63]
Z. Guo, R. Xu, Y. Yao, J. Cui, Z. Ni, C. Ge, T. Chua, Z. Liu, and G. Huang (2024)
Llava-uhd: an lmm perceiving any aspect ratio and high-resolution images.
In European Conference on Computer Vision,
pp. 390–406.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p1.13),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.14.1.1.1).
- [64]
S. Gurukar and A. Kadav (2025)
Long-vmnet: accelerating long-form video understanding via fixed memory.
arXiv preprint arXiv:2503.13707.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [65]
A. Harell, Y. Foroutan, N. Ahuja, P. Datta, B. Kanzariya, V. S. Somayazulu, O. Tickoo, A. de Andrade, and I. V. Bajić (2025)
Rate-distortion theory in coding for machines and its applications.
IEEE Transactions on Pattern Analysis and Machine Intelligence.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [66]
D. He, Z. Yang, W. Peng, R. Ma, H. Qin, and Y. Wang (2022)
Elic: efficient learned image compression with unevenly grouped space-channel contextual adaptive coding.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 5718–5727.
Cited by: [§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1),
[Figure 12](https://arxiv.org/html/2601.20742v1#S4.F12).
- [67]
Y. He, F. Chen, J. Liu, W. Shao, H. Zhou, K. Zhang, and B. Zhuang (2024)
Zipvl: efficient large vision-language models with dynamic token sparsification and kv cache compression.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [68]
J. Ho, A. Jain, and P. Abbeel (2020)
Denoising diffusion probabilistic models.
Advances in neural information processing systems 33, pp. 6840–6851.
Cited by: [Figure 5](https://arxiv.org/html/2601.20742v1#S3.F5).
- [69]
Q. Hu, H. Zhong, Z. Zheng, X. Zhang, Z. Cheng, L. Song, G. Zhai, and Y. Wang (2025)
VRVVC: variable-rate nerf-based volumetric video compression.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39, pp. 3563–3571.
Cited by: [§5.2.1](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS1.p1.1).
- [70]
K. Huang, H. Zou, Y. Xi, B. Wang, Z. Xie, and L. Yu (2024)
Ivtp: instruction-guided visual token pruning for large vision-language models.
In European Conference on Computer Vision,
pp. 214–230.
Cited by: [TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.6.1.1.1).
- [71]
X. Huang, H. Zhou, and K. Han (2025)
Prunevid: visual token pruning for efficient video large language models.
In Findings of the Association for Computational Linguistics: ACL 2025,
pp. 19959–19973.
Cited by: [TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.8.1.1.1).
- [72]
Y. Huang, J. Zhang, Z. Shan, and J. He (2024)
Compression represents intelligence linearly.
arXiv preprint arXiv:2404.09937.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p1.1),
[§4.3.3](https://arxiv.org/html/2601.20742v1#S4.SS3.SSS3.p1.1).
- [73]
A. Jaegle, F. Gimeno, A. Brock, et al. (2021)
Perceiver io: a general architecture for structured inputs & outputs.
arXiv preprint arXiv:2107.14795.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [74]
Z. Jia, B. Li, J. Li, W. Xie, L. Qi, H. Li, and Y. Lu (2025)
Towards practical real-time neural video compression.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 12543–12552.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.4.2](https://arxiv.org/html/2601.20742v1#S2.SS4.SSS2.p1.1).
- [75]
F. Jiang, L. Dong, Y. Peng, K. Wang, K. Yang, C. Pan, and X. You (2024)
Large ai model empowered multimodal semantic communications.
IEEE Communications Magazine.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [76]
T. Jiang, X. Jiang, Y. Ma, X. Wen, B. Li, K. Zhan, P. Jia, Y. Liu, S. Sun, and X. Lang (2025)
The better you learn, the smarter you prune: towards efficient vision-language-action models via differentiable token pruning.
External Links: 2509.12594,
[Link](https://arxiv.org/abs/2509.12594)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [77]
Y. Jiang, A. Gupta, Z. Zhang, G. Wang, Y. Dou, Y. Chen, L. Fei-Fei, A. Anandkumar, Y. Zhu, and L. Fan (2023)
VIMA: general robot manipulation with multimodal prompts.
External Links: 2210.03094,
[Link](https://arxiv.org/abs/2210.03094)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [78]
P. Jin, R. Takanobu, W. Zhang, X. Cao, and L. Yuan (2024)
Chat-univi: unified visual representation empowers large language models with image and video understanding.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 13700–13710.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [79]
X. Jin, R. Feng, S. Sun, R. Feng, T. He, and Z. Chen (2023)
Semantical video coding: instill static-dynamic clues into structured bitstream for ai tasks.
Journal of Visual Communication and Image Representation 93, pp. 103816.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [80]
Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars, and L. Tang (2017)
Neurosurgeon: collaborative intelligence between the cloud and mobile edge.
ACM SIGARCH Computer Architecture News 45 (1), pp. 615–629.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p1.1).
- [81]
C. Kao, C. Chien, Y. Tseng, Y. Chen, A. Gnutti, S. Lo, W. Peng, and R. Leonardi
Bridging compressed image latents and multimodal large language models.
In The Thirteenth International Conference on Learning Representations,
Cited by: [Figure 12](https://arxiv.org/html/2601.20742v1#S4.F12),
[§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p3.1).
- [82]
M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn (2024)
OpenVLA: an open-source vision-language-action model.
External Links: 2406.09246,
[Link](https://arxiv.org/abs/2406.09246)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [83]
Y. Kim, H. Jeong, J. Yu, Y. Kim, J. Lee, S. Y. Jeong, and H. Y. Kim (2023)
End-to-end learnable multi-scale feature compression for vcm.
IEEE Transactions on Circuits and Systems for Video Technology 34 (5), pp. 3156–3167.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [84]
D. Kondratyuk, L. Yu, X. Gu, J. Lezama, J. Huang, G. Schindler, R. Hornung, V. Birodkar, J. Yan, M. Chiu, et al. (2023)
Videopoet: a large language model for zero-shot video generation.
arXiv preprint arXiv:2312.14125.
Cited by: [§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p3.1).
- [85]
W. Kong, Q. Tian, Z. Zhang, R. Min, Z. Dai, J. Zhou, J. Xiong, X. Li, B. Wu, J. Zhang, K. Wu, Q. Lin, J. Yuan, Y. Long, A. Wang, A. Wang, C. Li, D. Huang, F. Yang, H. Tan, H. Wang, J. Song, J. Bai, J. Wu, J. Xue, J. Wang, K. Wang, M. Liu, P. Li, S. Li, W. Wang, W. Yu, X. Deng, Y. Li, Y. Chen, Y. Cui, Y. Peng, Z. Yu, Z. He, Z. Xu, Z. Zhou, Z. Xu, Y. Tao, Q. Lu, S. Liu, D. Zhou, H. Wang, Y. Yang, D. Wang, Y. Liu, J. Jiang, and C. Zhong (2025)
HunyuanVideo: a systematic framework for large video generative models.
External Links: 2412.03603,
[Link](https://arxiv.org/abs/2412.03603)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [86]
B. Korbar, Z. Al-Halah, and K. Grauman (2024)
Text-conditioned resampler for long-form video understanding.
In European Conference on Computer Vision (ECCV),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [87]
V. Kostina and S. Verdú (2012)
Fixed-length lossy compression in the finite blocklength regime.
IEEE Transactions on Information Theory 58 (6), pp. 3309–3338.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [88]
A. B. Koyuncu, H. Gao, A. Boev, G. Gaikov, E. Alshina, and E. Steinbach (2022)
Contextformer: a transformer with spatio-channel attention for context modeling in learned image compression.
In European conference on computer vision,
pp. 447–463.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1).
- [89]
T. Kudo and J. Richardson (2018)
SentencePiece: a simple and language independent subword tokenizer and detokenizer for neural text processing.
arXiv preprint arXiv:1808.06226.
Cited by: [§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p1.1).
- [90]
L. Kuhn, Y. Gal, and S. Farquhar (2023)
Semantic uncertainty: linguistic invariances for uncertainty estimation in natural language generation.
arXiv preprint arXiv:2302.09664.
Cited by: [§4.2.1](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS1.p1.1),
[§4.2.3](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS3.p1.2.2).
- [91]
D. B. Kurka and D. Gündüz (2021)
Bandwidth-agile image transmission with deep joint source-channel coding.
IEEE Transactions on Wireless Communications 20 (12), pp. 8081–8095.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1),
[§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [92]
N. Le, H. Zhang, F. Cricri, R. Ghaznavi-Youvalari, and E. Rahtu (2021)
Image coding for machines: an end-to-end learned approach.
In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
pp. 1590–1594.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [93]
D. Lee, S. Jeon, Y. Jeong, J. Kim, and J. Seo (2023)
Exploring the video coding for machines standard: current status and future directions.
JOURNAL OF BROADCAST ENGINEERING 28 (7), pp. 888–903.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [94]
M. Lee, S. Park, S. Oh, Y. Kim, S. Y. Jeong, J. Lee, and D. Sim (2023)
Transform-based feature map compression method for video coding for machines (vcm).
Electronics 12 (19), pp. 4042.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [95]
S. Lee, Y. Wang, H. Etukuru, H. J. Kim, N. M. M. Shafiullah, and L. Pinto (2024)
Behavior generation with latent actions.
External Links: 2403.03181,
[Link](https://arxiv.org/abs/2403.03181)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p3.1).
- [96]
E. Lei, H. Hassani, and S. S. Bidokhti (2023)
Neural estimation of the rate-distortion function with applications to operational source coding.
IEEE Journal on Selected Areas in Information Theory 3 (4), pp. 674–686.
Cited by: [§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1).
- [97]
E. Lei, H. Hassani, and S. S. Bidokhti (2024)
Approaching rate-distortion limits in neural compression with lattice transform coding.
arXiv preprint arXiv:2403.07320.
Cited by: [§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1).
- [98]
B. Lester, R. Al-Rfou, and N. Constant (2021)
The power of scale for parameter-efficient prompt tuning.
arXiv preprint arXiv:2104.08691.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [99]
B. Li, S. Wang, S. Wang, and Y. Ye (2024)
High efficiency image compression for large visual-language models.
IEEE Transactions on Circuits and Systems for Video Technology.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p3.1),
[§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [100]
B. Li, Y. Zhang, D. Guo, R. Zhang, F. Li, H. Zhang, K. Zhang, P. Zhang, Y. Li, Z. Liu, and C. Li (2025)
Llava-onevision: easy visual task transfer.
tmlr.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [101]
B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan (2023)
Seed-bench: benchmarking multimodal llms with generative comprehension.
arXiv preprint arXiv:2307.16125.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [102]
C. T. Li and V. Anantharam (2021)
A unified framework for one-shot achievability via the poisson matching lemma.
IEEE Transactions on Information Theory 67 (5), pp. 2624–2651.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [103]
C. Li, G. Lu, D. Feng, H. Wu, Z. Zhang, X. Liu, G. Zhai, W. Lin, and W. Zhang (2024)
Misc: ultra-low bitrate image semantic compression driven by large multimodal model.
IEEE Transactions on Image Processing.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1),
[§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p2.1).
- [104]
H. Li, S. Li, W. Dai, C. Li, J. Zou, and H. Xiong (2023)
Frequency-aware transformer for learned image compression.
arXiv preprint arXiv:2310.16387.
Cited by: [§2.2.2](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS2.p1.1),
[§2](https://arxiv.org/html/2601.20742v1#S2.p1.1).
- [105]
H. Li, S. Li, S. Ding, W. Dai, M. Cao, C. Li, J. Zou, and H. Xiong (2024)
Image compression for machine and human vision with spatial-frequency adaptation.
In European Conference on Computer Vision,
pp. 382–399.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[Figure 12](https://arxiv.org/html/2601.20742v1#S4.F12).
- [106]
H. Li and X. Zhang (2024)
Human–machine collaborative image compression method based on implicit neural representations.
IEEE Journal on Emerging and Selected Topics in Circuits and Systems 14 (2), pp. 198–208.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [107]
J. Li, B. Li, and Y. Lu (2021)
Deep contextual video compression.
Advances in Neural Information Processing Systems 34, pp. 18114–18125.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.4.2](https://arxiv.org/html/2601.20742v1#S2.SS4.SSS2.p1.1).
- [108]
J. Li, D. Li, C. Xiong, and S. C.H. Hoi (2023)
BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models.
In Proceedings of the 40th International Conference on Machine Learning (ICML),
Proceedings of Machine Learning Research, Vol. 202, pp. 19730–19742.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.5.1).
- [109]
K. Li, Y. Wang, Y. He, Y. Li, Y. Wang, Y. Liu, Z. Wang, J. Xu, G. Chen, P. Luo, L. Wang, and Y. Qiao (2024)
MVBench: a comprehensive multi-modal video understanding benchmark.
External Links: 2311.17005,
[Link](https://arxiv.org/abs/2311.17005)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p4.1).
- [110]
L. Li, Z. Shen, Z. Wang, L. Shen, and P. Tan (2022)
Streaming radiance fields for 3d video synthesis.
Advances in Neural Information Processing Systems 35, pp. 13485–13498.
Cited by: [§5.2.1](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS1.p1.1).
- [111]
S. Li, H. Li, Y. Liao, and L. Yu (2024)
Nerfcodec: neural feature compression meets neural radiance fields for memory-efficient scene representation.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 21274–21283.
Cited by: [§5.2.1](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS1.p1.1).
- [112]
X. L. Li and P. Liang (2021)
Prefix-tuning: optimizing continuous prompts for generation.
arXiv preprint arXiv:2101.00190.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [113]
X. Li, J. Li, L. Chen, et al. (2024)
TokenPacker: pack more visual tokens into llms.
arXiv preprint arXiv:2407.02392.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1).
- [114]
Y. Li, C. Wang, and J. Jia (2024)
Llama-vid: an image is worth 2 tokens in large language models.
In European Conference on Computer Vision,
pp. 323–340.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p1.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p2.1).
- [115]
Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J. Wen (2023)
Evaluating object hallucination in large vision-language models.
arXiv preprint arXiv:2305.10355.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [116]
Z. Li, Y. Zhou, H. Wei, C. Ge, and J. Jiang (2025)
Toward extreme image compression with latent feature guidance and diffusion prior.
IEEE Transactions on Circuits and Systems for Video Technology 35 (1), pp. 888–899.
External Links: [Document](https://dx.doi.org/10.1109/TCSVT.2024.3455576)
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [117]
Z. Li, C. Huang, X. Wang, H. Hu, C. Wyeth, D. Bu, Q. Yu, W. Gao, X. Liu, and M. Li (2025)
Lossless data compression by large models.
Nature Machine Intelligence, pp. 1–6.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p4.1).
- [118]
Y. Liang, X. Li, X. Chen, Y. Zheng, H. Chen, B. Li, and X. Xue (2025)
Training-free pyramid token pruning for efficient large vision-language models via region, token, and instruction-guided importance.
arXiv preprint arXiv:2509.15704.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1).
- [119]
B. Lin, Y. Ye, B. Zhu, J. Cui, M. Ning, P. Jin, and L. Yuan (2024)
Video-llava: learning united visual representation by alignment before projection.
In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pp. 5971–5984.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [120]
Z. Lin, M. Lin, L. Lin, and R. Ji (2025)
Boosting multimodal large language models with visual tokens withdrawal for rapid inference.
In aaai,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [121]
Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2022)
Flow matching for generative modeling.
arXiv preprint arXiv:2210.02747.
Cited by: [Figure 5](https://arxiv.org/html/2601.20742v1#S3.F5).
- [122]
H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee (2024)
Llavanext: improved reasoning, ocr, and world knowledge.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p1.13),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1).
- [123]
H. Liu, C. Li, Q. Wu, and Y. J. Lee (2023)
Visual instruction tuning.
In nips,
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p1.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.6.1),
[Figure 10](https://arxiv.org/html/2601.20742v1#S4.F10),
[§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1),
[§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p6.1),
[TABLE IV](https://arxiv.org/html/2601.20742v1#S4.T4.12.1),
[TABLE IV](https://arxiv.org/html/2601.20742v1#S4.T4.2.2).
- [124]
H. Liu, C. Li, Y. Yan, Y. Li, and Y. J. Lee (2023)
Improved baselines with visual instruction tuning.
arXiv preprint arXiv:2310.03744.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [125]
J. Liu, R. Feng, Y. Qi, Q. Chen, Z. Chen, W. Zeng, and X. Jin (2024)
Rate-distortion-cognition controllable versatile neural image compression.
In European Conference on Computer Vision,
pp. 329–348.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [126]
J. Liu, Z. Jia, J. Li, B. Li, X. Jin, W. Zeng, and Y. Lu (2025)
When mllms meet compression distortion: a coding paradigm tailored to mllms.
arXiv preprint arXiv:2509.24258.
Cited by: [Figure 11](https://arxiv.org/html/2601.20742v1#S4.F11),
[Figure 12](https://arxiv.org/html/2601.20742v1#S4.F12),
[§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [127]
J. Liu, J. Lin, Y. Wei, K. Shao, K. Tao, J. Huang, X. Yang, Z. Chen, H. Wang, and X. Jin (2025)
Revisiting mllm token technology through the lens of classical visual coding.
arXiv preprint arXiv:2508.13460.
Cited by: [§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p3.1).
- [128]
J. Liu, H. Sun, and J. Katto (2022)
Improving multiple machine vision tasks in the compressed domain.
In 2022 26th International Conference on Pattern Recognition (ICPR),
pp. 331–337.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2).
- [129]
J. Liu, H. Sun, and J. Katto (2022)
Semantic segmentation in learned compressed domain.
In 2022 Picture Coding Symposium (PCS),
pp. 181–185.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [130]
J. Liu, H. Sun, and J. Katto (2023)
Learned image compression with mixed transformer-cnn architectures.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 14388–14397.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.2.2](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS2.p1.1),
[§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1),
[§2](https://arxiv.org/html/2601.20742v1#S2.p1.1),
[§4.2.1](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS1.p1.1).
- [131]
J. Liu, Y. Wei, J. Lin, S. Zhao, H. Sun, Z. Chen, W. Zeng, and X. Jin (2024)
Tell codec what worth compressing: semantically disentangled image coding for machine with lmms.
In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP),
pp. 1–5.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p2.1).
- [132]
S. Liu, C. Wang, Y. Zhang, and H. Li (2025)
Dynamic-vlm: instance-aware token budgeting for efficient multimodal generation.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [133]
T. Liu, L. Shi, R. Hong, Y. Hu, Q. Yin, and L. Zhang (2024)
Multi-stage vision token dropping: towards efficient multimodal large language model.
arXiv preprint arXiv:2411.10803.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [134]
X. Liu, Z. Wang, J. Chen, Y. Han, Y. Wang, J. Yuan, J. Song, L. Zhang, S. Huang, and H. Chen (2025)
Global compression commander: plug-and-play inference acceleration for high-resolution large vision-language models.
arXiv preprint arXiv:2501.05179.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [135]
Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. (2024)
Mmbench: is your multi-modal model an all-around player?.
In European conference on computer vision,
pp. 216–233.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [136]
C. Lu, S. Liang, X. Wang, and W. Wang (2025)
Reinforcement learning-based token pruning in vision transformers: a markov game approach.
arXiv preprint arXiv:2503.23459.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.10.1.1.1).
- [137]
G. Lu, W. Ouyang, D. Xu, X. Zhang, C. Cai, and Z. Gao (2019)
Dvc: an end-to-end deep video compression framework.
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 11006–11015.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2).
- [138]
J. Lu, D. Batra, D. Parikh, and S. Lee (2019)
Vilbert: pretraining task-agnostic visiolinguistic representations for vision-and-language tasks.
Advances in neural information processing systems 32.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [139]
J. Lu, L. Song, M. Xu, B. Ahn, Y. Wang, C. Chen, A. Dehghan, and Y. Yang (2025)
Atoken: a unified tokenizer for vision.
arXiv preprint arXiv:2509.14476.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.5.3](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS3.p1.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p4.1).
- [140]
J. Lu, L. Zhang, X. Zhou, M. Li, W. Li, and S. Gu (2025)
Learned image compression with dictionary-based entropy model.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 12850–12859.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.2.2](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS2.p1.1),
[Figure 12](https://arxiv.org/html/2601.20742v1#S4.F12).
- [141]
Z. Lu, R. Li, K. Lu, X. Chen, E. Hossain, Z. Zhao, and H. Zhang (2023)
Semantics-empowered communications: a tutorial-cum-survey.
IEEE Communications Surveys & Tutorials 26 (1), pp. 41–79.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [142]
J. Luo, Y. Zhang, X. Yang, K. Wu, Q. Zhu, L. Liang, J. Chen, and Y. Li (2025)
When large vision-language model meets large remote sensing imagery: coarse-to-fine text-guided token pruning.
arXiv preprint arXiv:2503.07588.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1).
- [143]
C. Ma, Y. Jiang, J. Wu, J. Yang, X. Yu, Z. Yuan, B. Peng, and X. Qi (2025)
Unitok: a unified tokenizer for visual generation and understanding.
arXiv preprint arXiv:2502.20321.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.5.3](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS3.p1.1).
- [144]
Z. Ma, Y. Zhu, X. Zhou, et al. (2025)
VideoChat-flash: towards fast and accurate video-language understanding.
arXiv preprint arXiv:2501.00574.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1).
- [145]
M. Maaz, H. Rasheed, S. Khan, and F. Khan (2024)
Video-chatgpt: towards detailed video understanding via large vision and language models.
In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 12585–12602.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [146]
R. Mao, X. Feng, C. Gao, L. Li, D. Liu, and X. Sun (2024)
Perceptual image compression with conditional diffusion transformers.
In 2024 IEEE International Conference on Visual Communications and Image Processing (VCIP),
pp. 1–5.
Cited by: [§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [147]
P. Mary, J. Gorce, A. Unsal, and H. V. Poor (2016)
Finite blocklength information theory: what is the practical impact on wireless communications?.
In 2016 IEEE Globecom Workshops (GC Wkshps),
pp. 1–6.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [148]
F. Mentzer, G. D. Toderici, M. Tschannen, and E. Agustsson (2020)
High-fidelity generative image compression.
Advances in neural information processing systems 33, pp. 11913–11924.
Cited by: [§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [149]
S. Minaee, T. Mikolov, N. Nikzad, M. Chenaghlu, R. Socher, X. Amatriain, and J. Gao (2024)
Large language models: a survey.
arXiv preprint arXiv:2402.06196.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [150]
D. Minnen, J. Ballé, and G. D. Toderici (2018)
Joint autoregressive and hierarchical priors for learned image compression.
Advances in neural information processing systems 31.
Cited by: [§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1),
[§2](https://arxiv.org/html/2601.20742v1#S2.p1.1),
[§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1).
- [151]
D. Minnen and S. Singh (2020)
Channel-wise autoregressive entropy models for learned image compression.
In 2020 IEEE International Conference on Image Processing (ICIP),
pp. 3339–3343.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1).
- [152]
MPEG (2025)
Explorations: video coding for machines (part 34).
Note: [https://www.mpeg.org/standards/Explorations/34/](https://www.mpeg.org/standards/Explorations/34/)Accessed: 2025-12-16
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [153]
MPEG (2025)
Explorations: video coding for machines (vcm).
Note: [https://www.mpeg.org/standards/Explorations/34/](https://www.mpeg.org/standards/Explorations/34/)Accessed Dec. 2025
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [154]
M. J. Muckley, A. El-Nouby, K. Ullrich, H. Jégou, and J. Verbeek (2023)
Improving statistical fidelity for neural image compression with implicit local likelihood models.
In International Conference on Machine Learning,
pp. 25426–25443.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [155]
R. F. Nalewajski (2011)
Elements of information theory.
In Perspectives in Electronic Structure Theory,
pp. 371–395.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [156]
B. Nazer and M. Gastpar (2007)
Computation over multiple-access channels.
IEEE Transactions on Information Theory 53 (10), pp. 3498–3516.
External Links: [Document](https://dx.doi.org/10.1109/TIT.2007.904785)
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [157]
K. O’shea and R. Nash (2015)
An introduction to convolutional neural networks.
arXiv preprint arXiv:1511.08458.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [158]
M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby, et al. (2024)
DINOv2: learning robust visual features without supervision.
Transactions on Machine Learning Research Journal.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [159]
T. Park, E. Hong, Y. Jeon, N. Lee, and Y. Kim (2025)
Robust deep joint source channel coding for task-oriented semantic communications.
arXiv preprint arXiv:2503.12907.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [160]
B. Peng, C. Li, W. Zeng, et al. (2023)
Kosmos-2: grounding multimodal large language models to the world.
arXiv preprint arXiv:2306.14824.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [161]
K. Pertsch, K. Stachowicz, B. Ichter, D. Driess, S. Nair, Q. Vuong, O. Mees, C. Finn, and S. Levine (2025)
Fast: efficient action tokenization for vision-language-action models.
preprint arXiv:2501.09747.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p2.1),
[§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [162]
Y. Polyanskiy, H. V. Poor, and S. Verdú (2010)
Channel coding rate in the finite blocklength regime.
IEEE Transactions on Information Theory 56 (5), pp. 2307–2359.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [163]
L. Qiao, M. Boloursaz Mashhadi, Z. Gao, R. Schober, and D. Gündüz (2025)
ToDMA: large model-driven token-domain multiple access for semantic communications.
External Links: 2505.10946
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [164]
L. Qiao, M. Boloursaz Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. Niyato (2025)
Token communications: a unified framework for cross-modal context-aware semantic communications.
External Links: 2502.12096
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [165]
L. Qiao, M. B. Mashhadi, Z. Gao, R. Tafazolli, M. Bennis, and D. T. Niyato (2025)
Token communications: a large model-driven framework for cross-modal context-aware semantic communications.
IEEE Wireless Communications Magazine.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p3.1).
- [166]
L. Qu, H. Zhang, Y. Liu, X. Wang, Y. Jiang, Y. Gao, H. Ye, D. K. Du, Z. Yuan, and X. Wu (2025)
Tokenflow: unified image tokenizer for multimodal understanding and generation.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 2545–2555.
Cited by: [§3.5.3](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS3.p1.1).
- [167]
M. Rabbani and R. Joshi (2002)
An overview of the jpeg 2000 still image compression standard.
Signal processing: Image communication 17 (1), pp. 3–48.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§2.2.1](https://arxiv.org/html/2601.20742v1#S2.SS2.SSS1.p1.1),
[§2](https://arxiv.org/html/2601.20742v1#S2.p1.1).
- [168]
A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever (2021)
Learning transferable visual models from natural language supervision.
In Proceedings of the 38th International Conference on Machine Learning (ICML),
Proceedings of Machine Learning Research, Vol. 139, pp. 8748–8763.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p2.4),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.3.1),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [169]
A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever (2021)
Zero-shot text-to-image generation.
In Proceedings of the 38th International Conference on Machine Learning (ICML),
Proceedings of Machine Learning Research, Vol. 139, pp. 8821–8831.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.1.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p2.1).
- [170]
Y. Rao, W. Zhao, B. Liu, J. Lu, J. Zhou, and C. Hsieh (2021)
Dynamicvit: efficient vision transformers with dynamic token sparsification.
Advances in neural information processing systems 34, pp. 13937–13949.
Cited by: [TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.4.1.1.1).
- [171]
S. Ren, Y. Liu, K. Xu, et al. (2024)
TimeChat: a time-sensitive multimodal large language model for long video understanding.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1).
- [172]
C. Renggli, M. Minderer, Y. Tay, et al. (2022)
PatchMerger: reducing the number of tokens in vision transformers.
arXiv preprint arXiv:2202.12015.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [173]
R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer (2022)
High-resolution image synthesis with latent diffusion models.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
pp. 10684–10695.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.15.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p2.1).
- [174]
M. S. Ryoo, H. Zhou, S. Kendre, C. Qin, L. Xue, M. Shu, J. Park, K. Ranasinghe, S. Savarese, R. Xu, et al. (2024)
Xgen-mm-vid (blip-3-video): you only need 32 tokens to represent a video even in vlms.
arXiv preprint arXiv:2410.16267.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [175]
M. S. Ryoo, K. Gopalakrishnan, K. Kahatapitiya, T. Xiao, K. Rao, A. Stone, Y. Lu, J. Ibarz, and A. Arnab (2023)
Token turing machines.
External Links: 2211.09119,
[Link](https://arxiv.org/abs/2211.09119)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [176]
A. Şahin and R. Yang (2023)
A survey on over-the-air computation.
IEEE Communications Surveys & Tutorials 25 (3), pp. 1877–1908.
External Links: [Document](https://dx.doi.org/10.1109/COMST.2023.3264649)
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [177]
T. Saikh, T. Ghosal, A. Mittal, A. Ekbal, and P. Bhattacharyya (2022)
Scienceqa: a novel resource for question answering on scholarly articles.
International Journal on Digital Libraries 23 (3), pp. 289–301.
Cited by: [Figure 10](https://arxiv.org/html/2601.20742v1#S4.F10),
[§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p6.1).
- [178]
J. Scarlett, A. Martinez, and A. G. i Fàbregas (2014)
Second-order rate region of constant-composition codes for the multiple-access channel.
IEEE Transactions on Information Theory 61 (1), pp. 157–172.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [179]
Y. Shang, H. Sun, Z. Dong, X. Shen, M. Ma, B. Liu, B. Fu, H. Chen, Z. Zhang, Y. Jiang, et al. (2024)
LLaVA-prumerge: adaptive token reduction for efficient large multimodal models.
arXiv preprint arXiv:2403.15388.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1),
[TABLE IV](https://arxiv.org/html/2601.20742v1#S4.T4).
- [180]
C. Shani, L. Soffer, D. Jurafsky, Y. LeCun, and R. Shwartz-Ziv (2025)
From tokens to thoughts: how llms and humans trade compression for meaning.
arXiv preprint arXiv:2505.17117.
Cited by: [§4.2.1](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS1.p1.1),
[§4.2.3](https://arxiv.org/html/2601.20742v1#S4.SS2.SSS3.p1.2.2),
[§4.3.3](https://arxiv.org/html/2601.20742v1#S4.SS3.SSS3.p1.1).
- [181]
C. E. Shannon (1948)
A mathematical theory of communication.
The Bell system technical journal 27 (3), pp. 379–423.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [182]
K. Shao, K. Tao, C. Qin, H. You, Y. Sui, and H. Wang (2025)
Holitom: holistic token merging for fast video large language models.
preprint arXiv:2505.21334.
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.9.1.1.1).
- [183]
L. Shen, G. Gong, T. He, Y. Zhang, P. Liu, S. Zhao, and G. Ding (2025)
Fastvid: dynamic density pruning for fast video large language models.
arXiv preprint arXiv:2503.11187.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [184]
X. Shen, Y. Xiong, C. Zhao, L. Wu, J. Chen, C. Zhu, Z. Liu, F. Xiao, B. Varadarajan, F. Bordes, Z. Liu, H. Xu, H. J. Kim, B. Soran, R. Krishnamoorthi, M. Elhoseiny, and V. Chandra (2024)
LongVU: spatiotemporal adaptive compression for long video-language understanding.
External Links: 2410.17434,
[Link](https://arxiv.org/abs/2410.17434)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [185]
H. Shi, B. Xie, Y. Liu, L. Sun, F. Liu, T. Wang, E. Zhou, H. Fan, X. Zhang, and G. Huang (2025)
MemoryVLA: perceptual-cognitive memory in vision-language-action models for robotic manipulation.
External Links: 2508.19236,
[Link](https://arxiv.org/abs/2508.19236)
Cited by: [§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1).
- [186]
A. Singh, V. Natarjan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach (2019)
Towards vqa models that can read.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition,
pp. 8317–8326.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [187]
W. Song, Y. Wang, Z. Song, Y. Li, H. Sun, W. Chen, Z. Zhou, J. Xu, J. Wang, and K. Yu (2025)
Dualtoken: towards unifying visual understanding and generation with dual visual vocabularies.
arXiv preprint arXiv:2503.14324.
Cited by: [§3.5.3](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS3.p1.1).
- [188]
M. Stefanini, M. Cornia, L. Baraldi, S. Cascianelli, G. Fiameni, and R. Cucchiara (2022)
From show to tell: a survey on deep learning-based image captioning.
IEEE transactions on pattern analysis and machine intelligence 45 (1), pp. 539–559.
Cited by: [§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1).
- [189]
B. Sun, J. Zhao, X. Wei, and Q. Hou (2025)
LLaVA-scissor: token compression with semantic connected components for video llms.
arXiv preprint arXiv:2506.21862.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [190]
Q. Sun, Y. Cui, X. Zhang, F. Zhang, Q. Yu, Y. Wang, Y. Rao, J. Liu, T. Huang, and X. Wang (2024)
Generative multimodal models are in-context learners.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 14398–14409.
Cited by: [§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p1.1).
- [191]
Q. Sun, Q. Yu, Y. Cui, F. Zhang, X. Wang, et al. (2023)
Emu: generative pretraining in multimodality.
arXiv preprint arXiv:2307.05222.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p3.2).
- [192]
Y. Sun, Y. Xin, H. Li, J. Sun, C. Lin, and R. T. Batista-Navarro (2025)
Lvpruning: an effective yet simple language-guided vision token pruning approach for multi-modal large language models.
In Findings of the Association for Computational Linguistics: NAACL 2025,
pp. 4299–4308.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [193]
D. Surís, S. Menon, and C. Vondrick (2023)
ViperGPT: visual inference via python execution for reasoning.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 11720–11730.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [194]
H. Tan and M. Bansal (2019)
Lxmert: learning cross-modality encoder representations from transformers.
arXiv preprint arXiv:1908.07490.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p2.1).
- [195]
X. Tan, P. Ye, C. Tu, J. Cao, Y. Yang, L. Zhang, D. Zhou, and T. Chen (2025)
Tokencarve: information-preserving visual token compression in multimodal large language models.
preprint arXiv:2503.10501.
Cited by: [§4.6](https://arxiv.org/html/2601.20742v1#S4.SS6.p3.1).
- [196]
C. Team (2024)
Chameleon: mixed-modal early-fusion foundation models.
arXiv preprint arXiv:2405.09818.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p4.2),
[§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p1.1).
- [197]
S. Teerapittayanon, B. McDanel, and H. Kung (2017)
Distributed deep neural networks over the cloud, the edge and end devices.
In 2017 IEEE 37th international conference on distributed computing systems (ICDCS),
pp. 328–339.
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p1.1).
- [198]
L. Theis, T. Salimans, M. D. Hoffman, and F. Mentzer (2022)
Lossy compression with gaussian diffusion.
arXiv preprint arXiv:2206.08889.
Cited by: [Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.5.1](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS1.p2.1).
- [199]
K. Tian, Y. Jiang, Z. Yuan, B. Peng, and L. Wang (2024)
Visual autoregressive modeling: scalable image generation via next-scale prediction.
Advances in neural information processing systems 37, pp. 84839–84865.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1).
- [200]
I. O. Tolstikhin, N. Houlsby, A. Kolesnikov, L. Beyer, X. Zhai, T. Unterthiner, J. Yung, A. Steiner, D. Keysers, J. Uszkoreit, et al. (2021)
Mlp-mixer: an all-mlp architecture for vision.
Advances in neural information processing systems 34, pp. 24261–24272.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [201]
K. Tong, Y. Wu, Y. Li, K. Zhang, L. Zhang, and X. Jin (2023)
Qvrf: a quantization-error-aware variable rate framework for learned image compression.
In 2023 IEEE International Conference on Image Processing (ICIP),
pp. 1310–1314.
Cited by: [§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1).
- [202]
M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa, et al. (2025)
Siglip 2: multilingual vision-language encoders with improved semantic understanding, localization, and dense features.
preprint arXiv:2502.14786.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p2.4),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.4.1).
- [203]
R. Tu, S. Xie, et al. (2024)
VL-cache: learning to cache visual tokens for efficient multimodal llms.
arXiv preprint arXiv:2410.23317.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [204]
A. van den Oord, O. Vinyals, and K. Kavukcuoglu (2017)
Neural discrete representation learning.
In Advances in Neural Information Processing Systems (NeurIPS),
Vol. 30.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.2.2](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS2.p1.1),
[§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.10.1),
[§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p3.1),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [205]
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin (2017)
Attention is all you need.
Advances in neural information processing systems 30.
Cited by: [Figure 5](https://arxiv.org/html/2601.20742v1#S3.F5),
[§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [206]
G. K. Wallace (1991)
The jpeg still picture compression standard.
Communications of the ACM 34 (4), pp. 30–44.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[Figure 2](https://arxiv.org/html/2601.20742v1#S2.F2),
[§2.1](https://arxiv.org/html/2601.20742v1#S2.SS1.p1.1),
[§2.3.1](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS1.p1.1),
[§4.3.2](https://arxiv.org/html/2601.20742v1#S4.SS3.SSS2.p1.1).
- [207]
Z. Wan, Z. Wang, H. Zhang, et al. (2024)
LOOK-m: learning to organize kv cache for multimodal llms.
In Findings of the Association for Computational Linguistics: EMNLP 2024,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [208]
H. Wang, Z. Yu, G. Spadaro, C. Ju, V. Quétu, S. Xiao, and E. Tartaglione (2025)
Folder: accelerating multi-modal large language models with enhanced performance.
In iccv,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [209]
J. Wang, Y. Jiang, Z. Yuan, B. Peng, Z. Wu, and Y. Jiang (2024)
OmniTokenizer: a joint image-video tokenizer for visual generation.
External Links: 2406.09399,
[Link](https://arxiv.org/abs/2406.09399)
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1).
- [210]
P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. (2024)
Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution.
preprint arXiv:2409.12191.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1),
[§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p3.1).
- [211]
W. Wang, Z. Gao, L. Gu, H. Pu, L. Cui, X. Wei, Z. Liu, L. Jing, S. Ye, J. Shao, et al. (2025)
Internvl3. 5: advancing open-source multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p1.13).
- [212]
Y. Wang, K. Li, Y. Li, Y. He, B. Huang, Z. Zhao, H. Zhang, J. Xu, Y. Liu, Z. Wang, et al. (2022)
Internvideo: general video foundation models via generative and discriminative learning.
arXiv preprint arXiv:2212.03191.
Cited by: [§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.9.1).
- [213]
Z. Wang, L. Zou, S. Wei, K. Li, F. Liao, H. Mi, and R. Lai (2025)
Large-language-model-enabled text semantic communication systems.
Applied Sciences 15 (13), pp. 7227.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [214]
H. Wei, Y. Sun, and Y. Li (2025)
DeepSeek-ocr: contexts optical compression.
arXiv preprint arXiv:2510.18234.
Cited by: [TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.2.1.1.1).
- [215]
Y. Wen, Q. Cao, Q. Fu, S. Mehta, and M. Najibi (2024)
Efficient vision-language models by summarizing visual tokens into compact registers.
arXiv preprint arXiv:2410.14072.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [216]
J. Wolf and J. Ziv (1970)
Transmission of noisy information to a noisy receiver with minimum distortion.
IEEE Transactions on Information Theory 16 (4), pp. 406–411.
Cited by: [§4.4.1](https://arxiv.org/html/2601.20742v1#S4.SS4.SSS1.p1.2).
- [217]
C. Wu, X. Chen, Z. Wu, Y. Ma, X. Liu, Z. Pan, W. Liu, Z. Xie, X. Yu, C. Ruan, et al. (2025)
Janus: decoupling visual encoding for unified multimodal understanding and generation.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 12966–12977.
Cited by: [§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p2.1).
- [218]
P. Wu, Z. Yu, Y. Liu, C. Wu, E. Zhou, and J. Shen (2025)
MARC: memory-augmented rl token compression for efficient video understanding.
arXiv preprint arXiv:2510.07915.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1).
- [219]
Y. Wu, Y. Zhang, Y. Wang, et al. (2024)
VILA-u: a unified foundation model integrating visual understanding and generation.
arXiv preprint arXiv:2409.04429.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p4.2),
[§3.5.3](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS3.p1.1).
- [220]
Z. Xiao, C. Ye, Y. Feng, Y. Hu, T. Jiao, L. Cai, and G. Liu (2025)
Transmission with machine language tokens: a paradigm for task-oriented agent communication.
External Links: 2507.21454
Cited by: [§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p1.1),
[§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [221]
H. Xie, Z. Qin, G. Y. Li, and B. Juang (2021)
Deep learning enabled semantic communication systems.
IEEE transactions on signal processing 69, pp. 2663–2675.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [222]
H. Xie, Z. Qin, X. Tao, and K. B. Letaief (2022)
Task-oriented multi-user semantic communications.
IEEE Journal on Selected Areas in Communications 40 (9), pp. 2584–2597.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [223]
J. Xie, Z. Zhang, Z. Li, et al. (2024)
Show-o: one single transformer to unify multimodal understanding and generation.
arXiv preprint arXiv:2408.12528.
Cited by: [Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p3.2),
[§3.5.2](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS2.p1.1).
- [224]
J. Xie, Z. Yang, and M. Z. Shou (2025)
Show-o2: improved native unified multimodal models.
arXiv preprint arXiv:2506.15564.
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p4.2),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [225]
L. Xing, Q. Huang, X. Dong, J. Lu, P. Zhang, Y. Zang, Y. Cao, C. He, J. Wang, F. Wu, et al. (2025)
Pyramiddrop: accelerating your large vision-language models via pyramid visual redundancy reduction.
In cvpr,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [226]
C. Xu, H. Zhao, R. Zhang, and W. Li (2025)
CoViPAL: contextualized visual pruning across layers for efficient lvlms.
In Findings of the Association for Computational Linguistics: EMNLP,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p7.1).
- [227]
L. Xu, Y. Zhao, D. Zhou, Z. Lin, S. K. Ng, and J. Feng (2024)
PLLaVA : parameter-free llava extension from images to videos for video dense captioning.
External Links: 2404.16994,
[Link](https://arxiv.org/abs/2404.16994)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [228]
C. Yang, Y. Sui, J. Xiao, L. Huang, Y. Gong, C. Li, J. Yan, Y. Bai, P. Sadayappan, X. Hu, et al. (2025)
Topv: compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model.
In cvpr,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p2.1).
- [229]
L. Yang, Z. Zhang, Y. Song, S. Hong, R. Xu, Y. Zhao, W. Zhang, B. Cui, and M. Yang (2023)
Diffusion models: a comprehensive survey of methods and applications.
ACM computing surveys 56 (4), pp. 1–39.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [230]
M. Yang, C. Bian, and H. Kim (2021)
Deep joint source channel coding for wireless image transmission with ofdm.
In ICC 2021-IEEE International Conference on Communications,
pp. 1–6.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1),
[§5.3.3](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS3.p2.1).
- [231]
S. Yang, Y. Chen, Z. Tian, C. Wang, J. Li, B. Yu, and J. Jia (2025)
Visionzip: longer is better but not necessary in vision language models.
In cvpr,
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.7.1.1.1).
- [232]
S. Yang, J. Li, X. Lai, B. Yu, H. Zhao, and J. Jia (2025)
Visionthink: smart and efficient vision language model via reinforcement learning.
arXiv preprint arXiv:2507.13348.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.11.1.1.1).
- [233]
W. Yang, H. Du, Z. Q. Liew, W. Y. B. Lim, Z. Xiong, D. Niyato, X. Chi, X. Shen, and C. Miao (2022)
Semantic communications for future internet: fundamentals, applications, and challenges.
IEEE Communications Surveys & Tutorials 25 (1), pp. 213–250.
Cited by: [§5.3.2](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS2.p1.1).
- [234]
W. Yang, H. Huang, Y. Hu, L. Duan, and J. Liu (2024)
Video coding for machines: compact visual representation compression for intelligent collaborative analytics.
IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (7), pp. 5174–5191.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [235]
Y. Yang, S. Mandt, L. Theis, et al. (2023)
An introduction to neural data compression.
Foundations and Trends® in Computer Graphics and Vision 15 (2), pp. 113–200.
Cited by: [§2.3.2](https://arxiv.org/html/2601.20742v1#S2.SS3.SSS2.p1.1).
- [236]
Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, et al. (2024)
Cogvideox: text-to-video diffusion models with an expert transformer.
arXiv preprint arXiv:2408.06072.
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [237]
L. Yao, L. Li, S. Ren, L. Wang, Y. Liu, X. Sun, and L. Hou (2024)
Deco: decoupling token compression from semantic abstraction in multimodal large language models.
arXiv preprint arXiv:2405.20985.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [238]
M. H. Yassaee, M. R. Aref, and A. Gohari (2013)
A technique for deriving one-shot achievability results in network information theory.
In 2013 IEEE International Symposium on Information Theory,
pp. 1287–1291.
Cited by: [§5.3.1](https://arxiv.org/html/2601.20742v1#S5.SS3.SSS1.p1.1).
- [239]
Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi, et al. (2023)
Mplug-owl: modularization empowers large language models with multimodality.
preprint arXiv:2304.14178.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [240]
W. Ye, Q. Wu, W. Lin, and Y. Zhou (2025)
Fit and prune: fast and training-free visual token pruning for multi-modal large language models.
In aaai,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [241]
X. Ye, Y. Gan, Y. Ge, X. Zhang, and Y. Tang (2025)
Atp-llava: adaptive token pruning for large vision language models.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 24972–24982.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [242]
X. Ye, Y. Gan, X. Huang, Y. Ge, and Y. Tang (2025)
Voco-llama: towards vision compression with large language models.
In Proceedings of the Computer Vision and Pattern Recognition Conference,
pp. 29836–29846.
Cited by: [TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.3.1.1.1).
- [243]
K. Yin, Q. Liu, X. Shen, Y. He, W. Yang, and S. Wang (2025)
Unified coding for both human perception and generalized machine analytics with clip supervision.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39, pp. 9517–9525.
Cited by: [§5.2.2](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS2.p1.1).
- [244]
L. Yu, Y. Cheng, K. Sohn, J. Lezama, H. Zhang, H. Chang, A. G. Hauptmann, M. Yang, Y. Hao, I. Essa, et al. (2023)
Magvit: masked generative video transformer.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 10459–10469.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p1.1).
- [245]
L. Yu, J. Lezama, N. B. Gundavarapu, L. Versari, K. Sohn, D. Minnen, Y. Cheng, V. Birodkar, A. Gupta, X. Gu, et al. (2023)
Language model beats diffusion–tokenizer is key to visual generation.
arXiv preprint arXiv:2310.05737.
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.2.2](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS2.p3.1),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.13.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p3.1).
- [246]
L. Yu, R. Bavishi, A. Sharma, et al. (2023)
Scaling autoregressive multi-modal models: pretraining and instruction tuning.
arXiv preprint arXiv:2309.02591.
Note: CM3LeOn
Cited by: [§3.1.3](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS3.p3.2),
[§3.2.2](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS2.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.14.1).
- [247]
Q. Yu, M. Weber, X. Deng, X. Shen, D. Cremers, and L. Chen (2024)
An image is worth 32 tokens for reconstruction and generation.
Advances in Neural Information Processing Systems 37, pp. 128940–128966.
Cited by: [Figure 5](https://arxiv.org/html/2601.20742v1#S3.F5),
[§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p2.1).
- [248]
S. Yu, S. Kwak, H. Jang, J. Jeong, J. Huang, J. Shin, and S. Xie (2024)
Representation alignment for generation: training diffusion transformers is easier than you think.
arXiv preprint arXiv:2410.06940.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.16.1),
[§5.1.1](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS1.p2.1).
- [249]
Z. Yu, D. Xu, J. Yu, T. Yu, Z. Zhao, Y. Zhuang, and D. Tao (2019)
ActivityNet-qa: a dataset for understanding complex web videos via question answering.
External Links: 1906.02467,
[Link](https://arxiv.org/abs/1906.02467)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p4.1).
- [250]
X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. (2024)
Mmmu: a massive multi-discipline multimodal understanding and reasoning benchmark for expert agi.
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,
pp. 9556–9567.
Cited by: [§4.7](https://arxiv.org/html/2601.20742v1#S4.SS7.p5.1).
- [251]
X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer (2023)
Sigmoid loss for language image pre-training.
In Proceedings of the IEEE/CVF international conference on computer vision,
pp. 11975–11986.
Cited by: [§1](https://arxiv.org/html/2601.20742v1#S1.p2.1),
[Figure 7](https://arxiv.org/html/2601.20742v1#S3.F7),
[§3.5.1](https://arxiv.org/html/2601.20742v1#S3.SS5.SSS1.p1.1),
[§5.1.2](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS2.p2.1),
[§5.1.3](https://arxiv.org/html/2601.20742v1#S5.SS1.SSS3.p1.1).
- [252]
C. Zhang, K. Ma, T. Fang, W. Yu, H. Zhang, Z. Zhang, Y. Xie, K. Sycara, H. Mi, and D. Yu (2025)
VScan: rethinking visual token reduction for efficient large vision-language models.
arXiv preprint arXiv:2505.22654.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [253]
H. Zhang, X. Li, and L. Bing (2023)
Video-llama: an instruction-tuned audio-visual language model for video understanding.
arXiv preprint arXiv:2306.02858.
Cited by: [§3.2.1](https://arxiv.org/html/2601.20742v1#S3.SS2.SSS1.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.8.1).
- [254]
H. Zhang, J. Zhang, X. Ji, Q. Wang, and F. Zhang (2025)
DynTok: dynamic compression of visual tokens for efficient and effective video understanding.
arXiv preprint arXiv:2506.03990.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [255]
J. Zhang, T. Lin, X. Li, H. Wang, and Y. Guo (2025)
Rethinking visual token reduction in lvlms under cross-modal misalignment.
arXiv preprint arXiv:2506.22283.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p6.1).
- [256]
L. Zhang, A. Rao, and M. Agrawala (2023)
Adding conditional control to text-to-image diffusion models.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV),
pp. 3813–3824.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p4.1).
- [257]
Q. Zhang, S. Wang, X. Zhang, C. Jia, Z. Wang, S. Ma, and W. Gao (2024)
Perceptual video coding for machines via satisfied machine ratio modeling.
IEEE Transactions on Pattern Analysis and Machine Intelligence 46 (12), pp. 7651–7668.
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [258]
Q. Zhang, A. Cheng, M. Lu, R. Zhang, Z. Zhuo, J. Cao, S. Guo, Q. She, and S. Zhang (2025)
Beyond text-visual attention: exploiting visual cues for effective token pruning in vlms.
In ICCV,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[TABLE I](https://arxiv.org/html/2601.20742v1#S3.T1.3.12.1.1.1).
- [259]
S. Zhang, Q. Fang, Z. Yang, and Y. Feng (2025)
Llava-mini: efficient image and video large multimodal models with one vision token.
arXiv preprint arXiv:2501.03895.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).
- [260]
Y. Zhang, C. Rosewarne, S. Liu, and C. Hollmann (2022)
Call for evidence for video coding for machines.
ISO/IEC JTC 1/SC 29/WG 2.
Cited by: [§2.5.2](https://arxiv.org/html/2601.20742v1#S2.SS5.SSS2.p2.1).
- [261]
Y. Zhang, K. Yu, S. Wu, and Z. He (2024)
Conceptual codebook learning for vision-language models.
In European Conference on Computer Vision,
pp. 235–251.
Cited by: [§3.1.1](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS1.p1.1).
- [262]
Y. Zhang, C. Fan, J. Ma, W. Zheng, T. Huang, K. Cheng, D. Gudovskiy, T. Okuno, Y. Nakata, K. Keutzer, et al. (2024)
Sparsevlm: visual token sparsification for efficient vision-language model inference.
preprint arXiv:2410.04417.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1),
[§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p5.1).
- [263]
Y. Zhang (2022)
Video coding for machines (vcm): overview and future plan.
Note: [https://www.itu.int/en/ITU-T/Workshops-and-Seminars/2022/0118/Documents/Yuan%20Zhang.pdf](https://www.itu.int/en/ITU-T/Workshops-and-Seminars/2022/0118/Documents/Yuan%20Zhang.pdf)ITU Workshop Slides, Accessed Dec. 2025
Cited by: [§5.2.3](https://arxiv.org/html/2601.20742v1#S5.SS2.SSS3.p1.1).
- [264]
Z. Zhang, S. Leng, H. Cheng, et al. (2024)
Video-llama 2: advancing spatial-temporal modeling and audio understanding in video-llms.
arXiv preprint arXiv:2406.07476.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p4.1),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p3.1).
- [265]
S. Zhao, Y. Zhang, X. Cun, S. Yang, M. Niu, X. Li, W. Hu, and Y. Shan (2024)
CV-vae: a compatible video vae for latent generative video models.
External Links: 2405.20279,
[Link](https://arxiv.org/abs/2405.20279)
Cited by: [Figure 6](https://arxiv.org/html/2601.20742v1#S3.F6),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1),
[§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [266]
Y. Zhao, Y. Xiong, and P. Krähenbühl (2024)
Image and video tokenization with binary spherical quantization.
External Links: 2406.07548,
[Link](https://arxiv.org/abs/2406.07548)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p1.1).
- [267]
B. Zheng, N. Ma, S. Tong, and S. Xie (2025)
Diffusion transformers with representation autoencoders.
arXiv preprint arXiv:2510.11690.
Cited by: [§3.3](https://arxiv.org/html/2601.20742v1#S3.SS3.p3.1),
[TABLE II](https://arxiv.org/html/2601.20742v1#S3.T2.1.17.1).
- [268]
Z. Zheng, X. Peng, T. Yang, C. Shen, S. Li, H. Liu, Y. Zhou, T. Li, and Y. You (2024)
Open-sora: democratizing efficient video production for all.
External Links: 2412.20404,
[Link](https://arxiv.org/abs/2412.20404)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [269]
L. Zhou, C. Ruan, N. Ling, Z. Chen, W. Wang, and W. Jiang (2025)
TVC: tokenized video compression with ultra-low bit rate.
External Links: 2504.16953,
[Link](https://arxiv.org/abs/2504.16953)
Cited by: [§3.4.2](https://arxiv.org/html/2601.20742v1#S3.SS4.SSS2.p2.1).
- [270]
D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny (2024)
Minigpt-4: enhancing vision-language understanding with advanced large language models.
In iclr,
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7),
[§4.1](https://arxiv.org/html/2601.20742v1#S4.SS1.p1.1).
- [271]
J. Zhuang, L. Lu, M. Dai, R. Hu, J. Chen, Q. Liu, and H. Hu (2025)
St3: accelerating multimodal large language model by spatial-temporal visual token trimming.
In Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 39, pp. 11049–11057.
Cited by: [§3.1.2](https://arxiv.org/html/2601.20742v1#S3.SS1.SSS2.p3.7).