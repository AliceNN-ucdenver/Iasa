# 🔥 MASTER CERTIFICATION TABLE — Building AI Agents with Multimodal Models

## Complete Deep-Dive Study Guide for NVIDIA DLI Instructor Certification

---

# Part 1 — Multimodal Data

---

## 1.1 Multimodal Concept

**High-Level Concept:**
Multimodal AI combines different signal types (vision, audio, text, tabular) so models can reason using multiple "senses." Each human sense maps to one modality: vision = images/video, audition = audio/speech, language = text. No single modality captures complete reality — cameras miss depth, LiDAR misses color, text misses visual nuance.

**Deep Dive:**

Why multimodal matters:
- Single modality = partial view of reality
- Camera: rich texture/color, no depth, fails in darkness
- LiDAR: precise depth, sparse, no color/texture
- Text: abstract concepts, no visual grounding
- Audio: temporal patterns, no spatial info

Modality alignment challenges:
- Temporal sync (audio-video lip sync)
- Spatial alignment (LiDAR-camera calibration)
- Semantic alignment (image-caption correspondence)

Common architectures:
- **Dual encoder (CLIP):** separate encoders, shared embedding space
- **Cross-attention (Flamingo):** one modality attends to other
- **Early fusion (RGB-D):** concatenate inputs before encoder

**Notebook Context:** Lab 01a, 01b explore different modalities. Key takeaway: understand shape conventions `[B,C,H,W]` for vision, `[B,T]` for audio waveforms, `[B,N,D]` for sequences.

---

## 1.2 Vision Data Types

**High-Level Concept:**
RGB images `[3,H,W]`, grayscale `[1,H,W]`, depth maps `[1,H,W]` with distance values, point clouds `[N,3+]` sparse 3D coordinates. Each requires different preprocessing and model architectures.

**Deep Dive:**

**RGB Images:**
```
Shape: [Batch, 3, Height, Width]
Values: 0-255 raw → normalize to 0-1 or ImageNet stats
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
```

**Grayscale:**
```
Shape: [B, 1, H, W]
Conversion: gray = 0.2126*R + 0.7152*G + 0.0722*B

Why these weights? Matches human luminance perception (more green cones)
```

**Depth Maps:**
```
Shape: [B, 1, H, W] — pixel value = distance in meters
Dense (stereo cameras) vs sparse (projected LiDAR)
Often normalized: depth_norm = (depth - min) / (max - min)
```

**Point Clouds:**
```
Shape: [N, 3] minimum (x,y,z) or [N, 4+] with intensity/color
Sparse, unordered — can't use Conv2d directly
Processing: PointNet, voxelization, or projection to 2D
```

**Notebook Context:** Lab 01b — Load and visualize each type. Use `PIL` for images, `Open3D` for point clouds. Verify shapes match expected dimensions before feeding to models.

---

## 1.3 Audio Spectrogram

**High-Level Concept:**
Audio is transformed into a 2D image-like representation via Short-Time Fourier Transform (STFT) so vision models can process it. Raw audio = 1D waveform `[T samples]`. STFT windows the signal, applies FFT per window, stacks results into `[Freq, Time]` matrix. Mel scale warps frequency to match human perception.

**Deep Dive:**

**Step 1 — Raw Audio:**
```
Shape: [T] where T = samples (e.g., 16000/sec)
5 seconds @ 16kHz = [80000] values
Each value = microphone displacement amplitude
```

**Step 2 — STFT:**
```python
import librosa
stft = librosa.stft(y, n_fft=2048, hop_length=512)
# Output: [1025, T/hop] complex values
# 1025 = n_fft/2 + 1 frequency bins
```

**Step 3 — Mel Scale:**
```
Formula: mel = 2595 × log₁₀(1 + f/700)

Compresses high frequencies (perceptually uniform)
10kHz vs 10.1kHz sounds same; 100Hz vs 200Hz very different
```

```python
mel_spec = librosa.feature.melspectrogram(
    y=y, sr=16000, n_mels=128, hop_length=512
)
# Output: [128, T/hop]
```

**Step 4 — Log Amplitude:**
```python
mel_db = librosa.power_to_db(mel_spec)
# Converts to decibels (human loudness perception is log)
```

**Result:** `[1, 128, T]` — treat as grayscale image for CNN/ViT

**Notebook Context:** Lab 01b — Convert audio → mel spectrogram → visualize with `plt.imshow`. This proves audio AI is just image AI on spectrograms. Same augmentations work (crop, mask, mixup).

---

## 1.4 Color Mode of Image

**High-Level Concept:**
Channel ordering: RGB (PyTorch/PIL) vs BGR (OpenCV). Feeding BGR to RGB-trained model swaps red↔blue causing catastrophic accuracy drop. Critical preprocessing step.

**Deep Dive:**

**Modes:**
- **RGB:** Red-Green-Blue (PIL, PyTorch, most models)
- **BGR:** Blue-Green-Red (OpenCV default `cv2.imread()`)
- **RGBA:** RGB + Alpha (transparency)
- **Grayscale:** Single intensity channel

**Common Bug:**
```python
# WRONG — OpenCV loads BGR
img = cv2.imread("photo.jpg")  # [H,W,3] in BGR
tensor = transforms.ToTensor()(img)  # Assumes RGB!

# CORRECT — Convert first
img = cv2.imread("photo.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
tensor = transforms.ToTensor()(img)

# OR — Use PIL directly
from PIL import Image
img = Image.open("photo.jpg")  # Already RGB
tensor = transforms.ToTensor()(img)
```

**Visual effect of BGR→RGB swap:**
- Red apples appear blue
- Blue sky appears orange/red
- Model accuracy drops 50%+

**RGBA handling:**
```python
# Most models expect 3 channels
img = Image.open("logo.png")  # [H,W,4] RGBA
img = img.convert("RGB")  # Composites alpha onto white
```

**Notebook Context:** Lab 01b — Check image shapes and channel ordering. Use `cv2.cvtColor()` for conversion. Always verify first pixel values match expected color.

---

## 1.5 Shape of CT Scans

**High-Level Concept:**
3D volumetric data: `[D, H, W]` where D=depth (spatial slices through body), not time. Requires Conv3d or slice-wise processing. Hounsfield Units (HU) encode tissue density.

**Deep Dive:**

**Structure:**
```
Shape: [Batch, Channels, Depth, Height, Width]
       [B, 1, D, H, W]

Typical: [1, 1, 200, 512, 512]
         └── 200 slices, each 512×512

D = number of axial slices (spatial depth)
NOT video frames — adjacent slices show adjacent anatomy
```

**Hounsfield Units (HU):**
```
Air:         -1000 HU
Lung:         -500 HU
Water:           0 HU
Soft tissue:   +40 HU
Bone:        +1000 HU
Metal:       +3000 HU
```

**Windowing (visualization):**
```python
def apply_window(volume, center, width):
    """Map HU range to display range [0, 1]"""
    low = center - width // 2
    high = center + width // 2
    return np.clip((volume - low) / (high - low), 0, 1)

# Lung window: center=-600, width=1500
# Bone window: center=400, width=1800
# Same CT, different windows reveal different anatomy
```

**Processing options:**
- `Conv3d`: Full volumetric convolution (memory heavy)
- Slice-wise `Conv2d` + aggregation (lighter)
- 2.5D: Stack adjacent slices as channels

**Notebook Context:** Concept check in 01b. Key insight: You can't use standard Conv2d on raw CT — depth dimension contains spatial info (tumor context spans slices).

---

## 1.6 LiDAR vs Camera Data

**High-Level Concept:**
Camera = dense 2D color, ambiguous depth. LiDAR = sparse 3D structure, no texture. Understanding projection between coordinate systems is essential for sensor fusion in autonomous vehicles and robotics.

**Deep Dive:**

**Data Format Comparison:**

| Aspect | Camera | LiDAR |
|--------|--------|-------|
| Output | Dense image `[3,H,W]` | Sparse point cloud `[N,3+]` |
| Depth | None (2D projection) | Precise (time-of-flight) |
| Color | Full RGB | Intensity only |
| Density | Every pixel | 100k-1M points, gaps |
| Cost | $50-500 | $1k-75k |
| Weather | Fails in dark | Works at night |

**LiDAR Coordinate System:**
```
X: forward (vehicle direction)
Y: left
Z: up
Point: [x, y, z, intensity] in meters
```

**Camera Coordinate System:**
```
X: right (image columns)
Y: down (image rows)
Z: forward (depth)
Pixel: [u, v] in pixel units
```

**Notebook Context:** Lab 01b — Visualize point clouds with Open3D, project onto images. Understand why LiDAR appears "sparse" when projected — fewer points than pixels.

---

## 1.7 LiDAR → Camera Projection

**High-Level Concept:**
Project 3D LiDAR points onto 2D camera image plane using calibration matrices. Two-step transformation: extrinsic (sensor poses) then intrinsic (camera optics).

**Deep Dive:**

**Step 1 — Extrinsic Transformation:**

Transform from LiDAR frame to camera frame:
```
P_camera = R × P_lidar + t

Where:
  R: [3×3] rotation matrix (sensor angle offset)
  t: [3×1] translation vector (physical mounting offset)
  P_lidar: [3×1] point in LiDAR coordinates
```

**Step 2 — Intrinsic Projection:**

Project 3D camera coordinates to 2D pixels:
```
┌ u ┐       ┌ fx  0  cx ┐   ┌ X ┐
│ v │ = 1/Z │ 0  fy  cy │ × │ Y │
└ 1 ┘       └ 0   0   1 ┘   └ Z ┘

Where:
  fx, fy: focal lengths in pixels
  cx, cy: principal point (optical axis intersection)
  Z: depth — division by Z creates perspective
```

**Complete Code:**
```python
def project_lidar_to_camera(points, R, t, K):
    """
    Project LiDAR points onto camera image.
    
    Args:
        points: [N, 3] in LiDAR frame
        R: [3, 3] rotation matrix
        t: [3, 1] translation vector
        K: [3, 3] camera intrinsic matrix
    """
    # Step 1: Transform to camera frame
    P_cam = (R @ points.T + t).T  # [N, 3]
    
    # Filter points behind camera
    valid = P_cam[:, 2] > 0
    P_cam = P_cam[valid]
    
    # Step 2: Project to pixels
    P_norm = P_cam / P_cam[:, 2:3]  # Divide by Z
    pixels = (K @ P_norm.T).T[:, :2]  # [N, 2]
    
    return pixels, P_cam[:, 2]  # pixels and depths
```

**Why division by Z?**

This is perspective projection — objects appear smaller when far away. Same math Renaissance painters discovered. A point at Z=10m projects to a larger pixel offset than the same point at Z=100m.

**Notebook Context:** Lab 01b — Apply projection formula. See that LiDAR points don't cover every pixel — this is "sparsity." Calibration files provide R, t, K matrices.

---

## 1.8 Early Fusion

**High-Level Concept:**
Combine raw/preprocessed features at input level before the main model. Concatenate modality tensors channel-wise, feed to single encoder. Requires aligned inputs (same resolution, coordinate frame).

**Deep Dive:**

**Architecture:**
```
[RGB Image]    [3, 224, 224]
      ↓              
[Depth Map]    [1, 224, 224]
      ↓              
───── Concatenate ─────
      ↓
[Combined]     [4, 224, 224]
      ↓
CNN with 4 input channels
      ↓
   Prediction
```

**Implementation:**
```python
# Modify first conv layer
model.conv1 = nn.Conv2d(
    in_channels=4,  # Was 3 for RGB
    out_channels=64,
    kernel_size=7, stride=2, padding=3
)

# Forward pass
x = torch.cat([rgb, depth], dim=1)  # [B,4,H,W]
out = model(x)
```

**Pros:**
- Model learns cross-modal features from scratch
- Can discover subtle correlations

**Cons:**
- Requires pixel-aligned inputs
- Can't use pretrained single-modality models easily
- If one modality is noisy, contaminates everything

**When to use:** Tightly coupled sensors (RGB-D cameras)

**Notebook Context:** Lab 01a — Modify VGG first layer to accept 4 channels (RGB+Depth). Must initialize new channel weights (copy from red channel or random init).

---

## 1.9 Late Fusion

**High-Level Concept:**
Run separate models per modality, combine predictions at decision level. Each modality has independent encoder. Outputs (logits/probabilities) are combined via averaging, voting, or learned weights.

**Deep Dive:**

**Architecture:**
```
[Image] → CNN_image → logits_img [1000]
                                      ↘
                                       → Weighted Avg → Final
                                      ↗
[Audio] → CNN_audio → logits_aud [1000]
```

**Implementation:**
```python
# Separate forward passes
logits_img = model_image(image)  # [B, num_classes]
logits_aud = model_audio(audio)  # [B, num_classes]

# Late fusion options:

# 1. Simple average
logits = (logits_img + logits_aud) / 2

# 2. Learned weights
logits = w1 * logits_img + w2 * logits_aud

# 3. Concatenate + MLP
combined = torch.cat([logits_img, logits_aud], dim=1)
logits = fusion_mlp(combined)
```

**Pros:**
- Use best pretrained model for each modality
- Robust to missing modalities (just drop that branch)
- Simple, modular

**Cons:**
- No cross-modal reasoning during feature extraction
- "Cat in image" and "cat in text" don't interact until end

**When to use:** Heterogeneous models, ensembling, when modalities may be missing

**Notebook Context:** Lab 01a — Run separate image/depth models, average probability outputs. Compare accuracy to early fusion.

---

## 1.10 Intermediate Fusion

**High-Level Concept:**
Modern best practice: separate encoders per modality, fusion layer at representation level. Cross-attention allows each modality to query the other for relevant features. Best performance, complex architecture.

**Deep Dive:**

**Architecture:**
```
[Image] → ViT → Features [196, 768]
                    ↓
              Cross-Attention
                    ↑
[Text]  → BERT → Features [N, 768]
                    ↓
              Fused Features
                    ↓
               Classifier
```

**Cross-Attention Mechanism:**
```python
# Text queries image features
Q = text_features @ W_q   # Query from text
K = image_features @ W_k  # Key from image
V = image_features @ W_v  # Value from image

attention = softmax(Q @ K.T / sqrt(d)) @ V
# Output: text enriched with relevant image info
```

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
  Q: queries from one modality [N, d]
  K, V: keys/values from other modality [M, d]
  √d_k: scaling factor (prevents softmax saturation)
```

**Fusion mechanisms:**
- Concatenation + MLP
- Gated fusion (learnable weights)
- Multi-head cross-attention (best)
- Transformer encoder over concatenated tokens

**Why it wins:**
- Avoids early-noise issues (each encoder pretrained)
- Enables deep cross-modal interactions
- Attention weights are interpretable

**Notebook Context:** Lab 02a — Implement fusion network. Concatenate features from both encoders, pass through MLP. Compare to early/late — intermediate typically wins.

---

# Part 2 — Intermediate Fusion & Contrastive Pretraining

---

## 2.1 CLIP Architecture

**High-Level Concept:**
Dual-encoder model: separate image and text encoders projecting to shared 512-dim space. NO cross-attention between modalities during encoding. Fusion happens only at similarity scoring. This enables offline embedding precomputation.

**Deep Dive:**

**Architecture Diagram:**
```
┌─────────────────────────────────────────────┐
│                    CLIP                      │
│                                              │
│  [Image]                    [Text]           │
│     ↓                          ↓             │
│  ViT-L/14                  Transformer       │
│  (24 layers)               (12 layers)       │
│     ↓                          ↓             │
│  [CLS] token               [EOS] token       │
│  [1, 1024]                 [1, 512]          │
│     ↓                          ↓             │
│  Linear Proj               Linear Proj       │
│     ↓                          ↓             │
│  [1, 512]                  [1, 512]          │
│     ↓                          ↓             │
│  L2 Normalize              L2 Normalize      │
│     ↓                          ↓             │
│  img_emb ─────── dot product ────── txt_emb  │
│                      ↓                       │
│               Similarity Score               │
└─────────────────────────────────────────────┘
```

**Key Insight — NOT Intermediate Fusion:**
```
CLIP has NO cross-attention during encoding
  • Image encoder never sees text
  • Text encoder never sees image
  • They only interact via dot product at the end

This is "late fusion in embedding space"
```

**Why this design?**
```
Benefit: Offline precomputation
  • Encode all images once, store embeddings
  • New text query? Just encode query, search stored embeddings
  • No need to re-run image encoder per query

If cross-attention existed:
  • Would need to run both encoders together
  • Can't precompute — O(N×M) forward passes
```

**Training Data:**
- 400 million image-text pairs
- Scraped from internet (alt text, captions)
- WebImageText dataset

**Zero-Shot Classification:**
```python
# Classify without training on target classes
prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
text_emb = clip.encode_text(prompts)  # [3, 512]
image_emb = clip.encode_image(image)   # [1, 512]

similarities = image_emb @ text_emb.T  # [1, 3]
prediction = similarities.argmax()     # Highest similarity wins
```

**Notebook Context:** Slides reference CLIP throughout. Lab 02b loads `openai/clip-vit-base-patch32`. Understand that CLIP's power comes from scale (400M pairs) not architecture complexity.

---

## 2.2 Contrastive Training

**High-Level Concept:**
Learn embeddings by pulling matched pairs together, pushing mismatched pairs apart. Self-supervised approach using batch structure for labels. No manual annotation needed — image-text pairing IS the supervision.

**Deep Dive:**

**The Setup:**
```
Batch of N image-text pairs:
  (image_0, text_0) ← match
  (image_1, text_1) ← match
  ...
  (image_{N-1}, text_{N-1}) ← match

Positive pairs: (img_i, txt_i) for all i
Negative pairs: (img_i, txt_j) for all i≠j

With N=1000:
  • 1000 positive pairs
  • 999,000 negative pairs
```

**Why this works:**
- Model must distinguish 1 correct caption from 999 wrong ones
- Forces learning of fine-grained visual-semantic features
- Larger batch = harder task = better representations

**CLIP used batch size 32,768**
- 32K positives vs 32K×32K negatives
- This is why contrastive learning needs massive compute

**Notebook Context:** Lab 02b — Build the contrastive training loop. Key insight: labels are implicit in batch structure — no human annotation required.

---

## 2.3 Cosine Similarity vs Dot Product

**High-Level Concept:**
Cosine similarity measures angle between vectors, ignoring magnitude. Normalized dot product. Range [-1, 1]. Standard for comparing embeddings because magnitude shouldn't affect semantic similarity.

**Deep Dive:**

**Cosine Similarity Formula:**
```
cosine(A, B) = (A · B) / (‖A‖ × ‖B‖)
             = Σ(A_i × B_i) / (√Σ(A_i²) × √Σ(B_i²))

Range: [-1, 1]
  +1: Identical direction (same meaning)
   0: Orthogonal (unrelated)
  -1: Opposite directions
```

**Dot Product:**
```
dot(A, B) = Σ(A_i × B_i) = ‖A‖ × ‖B‖ × cos(θ)

Problem: Magnitude conflated with direction
  • Long vector + short vector = low dot product
  • Even if semantically similar!
```

**In Practice — Normalize First:**
```python
# After normalization, dot product = cosine similarity
A_norm = A / A.norm(dim=-1, keepdim=True)
B_norm = B / B.norm(dim=-1, keepdim=True)

similarity = A_norm @ B_norm.T  # This IS cosine sim

# Equivalent to:
similarity = F.cosine_similarity(A, B)
```

**Why CLIP uses cosine:**
- Embedding magnitude is arbitrary
- Only direction encodes semantic meaning
- Normalized = focus on what matters

**Notebook Context:** Used everywhere: contrastive loss, retrieval, CLIP inference, vector DB search. Always normalize embeddings before similarity.

---

## 2.4 Ground Truth Labels for Contrastive Training

**High-Level Concept:**
For contrastive training, labels = indices because `image_i` matches `text_i` by construction. Self-supervised — batch structure provides supervision without manual labels.

**Deep Dive:**

**The Key Insight:**
```
Batch construction:
  (image_0, text_0)  ← Pair 0
  (image_1, text_1)  ← Pair 1
  ...
  (image_N, text_N)  ← Pair N

Question for image_0: "Which text matches?"
Answer: text_0 (index 0)

Question for image_1: "Which text matches?"
Answer: text_1 (index 1)

Therefore: labels = [0, 1, 2, ..., N-1]
         = torch.arange(N)
```

**Why this is elegant:**
- No human annotation required!
- The pairing structure IS the label
- Every batch is self-labeled
- Scales to 400M pairs without manual work

**Code:**
```python
batch_size = len(images)
labels = torch.arange(batch_size, device=device)

# labels[i] = i means:
# "The correct text for image_i is text_i"
```

**Notebook Context:** Lab 02b — `labels = torch.arange(batch_size)`. Common student confusion — clarify that labels aren't class IDs, they're pair indices.

---

## 2.5 Contrastive Loss (InfoNCE)

**High-Level Concept:**
Cross-entropy loss over similarity matrix. Symmetric: image→text + text→image. Temperature scaling controls distribution sharpness.

**Deep Dive:**

**Step-by-Step Computation:**
```python
# Step 1: Encode and normalize
img_emb = image_encoder(images)  # [N, 512]
txt_emb = text_encoder(texts)     # [N, 512]
img_emb = F.normalize(img_emb, dim=-1)
txt_emb = F.normalize(txt_emb, dim=-1)

# Step 2: Compute similarity matrix
logits = img_emb @ txt_emb.T  # [N, N]
# logits[i,j] = similarity(image_i, text_j)

# Step 3: Apply temperature (learnable)
logits = logits * temperature  # or / temperature

# Step 4: Labels = diagonal indices
labels = torch.arange(N, device=device)

# Step 5: Cross-entropy both directions
loss_i2t = F.cross_entropy(logits, labels)    # rows
loss_t2i = F.cross_entropy(logits.T, labels)  # cols
loss = (loss_i2t + loss_t2i) / 2
```

**What cross-entropy does here:**
```
For image_0:
  logits[0] = [sim(i0,t0), sim(i0,t1), sim(i0,t2), ...]
  labels[0] = 0

CE loss pushes sim(i0,t0) to be highest
and all sim(i0,t_j≠0) to be low
```

**Temperature Scaling:**
```
τ (temperature) controls sharpness:

logits/τ with τ=0.07 (CLIP default):
  • Divides by small number → larger values
  • Sharper softmax → more confident

τ=1.0: Softer distribution
τ=0.01: Very peaked, nearly argmax

CLIP learns τ as a parameter (log-scale)
```

**Notebook Context:** Lab 02b — Implement exact loss function. Temperature is crucial — too high loses signal, too low causes training instability.

---

## 2.6 `repeat_interleave` vs `repeat`

**High-Level Concept:**
Tensor duplication for batch alignment. `repeat_interleave` = element-wise repetition (stutter). `repeat` = tile entire tensor (echo). Mixing them scrambles labels.

**Deep Dive:**

**Visual Difference:**
```
x = torch.tensor([A, B, C])

x.repeat_interleave(3):
→ [A, A, A, B, B, B, C, C, C]
Think: "stutter" — each element repeats before moving on

x.repeat(3):
→ [A, B, C, A, B, C, A, B, C]
Think: "echo" — whole sequence repeats
```

**When Each Is Used:**
```python
# Scenario: 4 images, each needs to match 3 texts
images = torch.randn(4, 512)  # [I0, I1, I2, I3]
texts = torch.randn(3, 512)   # [T0, T1, T2]

# Want: Every image paired with every text
# (I0,T0), (I0,T1), (I0,T2), (I1,T0), ...

# Images: repeat each 3 times
img_exp = images.repeat_interleave(3, dim=0)  # [12, 512]
# [I0,I0,I0, I1,I1,I1, I2,I2,I2, I3,I3,I3]

# Texts: tile 4 times
txt_exp = texts.repeat(4, 1)  # [12, 512]
# [T0,T1,T2, T0,T1,T2, T0,T1,T2, T0,T1,T2]

# Now img_exp[i] pairs with txt_exp[i] correctly
```

**Common Bug:**
```python
# WRONG — using repeat for images
img_exp = images.repeat(3, 1)
# [I0,I1,I2,I3, I0,I1,I2,I3, I0,I1,I2,I3]
# Now I0 pairs with T0, I1 pairs with T1... WRONG alignment!
```

**Notebook Context:** Lab 02b — Used during batch preparation. Wrong operation = scrambled image-text pairs = model learns nothing.

---

## 2.7 Vector Database

**High-Level Concept:**
Storage system for embeddings with fast similarity search. HNSW (Hierarchical Navigable Small World) enables O(log N) approximate nearest neighbor search vs O(N) brute force.

**Deep Dive:**

**The Problem:**
```
1 million document chunks, each [768] dims
Query: find 10 most similar to query vector

Brute force: 1M dot products per query
At 1μs each: 1 second per query
At 1000 QPS: 1000 GPUs needed
```

**HNSW Solution:**
```
Build multi-layer graph during indexing:

Layer 3:  A ───────────────────── B  (sparse)
Layer 2:  A ────── C ────── B
Layer 1:  A ── D ── C ── E ── B
Layer 0:  A─D─F─G─C─H─I─E─J─B       (dense)

Search:
1. Start at Layer 3 (few nodes, coarse)
2. Greedy move toward query
3. Drop to Layer 2, continue
4. ... until Layer 0
5. Local refinement in dense neighborhood

Complexity: O(log N) vs O(N)
```

**Parameters:**
```
M: edges per node (more = accurate, more memory)
ef_construction: depth during build (more = better index)
ef_search: depth during query (speed vs accuracy tradeoff)

Typical: M=16, ef_construction=256, ef_search=128
```

**Vector DB Options:**
- **Milvus** — open source, distributed
- **Pinecone** — managed service
- **Weaviate** — hybrid search
- **Qdrant** — Rust, fast
- **pgvector** — Postgres extension

**Notebook Context:** Lab 04a — Use Milvus. Insert frame embeddings, query with text. DB handles similarity search internally.

---

## 2.8 RAG (Retrieval Augmented Generation)

**High-Level Concept:**
LLM retrieves external context before generating answer. Reduces hallucination by grounding responses in retrieved documents. Enables fresh/private knowledge without retraining.

**Deep Dive:**

**Pipeline Diagram:**
```
┌──────────── INDEXING (Offline) ────────────┐
│                                            │
│ Documents → Chunk → Embed → Vector DB      │
│    ↓         ↓       ↓         ↓           │
│  PDFs    500 chars  BGE     Milvus         │
│  HTML    per chunk  model   stored         │
└────────────────────────────────────────────┘

┌──────────── RETRIEVAL (Online) ────────────┐
│                                            │
│ Query → Embed → Vector Search → Top-K      │
│   ↓       ↓          ↓            ↓        │
│ "What   [768]     HNSW ANN    [chunk1,     │
│  is X?" vector    search       chunk2,     │
│                                chunk3]     │
└────────────────────────────────────────────┘

┌──────────── GENERATION ────────────────────┐
│                                            │
│ Augmented Prompt → LLM → Grounded Answer   │
│        ↓            ↓          ↓           │
│ "Context:        GPT-4     Based on        │
│  {chunks}        LLaMA     retrieved       │
│  Question:                 context...      │
│  {query}"                                  │
└────────────────────────────────────────────┘
```

**Why RAG Helps:**
```
Without RAG:
  • LLM only knows training data (cutoff date)
  • Hallucinates when uncertain
  • Can't cite sources
  • No private/enterprise knowledge

With RAG:
  • Fresh knowledge from your documents
  • Grounded in retrieved context
  • Can cite specific chunks
  • Works on proprietary data
```

**Prompt Template:**
```
System: Answer based only on the context below.

Context:
{chunk_1}
{chunk_2}
{chunk_3}

Question: {user_query}

Answer:
```

**Notebook Context:** Lab 03b — Build RAG pipeline. Extract PDF text, chunk, embed, store, retrieve, augment prompt. Core pattern for document AI.

---

# Part 3 — Cross-Modal Projection, VLMs, LLaVA & OCR Pipelines

---

## 3.1 Vision Language Models (VLMs)

**High-Level Concept:**
Models accepting both image + text, generating text with visual grounding. Core insight: treat projected image features as "visual tokens" that LLM processes alongside text tokens.

**Deep Dive:**

**Architecture:**
```
┌─────────────────────────────────────────┐
│              VLM Structure               │
│                                          │
│  [Image] → Vision Encoder → Projection   │
│               (CLIP ViT)     (MLP)       │
│                   ↓            ↓         │
│              [257, 1024] → [256, 4096]   │
│                              ↓           │
│                        Visual Tokens     │
│                              ↓           │
│  [Text] → Tokenize → Embed → Text Tokens │
│                              ↓           │
│      [Visual Tokens] + [Text Tokens]     │
│                              ↓           │
│                LLM (LLaMA/Vicuna)         │
│                              ↓           │
│               Generated Response          │
└─────────────────────────────────────────┘
```

**The Key Insight:**
```
LLMs process token sequences.
Images aren't tokens.

Solution: Make image features "look like" tokens
  • Project to same dimension as LLM embeddings
  • LLM treats them as "foreign language" tokens
  • Attention over [image tokens, text tokens] jointly
```

**Input Sequence:**
```
[IMG_1, IMG_2, ..., IMG_256, USER, :, W, h, a, t, ...]
│←── 256 visual tokens ──→│←── text tokens ──→│
```

**Notebook Context:** Lab 03a — Build VLM components. Understand that VLM = Vision Encoder + Projection + LLM glued together.

---

## 3.2 Cross-Modal Projection

**High-Level Concept:**
Learned transformation mapping vision embeddings to LLM embedding space. Dimension alignment is necessary but not sufficient — semantic alignment requires learning.

**Deep Dive:**

**The Problem:**
```
Vision encoder output: [batch, 256, 1024]
LLM embedding space:   [batch, *, 4096]

Issue 1 — Dimension mismatch:
  1024 ≠ 4096

Issue 2 — Semantic mismatch:
  Vision features encode visual patterns
  LLM expects "word-like" meanings
  Feature distributions completely different
```

**Solution — Learned Projection:**
```python
class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        
        # Simple linear (works but limited)
        # self.proj = nn.Linear(vision_dim, llm_dim)
        
        # 2-layer MLP (LLaVA uses this)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, vision_features):
        # [B, num_patches, vision_dim] → [B, num_patches, llm_dim]
        return self.proj(vision_features)
```

**Why MLP > Linear:**
```
Linear transformation: y = Wx + b
  • Can only rotate, scale, shear, translate
  • Preserves straight lines and ratios

But vision and language spaces have different topology:
  • "Dog" and "cat" close in language space
  • Dog and cat IMAGES might be far in vision space

MLP with nonlinearity can WARP the space:
  • Bend, fold, stretch
  • Map different topologies onto each other
```

**Training the Projector:**
```
Loss: MSE or contrastive between:
  • Projected image features
  • Target text embeddings

Goal: proj(vision_feature_of_dog) ≈ llm_embedding("dog")
```

**Notebook Context:** Lab 03a — Train projection layer. Minimize distance between projected visual features and corresponding text embeddings.

---

## 3.3 LLaVA Architecture

**High-Level Concept:**
Reference VLM: CLIP vision + MLP projection + Vicuna LLM. Two-stage training separates feature alignment (projector only) from instruction tuning (full stack).

**Deep Dive:**

**Complete Architecture:**
```
┌─────────────────────────────────────────┐
│               LLaVA                      │
│                                          │
│  ┌─────────────────┐                     │
│  │ Vision Encoder  │ ← CLIP ViT-L/14     │
│  │    (FROZEN)     │    pretrained       │
│  └────────┬────────┘                     │
│           │ [257, 1024]                  │
│           ▼                              │
│  ┌─────────────────┐                     │
│  │   Projection    │ ← 2-layer MLP       │
│  │   (TRAINED)     │    768K params      │
│  └────────┬────────┘                     │
│           │ [256, 4096]                  │
│           ▼                              │
│      Visual Tokens                       │
│           +                              │
│      Text Tokens ← from user prompt      │
│           │                              │
│           ▼                              │
│  ┌─────────────────┐                     │
│  │      LLM        │ ← Vicuna-7B/13B     │
│  │ (FROZEN→LoRA)   │    (LLaMA-tuned)    │
│  └────────┬────────┘                     │
│           │                              │
│           ▼                              │
│   Generated Response                     │
└─────────────────────────────────────────┘
```

**Two-Stage Training:**
```
Stage 1: Feature Alignment (Pretraining)
─────────────────────────────────────────
Vision encoder: FROZEN
Projection:     TRAINED ← only this
LLM:            FROZEN

Data: 558K image-caption pairs (CC3M filtered)
Goal: Learn to "speak" LLM's language
Compute: ~4 hours on 8 A100s


Stage 2: Visual Instruction Tuning
─────────────────────────────────────────
Vision encoder: FROZEN
Projection:     FINE-TUNED
LLM:            FINE-TUNED (or LoRA)

Data: 158K instruction-following conversations
      Generated by GPT-4 from images
Goal: Learn to follow visual instructions
```

**Why Two Stages?**
```
Stage 1 alone: Can describe images
              But doesn't follow instructions well

Stage 2 alone: Expensive, unstable
              Projector not initialized

Both stages: Projector learns basics first
            Then full system fine-tuned
            Efficient + stable + high quality
```

**Notebook Context:** Slides detail LLaVA. Lab 03a replicates architecture at smaller scale. Key insight: frozen pretrained components + small trainable connector = efficient VLM.

---

## 3.4 PDF Document Chunking

**High-Level Concept:**
Breaking documents into retrieval-friendly pieces. Chunking strategy dramatically affects RAG quality. Too big = diluted relevance. Too small = lost context.

**Deep Dive:**

**1. Fixed-Size Chunking:**
```python
def fixed_chunk(text, size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks

# Problem: May split mid-sentence, mid-table
# "The CEO stated that profitability" | "will increase by 40%"
```

**2. Recursive/Hierarchical:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)

# Tries paragraph → sentence → word → char
# Uses largest unit that fits size limit
```

**3. Semantic Chunking:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = text.split('. ')
embeddings = model.encode(sentences)

# Compute similarity between adjacent sentences
similarities = [cosine(emb[i], emb[i+1]) 
                for i in range(len(emb)-1)]

# Split where similarity drops (topic change)
breakpoints = [i for i, sim in enumerate(similarities) 
               if sim < threshold]
```

**4. Layout-Aware:**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("doc.pdf")

# Keep structure intact:
# - Tables as complete units
# - Headers + following paragraphs
# - Lists as complete items
# - Figures + captions together
```

**Chunk Size Guidelines:**
```
Embedding model context: 512 tokens typical
LLM context for RAG: 2048-8192 tokens

Sweet spot: 200-500 tokens per chunk
  • Large enough for context
  • Small enough for focused retrieval
  • With 100-token overlap
```

**Notebook Context:** Lab 03b — Use `unstructured` library. Compare chunking strategies. Layout-aware >> fixed-size for structured docs.

---

## 3.5 Identifying Page Elements

**High-Level Concept:**
Detect and classify document regions: text, tables, figures, headers. Layout analysis + OCR. Object detection models (YOLOX, Detectron2) trained on document layouts.

**Deep Dive:**

**Element Types:**
```
• Title           — Document/section headers
• NarrativeText   — Body paragraphs
• ListItem        — Bulleted/numbered items
• Table           — Structured tabular data
• Figure          — Images, charts, diagrams
• Caption         — Figure/table descriptions
• Header/Footer   — Page metadata
• PageNumber      — Pagination
• Formula         — Mathematical equations
```

**Detection Pipeline:**
```
PDF/Image
    ↓
Layout Model (YOLOX/LayoutLM)
    ↓
Bounding boxes + element types
    ↓
OCR per region (Tesseract/DocTR)
    ↓
Structured output with positions
```

**Using Unstructured Library:**
```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    "document.pdf",
    strategy="hi_res",  # Uses vision model
    infer_table_structure=True
)

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text[:100]}...")
    
# Output:
# Type: Title
# Text: Annual Report 2024...
# Type: NarrativeText  
# Text: Our company achieved record growth...
# Type: Table
# Text: |Revenue|Growth|....
```

**Table Handling:**
```python
tables = [e for e in elements if type(e).__name__ == 'Table']

for table in tables:
    # Option 1: Keep as markdown
    md = table.metadata.text_as_html
    
    # Option 2: Convert to DataFrame
    df = pd.read_html(table.metadata.text_as_html)[0]
    
    # Option 3: Describe in natural language
    description = f"Table showing {table.text[:50]}..."
```

**Tools:**
- `unstructured` — Python library, free
- LayoutLM/LayoutLMv3 — Microsoft, SOTA
- DocTR — OCR + layout
- NVIDIA DALI — GPU-accelerated

**Notebook Context:** Lab 03b — `unstructured` automatically tags elements. Filter by type for specialized processing.

---

# Part 4 — VSS, Knowledge Graphs, Vector RAG & Context-Aware RAG

---

## 4.1 VSS Applications

**High-Level Concept:**
Video Search & Summarization: query video archives with natural language, get timestamped results and summaries. Surveillance, media, sports, compliance, education — any domain with video at scale.

**Deep Dive:**

**Use Cases:**

**Surveillance/Safety:**
- "Show all PPE violations this week"
- "Find vehicles entering after hours"
- "Locate the red truck from yesterday"

**Media Production:**
- "Find clips of CEO mentioning revenue"
- "Locate all B-roll of city skylines"
- "Find interviews about product launch"

**Sports Analytics:**
- "Show all 3-point shots in Q4"
- "Find defensive formations against zone"
- "Locate player injuries this season"

**Education:**
- "Find lecture segments on neural networks"
- "Locate demonstrations of titration"

**Compliance:**
- "Find conversations mentioning pricing"
- "Locate policy violation instances"

**Value Proposition:**
```
Without VSS:
  • Human watches 100 hours of footage
  • Days of work, expensive, error-prone

With VSS:
  • Query in seconds
  • Timestamps + summaries returned
  • Human reviews only relevant clips
```

**Scale:**
- Security: 1000s of cameras × 24/7 = petabytes
- Media: millions of clips, growing daily
- Manual review impossible

**Notebook Context:** Lab 04a — Query traffic video database. Experience how VSS saves hours of manual review.

---

## 4.2 NVIDIA AI Blueprint for VSS

**High-Level Concept:**
End-to-end reference architecture integrating CV, VLM, LLM, RAG, and databases. NIMs (NVIDIA Inference Microservices) for each component, orchestrated via API.

**Deep Dive:**

**Architecture Diagram:**
```
┌──────────────────────────────────────────────┐
│       NVIDIA AI BLUEPRINT FOR VSS            │
│                                              │
│  Video Files                                 │
│      ↓                                       │
│  ┌────────────────────────────────┐          │
│  │        DATA PLANE              │          │
│  │  • DeepStream SDK (decode)     │          │
│  │  • Chunk videos (10-60s)       │          │
│  │  • Extract keyframes           │          │
│  └────────────┬───────────────────┘          │
│               ↓                              │
│  ┌────────────────────────────────┐          │
│  │        CV NIMs                 │          │
│  │  • Object detection            │          │
│  │  • Tracking (ID persistence)   │          │
│  │  • Segmentation                │          │
│  └────────────┬───────────────────┘          │
│               ↓                              │
│  ┌────────────────────────────────┐          │
│  │        VLM NIMs                │          │
│  │  • Frame captioning            │          │
│  │  • CLIP embedding              │          │
│  │  • OCR (text in frames)        │          │
│  └────────────┬───────────────────┘          │
│               ↓                              │
│       ┌───────┴───────┐                      │
│       ↓               ↓                      │
│  ┌──────────┐   ┌──────────┐                 │
│  │Vector DB │   │ Graph DB │                 │
│  │ (Milvus) │   │ (Neo4j)  │                 │
│  └─────┬────┘   └────┬─────┘                 │
│        └──────┬──────┘                       │
│               ↓                              │
│  ┌────────────────────────────────┐          │
│  │   NeMo Retriever (RAG NIM)     │          │
│  │  • Hybrid retrieval            │          │
│  │  • Re-ranking                  │          │
│  └────────────┬───────────────────┘          │
│               ↓                              │
│  ┌────────────────────────────────┐          │
│  │        LLM NIM                 │          │
│  │  • Answer generation           │          │
│  │  • Summarization               │          │
│  └────────────┬───────────────────┘          │
│               ↓                              │
│       Natural Language Response              │
│       + Timestamps + Video Clips             │
└──────────────────────────────────────────────┘
```

**NIM = NVIDIA Inference Microservice:**
- Containerized, optimized model serving
- Pre-built for common tasks
- GPU-accelerated inference
- REST API interface

**NeMo Retriever:**
- Ingestion: chunk + embed + index
- Retrieval: query + search + re-rank
- Supports vector, graph, hybrid

**Notebook Context:** Lab 04a — Interact with Blueprint API endpoints. Abstracts complexity of 5+ model pipeline into simple queries.

---

## 4.3 VSS Architecture (Implementation)

**High-Level Concept:**
Technical implementation: video → keyframes → embeddings → storage → retrieval → summarization. Each stage has design choices affecting latency, quality, cost.

**Deep Dive:**

**Processing Pipeline:**
```python
# Stage 1: Video Ingestion
def ingest_video(video_path, chunk_duration=30):
    chunks = []
    for start in range(0, video_length, chunk_duration):
        chunk = extract_frames(video, start, chunk_duration)
        chunks.append({
            'start': start,
            'end': start + chunk_duration,
            'frames': chunk
        })
    return chunks

# Stage 2: Frame Embedding
def embed_chunk(chunk, clip_model):
    keyframes = sample_keyframes(chunk['frames'], n=5)
    embeddings = clip_model.encode_image(keyframes)
    chunk_embedding = embeddings.mean(dim=0)
    return chunk_embedding

# Stage 3: Caption Generation
def caption_chunk(chunk, vlm_model):
    keyframe = chunk['frames'][len(chunk['frames'])//2]
    caption = vlm_model.generate(
        image=keyframe,
        prompt="Describe this scene in detail:"
    )
    return caption

# Stage 4: Indexing
def index_chunk(embedding, metadata, vector_db):
    vector_db.insert(
        embedding=embedding,
        metadata={
            'video_id': metadata['video_id'],
            'start_time': metadata['start'],
            'caption': metadata['caption']
        }
    )

# Stage 5: Retrieval
def search(query, clip_model, vector_db, top_k=10):
    query_embedding = clip_model.encode_text(query)
    results = vector_db.search(query_embedding, top_k=top_k)
    return results

# Stage 6: Summarization
def summarize(query, results, llm):
    context = "\n".join([r['caption'] for r in results])
    prompt = f"Based on video descriptions:\n{context}\n\nAnswer: {query}"
    return llm.generate(prompt)
```

**Design Choices:**
```
Chunk duration:
  • 10s: Fine-grained, more storage, precise retrieval
  • 60s: Coarse, less storage, may miss details
  • Sweet spot: 15-30s for most applications

Keyframe sampling:
  • 1 per chunk: Fast, may miss events
  • 5 per chunk: Balanced
  • Every frame: Expensive, diminishing returns

Embedding strategy:
  • Average frames: Simple, loses temporal info
  • Concatenate: Preserves order, larger vectors
  • Temporal pooling: Best, complex
```

**Notebook Context:** Lab 04a — Implement retrieval + summarization loop.

---

## 4.4 Vector RAG

**High-Level Concept:**
Semantic retrieval using embedding similarity. Find chunks with similar meaning to query, regardless of exact keyword match. Core of modern search.

**Deep Dive:**

**How It Works:**
```
Query: "What safety incidents occurred?"
          ↓
    Embed query → [768] vector
          ↓
    Search vector DB (HNSW)
          ↓
    Return top-k similar chunks

Chunks returned:
  • "Worker slipped near forklift" (sim: 0.82)
  • "Near-miss with loading equipment" (sim: 0.79)
  • "PPE violation at dock 3" (sim: 0.76)
```

**Advantages:**
- Semantic matching ("safety incidents" ↔ "near-miss")
- No keyword engineering required
- Works across paraphrases and synonyms
- Language-agnostic with multilingual embeddings

**Limitations:**
```
• May miss exact keyword matches
  Query: "Show clip 47"
  Vector search might find similar clips, not exact

• No relational reasoning
  Query: "Who talked to the person in the red shirt?"
  Requires entity tracking, not just similarity

• No temporal reasoning
  Query: "What happened AFTER the alarm?"
  Similarity doesn't understand "after"
```

**Implementation:**
```python
def vector_rag(query, embedding_model, vector_db, llm):
    # 1. Embed query
    query_vec = embedding_model.encode(query)
    
    # 2. Search
    results = vector_db.search(query_vec, top_k=5)
    
    # 3. Build context
    context = "\n\n".join([r.text for r in results])
    
    # 4. Generate
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = llm.generate(prompt)
    
    return answer, results
```

**Notebook Context:** Lab 04a, 04b — Standard vector search. Fast, effective for semantic queries.

---

## 4.5 Graph RAG

**High-Level Concept:**
Retrieval using knowledge graph relationships. Enables multi-hop reasoning, entity tracking, relational queries that vector search can't handle.

**Deep Dive:**

**Why Graphs?**
```
Query: "Which companies has the CEO of Acme worked for?"

Vector search finds:
  • Chunks mentioning CEO
  • Chunks mentioning Acme
  • Chunks mentioning companies
BUT: Can't chain the relationships

Graph query:
  MATCH (c:Company {name:"Acme"})<-[:CEO_OF]-(p:Person)
  MATCH (p)-[:WORKED_AT]->(prev:Company)
  RETURN prev.name

→ Follows relationship chain directly
```

**Knowledge Graph Structure:**
```
Nodes (Entities):
  • Person: {name, role, ...}
  • Company: {name, industry, ...}
  • Location: {name, type, ...}
  • Event: {type, timestamp, ...}

Edges (Relationships):
  • (:Person)-[:WORKS_AT]->(:Company)
  • (:Person)-[:LOCATED_IN]->(:Location)
  • (:Event)-[:INVOLVES]->(:Person)
  • (:Video)-[:SHOWS]->(:Entity)
```

**Building the Graph:**
```python
# From video captions/analysis
caption = "John Smith enters warehouse at 2pm"

# Extract entities
entities = ner_model(caption)
# [("John Smith", "Person"), ("warehouse", "Location")]

# Extract relationships
relations = relation_extractor(caption)
# [("John Smith", "enters", "warehouse")]

# Create graph nodes/edges
graph.create_node("Person", name="John Smith")
graph.create_node("Location", name="warehouse")
graph.create_edge("John Smith", "ENTERS", "warehouse", timestamp="2pm")
```

**Hybrid Retrieval:**
1. Parse query for entities/relationships
2. Graph query for structured info
3. Vector search for semantic context
4. Merge results
5. Generate answer with full context

**Notebook Context:** Lab 04b — Use Neo4j. Visualize entity graphs. See how graph queries answer relational questions.

---

## 4.6 Context-Aware RAG

**High-Level Concept:**
Filtered and enriched retrieval. Pre-filter by metadata (time, location, source) before vector search. Post-process to add surrounding context.

**Deep Dive:**

**Why Context Matters:**
```
Query: "Show safety incidents from Camera 3 yesterday"

Pure vector search:
  • Returns safety incidents from any camera, any time
  • Irrelevant results dilute context

Context-aware search:
  • Filter: camera=3, date=yesterday
  • Then: vector search within filtered set
  • Result: Precise, relevant hits
```

**Implementation:**
```python
def context_aware_rag(query, vector_db, llm):
    # 1. Extract metadata filters from query
    filters = extract_filters(query)  # LLM or rules
    # filters = {"camera": "3", "date": "2024-01-15"}
    
    # 2. Apply filters BEFORE vector search
    filtered_results = vector_db.search(
        query_embedding=embed(query),
        filter=filters,
        top_k=10
    )
    
    # 3. Expand context (adjacent segments)
    expanded = []
    for result in filtered_results:
        prev = get_chunk(result.id - 1)
        next = get_chunk(result.id + 1)
        expanded.append({
            'before': prev,
            'match': result,
            'after': next
        })
    
    # 4. Re-rank by relevance
    reranked = rerank_model(query, expanded)
    
    # 5. Generate with rich context
    context = format_context(reranked[:5])
    return llm.generate(f"Context:\n{context}\n\nQ: {query}")
```

**Filter Types:**
```
Temporal:
  • date = "2024-01-15"
  • time_range = ["09:00", "17:00"]
  • relative = "last 7 days"

Spatial:
  • camera_id = "CAM_03"
  • location = "warehouse"
  • zone = "loading_dock"

Categorical:
  • event_type = "safety_incident"
  • severity = "high"
```

**Router Pattern:**
```python
def extract_filters(query):
    prompt = f"""Extract search filters from this query:
    Query: {query}
    
    Return JSON with: camera, date, time, event_type
    If not specified, use null."""
    
    return json.loads(llm.generate(prompt))
```

**Notebook Context:** Lab 04a — Pass `filters` argument to search API. Critical for precision in large archives.

---

## 4.7 Cypher Query Language

**High-Level Concept:**
Declarative graph query language for Neo4j. Pattern matching syntax for traversing relationships. Essential for Graph RAG.

**Deep Dive:**

**Basic Syntax:**
```cypher
// Nodes in parentheses
(n:Label {property: value})

// Relationships in brackets
-[r:RELATIONSHIP_TYPE]->

// Full pattern
MATCH (person:Person)-[r:WORKS_AT]->(company:Company)
WHERE company.name = "Acme"
RETURN person.name, r.since
```

**Core Commands:**
```cypher
// CREATE - Add nodes/edges
CREATE (p:Person {name: "John", role: "engineer"})

// MATCH - Find patterns
MATCH (p:Person {name: "John"})
RETURN p

// MERGE - Create if not exists
MERGE (c:Company {name: "Acme"})

// WHERE - Filter results
MATCH (p:Person)
WHERE p.age > 30
RETURN p

// Relationships
MATCH (a)-[:KNOWS]->(b)   // Directed
MATCH (a)-[:KNOWS]-(b)    // Either direction
```

**Multi-Hop Queries:**
```cypher
// 2-hop: Friends of friends
MATCH (p:Person {name: "Alice"})-[:KNOWS]->()-[:KNOWS]->(fof)
WHERE fof <> p
RETURN DISTINCT fof.name

// Variable length: 1-3 hops
MATCH (start)-[:CONNECTED*1..3]-(end)
RETURN end

// Shortest path
MATCH path = shortestPath(
  (a:Person {name: "Alice"})-[*]-(b:Person {name: "Bob"})
)
RETURN path
```

**Aggregation:**
```cypher
// Count relationships
MATCH (p:Person)-[:WORKED_AT]->(c:Company)
RETURN p.name, count(c) as num_companies
ORDER BY num_companies DESC

// Collect into list
MATCH (p:Person)-[:KNOWS]->(friend)
RETURN p.name, collect(friend.name) as friends
```

**LLM-Generated Cypher (G-Retriever):**
```python
def text_to_cypher(natural_query, llm):
    prompt = f"""Convert to Cypher query:
    
    Schema:
    (:Person)-[:WORKS_AT]->(:Company)
    (:Person)-[:LOCATED_IN]->(:Location)
    
    Question: {natural_query}
    
    Cypher:"""
    
    return llm.generate(prompt)

# "Who works at Acme?" →
# MATCH (p:Person)-[:WORKS_AT]->(c:Company {name:"Acme"})
# RETURN p.name
```

**Notebook Context:** Lab 04b — Execute Cypher queries against Neo4j. Understand pattern syntax.

---

# Certification Checklist — Quick Self-Test

| Topic | Can You Explain? | Can You Code/Diagram? |
|-------|-----------------|----------------------|
| CNN architecture | Convolution, pooling, receptive field | Draw layer stack |
| PyTorch basics | Module, forward, backward, DataLoader | Write training loop |
| Vision data shapes | RGB, grayscale, CT, point cloud | Reshape tensors |
| Audio spectrograms | STFT, mel scale, why vision models work | `librosa` pipeline |
| Color modes | RGB vs BGR, common bugs | `cv2.cvtColor()` |
| LiDAR projection | Extrinsic R,t → Intrinsic K | Projection code |
| Fusion types | Early/Late/Intermediate trade-offs | Architecture diagrams |
| CLIP architecture | Dual encoder, NO cross-attention | Full diagram |
| Contrastive training | Positive/negative pairs, batch supervision | Loss computation |
| Cosine vs dot product | When to normalize | Vector math |
| Ground truth labels | Why `torch.arange(N)` | Code snippet |
| InfoNCE loss | Symmetric CE, temperature | Full implementation |
| `repeat_interleave` vs `repeat` | Stutter vs echo | Alignment examples |
| VLM architecture | Vision → projection → LLM | Diagram with dims |
| Cross-modal projection | Why MLP > Linear | Projection layer code |
| LLaVA | Two-stage training | Training diagram |
| PDF chunking | Fixed, recursive, semantic, layout | Comparison + code |
| Page elements | Types, detection pipeline | `unstructured` usage |
| Vector database | HNSW, ANN, parameters | Index + search code |
| RAG pipeline | Index → retrieve → augment → generate | Full diagram |
| Vector vs Graph RAG | When each excels | Query examples |
| Context-aware RAG | Metadata filtering, re-ranking | Filter extraction |
| Cypher queries | Pattern syntax, multi-hop | Write queries |
| VSS applications | Use cases | Domain examples |
| NVIDIA Blueprint | Components, NIMs, data flow | Architecture diagram |

---

*End of Master Certification Table*
