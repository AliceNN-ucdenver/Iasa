# Building AI Agents with Multimodal Models
## NVIDIA DLI Instructor Certification — Deep Dive Study Guide

---

# The Big Picture: What Problem Are We Solving?

Before diving into individual topics, understand the **core challenge**: Different sensors capture different slices of reality. A camera sees color and texture but has no idea how far away anything is. LiDAR knows precise distances but can't tell a red car from a blue one. Text describes concepts but can't show you what something looks like.

**Multimodal AI is about building systems that see reality more completely by combining these partial views.**

The entire course follows this progression:

```
Part 1: How do we REPRESENT different modalities? (Data formats, shapes)
Part 2: How do we ALIGN them into shared spaces? (CLIP, contrastive learning)
Part 3: How do we make LLMs UNDERSTAND visual input? (VLMs, projection)
Part 4: How do we RETRIEVE and REASON over multimodal data? (RAG, graphs)
```

---

# Part 1: Multimodal Data Representations

## 1.1 The Shape of Vision Data

### Why Shape Matters

Every neural network layer expects a specific tensor shape. Get it wrong and you get cryptic dimension mismatch errors. Get it right and the model "sees" your data correctly.

**The Universal Vision Convention (PyTorch):** `[Batch, Channels, Height, Width]` or `[B, C, H, W]`

### RGB Images

```
Shape: [B, 3, H, W]
       └── 3 channels: Red, Green, Blue
       
Example: A batch of 32 images at 224×224
         [32, 3, 224, 224]
         
Memory: 32 × 3 × 224 × 224 × 4 bytes = ~19 MB (float32)
```

**The values**: Each pixel has 3 intensity values (0-255 raw, or 0.0-1.0 normalized). These aren't arbitrary — they map to the actual wavelengths of light the camera sensor captured.

**Why 224×224?** ImageNet trained models at this resolution. It's not magic — just a convention that stuck. Modern models use 336, 384, or even 1024.

### Grayscale Images

```
Shape: [B, 1, H, W]
       └── 1 channel: Intensity only

Use cases:
- Medical imaging (X-rays care about density, not color)
- Document OCR (text is black on white)
- Depth maps (distance encoded as brightness)
```

**Conversion from RGB:**
```python
# Luminosity formula (how humans perceive brightness)
gray = 0.2126 * R + 0.7152 * G + 0.0722 * B

# Why these weights? 
# Human eyes have more green-sensitive cones than red or blue.
# This formula matches perceived brightness, not raw average.
```

### RGBA (Transparency)

```
Shape: [B, 4, H, W]
       └── 4th channel: Alpha (0=transparent, 1=opaque)

Watch out: Most vision models expect 3 channels.
           Loading a PNG with transparency? Drop or composite the alpha.
```

### CT Scans (Volumetric Data)

```
Shape: [B, 1, D, H, W]
       └── D = Depth (number of slices through the body)
       └── NOT time — this is spatial depth

Typical dimensions: [1, 1, 200, 512, 512]
                    200 slices, each 512×512
```

**Critical insight**: You can't use `Conv2d` on CT data — you need `Conv3d` because the depth dimension contains spatial information (the slice above and below a tumor are medically relevant).

**Hounsfield Units (HU)**: CT intensity values aren't 0-255. They're calibrated to tissue density:
- Air: -1000 HU
- Water: 0 HU  
- Bone: +1000 HU
- Metal implants: +3000 HU

**Windowing**: To visualize specific tissues, you map an HU range to display range:
```python
# Lung window: [-1000, -200] HU → [0, 255]
# Bone window: [-500, +1500] HU → [0, 255]
# Same scan, different windows reveal different anatomy
```

---

## 1.2 LiDAR Data and Camera Fusion (Deep Dive)

This is where most explanations fail. Let me walk you through what's actually happening.

### What LiDAR Actually Captures

A LiDAR sensor fires laser pulses in known directions and measures how long until each pulse returns. From time-of-flight, it calculates distance.

```
Raw LiDAR output: Point Cloud
Shape: [N, 4] where N = number of points (typically 100k-1M)

Each point: [x, y, z, intensity]
            └── x, y, z: 3D position in meters, relative to sensor
            └── intensity: How much laser light reflected back
```

**Coordinate system matters**: LiDAR typically uses:
- X: forward (direction car is moving)
- Y: left
- Z: up

### What the Camera Captures

```
Camera output: Dense 2D image
Shape: [3, H, W]

Every pixel has color, but NO depth information.
You can see a red car, but is it 10m or 100m away?
```

### The Fusion Problem: Different Coordinate Systems

Here's the core challenge: LiDAR gives you `[x, y, z]` in 3D meters. Camera gives you `[u, v]` pixel locations on a 2D image. How do you know which LiDAR point corresponds to which pixel?

**Answer: Geometric projection using calibration matrices.**

### Step-by-Step: Projecting LiDAR to Camera

Let's say you have a LiDAR point P at position `[x, y, z]` in the LiDAR coordinate frame.

**Step 1: Transform from LiDAR frame to Camera frame**

The LiDAR and camera are physically mounted in different positions on the vehicle. The **extrinsic matrix** `[R|t]` encodes this rigid transformation:

```
P_camera = R × P_lidar + t

Where:
- R is a 3×3 rotation matrix (how the camera is angled relative to LiDAR)
- t is a 3×1 translation vector (physical offset between sensors)
```

**Intuition**: If the camera is mounted 0.5m to the right and 0.3m higher than the LiDAR, the translation `t = [-0.5, 0.3, 0]` shifts all points accordingly. If the camera is tilted down 10°, the rotation R accounts for that.

**Step 2: Project from 3D camera coordinates to 2D pixel coordinates**

Now the point is in the camera's 3D space. We project it onto the image plane using the **intrinsic matrix** K:

```
┌ u ┐       ┌ fx  0  cx ┐   ┌ X ┐
│ v │ = 1/Z │ 0  fy  cy │ × │ Y │
└ 1 ┘       └ 0   0   1 ┘   └ Z ┘

Where:
- fx, fy: focal lengths (in pixels) — how much the lens "zooms"
- cx, cy: principal point — where the optical axis hits the sensor
- Z: depth (distance from camera)
```

**The division by Z is the key**: This is why objects appear smaller when far away. It's perspective projection — the same math Renaissance painters figured out.

### Complete Projection Code

```python
import numpy as np

def project_lidar_to_image(points_lidar, R, t, K):
    """
    Project LiDAR points onto camera image plane.
    
    Args:
        points_lidar: [N, 3] array of XYZ points in LiDAR frame
        R: [3, 3] rotation matrix (LiDAR → Camera)
        t: [3, 1] translation vector
        K: [3, 3] camera intrinsic matrix
    
    Returns:
        pixels: [N, 2] array of (u, v) pixel coordinates
        depths: [N] array of depths (for filtering/coloring)
    """
    # Step 1: Transform to camera coordinates
    points_camera = (R @ points_lidar.T + t).T  # [N, 3]
    
    # Step 2: Filter points behind camera (Z <= 0)
    valid = points_camera[:, 2] > 0
    points_camera = points_camera[valid]
    
    # Step 3: Project to pixel coordinates
    # Homogeneous: [X, Y, Z] → [X/Z, Y/Z, 1]
    points_normalized = points_camera / points_camera[:, 2:3]
    
    # Apply intrinsics
    pixels_homogeneous = (K @ points_normalized.T).T
    pixels = pixels_homogeneous[:, :2]  # Drop the 1
    
    depths = points_camera[:, 2]  # Keep depth for visualization
    
    return pixels, depths
```

### Why Fusion Helps: Complementary Strengths

| Aspect | Camera | LiDAR |
|--------|--------|-------|
| **Density** | Every pixel has data | Sparse (gaps between points) |
| **Depth** | None (2D only) | Precise to centimeters |
| **Texture/Color** | Rich RGB | Only intensity |
| **Cost** | $50 webcam works | $1000-$75000 |
| **Weather** | Fails in darkness | Works at night |
| **Range** | Unlimited (but no depth) | 50-200m typical |

**The fusion insight**: Use LiDAR to know WHERE things are. Use camera to know WHAT they are.

### Fusion Architectures for Autonomous Vehicles

**Early Fusion (Feature Concatenation)**
```
[RGB Image] → CNN → Features [256, H, W]
                              ↓
[LiDAR → Depth Map] → CNN → Features [256, H, W]  → Concat → [512, H, W] → Detector
```
**Problem**: Requires pixel-perfect alignment. Calibration drift = catastrophic failure.

**Late Fusion (Decision Combination)**
```
[RGB Image] → Detector → Bounding Boxes + Scores
                                                 → NMS/Averaging → Final Detections
[LiDAR Points] → PointNet → Bounding Boxes + Scores
```
**Problem**: Loses cross-modal reasoning. Can't answer "is the red blob in the image the same as the nearby LiDAR cluster?"

**Intermediate Fusion (Modern Best Practice)**
```
[RGB Image] → Vision Transformer → Image Tokens [196, 768]
                                                         ↓
[LiDAR → BEV] → 3D Backbone → Point Tokens [1000, 256] → Cross-Attention → Fused Features → Detector
```
**Why it wins**: Cross-attention lets the model learn which image regions correspond to which point clusters, without requiring perfect geometric alignment.

---

## 1.3 Audio Spectrograms: Sound as Vision

### The Insight

Sound is 1D (amplitude over time). Images are 2D (intensity over space). But frequency analysis gives us a second dimension, letting us treat audio as images.

### The Transformation Pipeline

**Step 1: Raw Audio**
```
Waveform: [T] where T = samples (e.g., 16000 samples/second)
Example: 5 seconds of audio at 16kHz = [80000] values
         Each value = microphone displacement at that instant
```

**Step 2: Short-Time Fourier Transform (STFT)**

The Fourier Transform decomposes a signal into frequencies. But audio changes over time — a word spoken at t=1s has different frequencies than the word at t=2s.

STFT solves this by windowing:
```
1. Take a small window (e.g., 25ms = 400 samples)
2. Apply FFT to get frequencies in that window
3. Slide window forward (e.g., 10ms hop)
4. Repeat for entire audio

Result: [Frequency bins, Time frames]
        Typically [257, T/hop]
```

**Step 3: Mel Scale**

Human hearing isn't linear — we're much better at distinguishing 100Hz vs 200Hz than 10000Hz vs 10100Hz.

The mel scale compresses high frequencies:
```
mel = 2595 × log₁₀(1 + f/700)

Effect: Linear spacing in mel = perceptually uniform spacing
```

**Step 4: Log Amplitude**

Human loudness perception is logarithmic (decibels). Taking log of the spectrogram energy matches this.

### Final Result

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio
y, sr = librosa.load('speech.wav', sr=16000)

# Compute mel spectrogram
mel_spec = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_mels=128,      # Number of mel frequency bins
    hop_length=512,   # Samples between frames
    n_fft=2048        # FFT window size
)

# Convert to log scale (dB)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Result shape: [128, T] — looks like a grayscale image!
# Can now feed to CNN/ViT designed for images
```

### Why This Matters for Multimodal

Once audio is a 2D spectrogram:
- Same CNN architectures work (ResNet, ViT)
- Same augmentations apply (crop, mask, mixup)
- Can align with text/images using CLIP-style training

**AudioCLIP, CLAP, Whisper** all exploit this insight.

---

## 1.4 Early vs Late vs Intermediate Fusion

### The Core Question

You have multiple modalities (image, text, audio, LiDAR). When do you combine them?

### Early Fusion: Combine Inputs

```
Concatenate raw/preprocessed features before the main model.

Example (RGB + Depth):
    [3, H, W] RGB
    [1, H, W] Depth
    ──────────────
    [4, H, W] Combined → Single CNN with 4 input channels
```

**Pros:**
- Model learns cross-modal features from scratch
- Can discover subtle correlations

**Cons:**
- Requires aligned inputs (same resolution, same coordinate frame)
- Can't use pretrained single-modality models
- If one modality is noisy/missing, contaminates everything

**When to use:** Tightly coupled modalities (RGB-D cameras where depth is per-pixel aligned)

### Late Fusion: Combine Decisions

```
Run separate models, combine their outputs.

Model_A(image) → Prediction_A [class probabilities or features]
Model_B(text)  → Prediction_B
                              ↓
                    Weighted Average or MLP → Final Prediction
```

**Pros:**
- Use best pretrained model for each modality
- Robust to missing modalities (just drop that branch)
- Simple to implement

**Cons:**
- No cross-modal reasoning during feature extraction
- "Cat in image" and "cat in text" don't interact until the very end

**When to use:** Combining heterogeneous systems; ensembling

### Intermediate Fusion: Combine Representations

```
Separate encoders → Fusion layer(s) → Joint processing

Image → ViT → [CLS] token [768]
                              → Cross-Attention / Concat+MLP → Joint Features → Output
Text  → BERT → [CLS] token [768]
```

**Pros:**
- Cross-modal attention learns which parts of image relate to which words
- Can fine-tune while preserving pretrained knowledge
- Best performance on most benchmarks

**Cons:**
- More complex architecture
- Need to design the fusion mechanism

**When to use:** Most modern multimodal tasks (VQA, captioning, retrieval)

### Where Does CLIP Fit?

**CLIP is NOT intermediate fusion.** This is a common misconception.

CLIP uses **dual encoders with late fusion in embedding space**:

```
Image → ViT → Linear → Normalized embedding [512]
                                                 → Dot Product → Similarity Score
Text  → Transformer → Linear → Normalized embedding [512]

There is NO cross-attention between image and text during encoding.
```

The "fusion" only happens when you compute similarity — the models never see each other's intermediate representations.

**Why this matters:** CLIP can encode images offline, store embeddings, and later compare to any text query. If it used cross-attention, you'd need to re-run the image through the model for every new query.

---

# Part 2: Contrastive Learning and CLIP

## 2.1 The Contrastive Learning Paradigm

### The Problem CLIP Solves

Traditional image classifiers need labeled data:
```
Image of dog → Label: "dog" (one of 1000 ImageNet classes)
```

**Limitations:**
- Fixed vocabulary (can't recognize "golden retriever puppy playing fetch")
- Expensive labeling (millions of images × human annotators)
- No understanding of concepts outside the label set

### CLIP's Insight

The internet has billions of image-text pairs — alt text, captions, descriptions. This is **free supervision**:

```
Photo of a dog playing fetch in the park
[Image of that scene]
```

The text tells us what's in the image. No human labeler needed.

### The Contrastive Objective

**Goal:** Learn embeddings where matching image-text pairs are close, non-matching pairs are far.

**Setup:** Given a batch of N image-text pairs:
```
Batch:
  (image_0, text_0)  ← These match
  (image_1, text_1)  ← These match
  ...
  (image_N-1, text_N-1)  ← These match
```

**Step 1:** Encode everything:
```python
image_embeddings = image_encoder(images)  # [N, 512]
text_embeddings = text_encoder(texts)      # [N, 512]
```

**Step 2:** Normalize (critical for cosine similarity):
```python
image_embeddings = F.normalize(image_embeddings, dim=-1)
text_embeddings = F.normalize(text_embeddings, dim=-1)
```

**Step 3:** Compute all pairwise similarities:
```python
# Matrix multiplication gives all dot products
logits = image_embeddings @ text_embeddings.T  # [N, N]

# logits[i, j] = similarity between image_i and text_j
# The diagonal (i=j) should be highest — those are the matches
```

**Step 4:** Apply temperature scaling:
```python
logits = logits * temperature  # or / temperature depending on convention
# Temperature controls "sharpness" of the softmax
# Lower temp → more peaked distribution → model more confident
```

**Step 5:** Compute loss (symmetric InfoNCE):
```python
labels = torch.arange(N)  # [0, 1, 2, ..., N-1]

# Image→Text: For each image, which text matches?
loss_i2t = F.cross_entropy(logits, labels)

# Text→Image: For each text, which image matches?  
loss_t2i = F.cross_entropy(logits.T, labels)

loss = (loss_i2t + loss_t2i) / 2
```

### Why `labels = torch.arange(N)`?

This confuses many students. Here's the intuition:

Cross-entropy loss expects:
- `logits[i]`: probability distribution over classes for sample i
- `labels[i]`: the correct class for sample i

In contrastive learning:
- `logits[i, :]`: similarity scores between image_i and ALL texts
- `labels[i] = i`: the correct text for image_i is text_i (same index)

So we're saying "image 0 matches text 0, image 1 matches text 1, ..." — which is true by construction of the batch.

### The Role of Batch Size

**Larger batch = more negatives = better representations**

With batch size N:
- Each image has 1 positive text and N-1 negative texts
- Model must distinguish the correct caption from N-1 wrong ones
- Harder task → more discriminative embeddings

CLIP used batch size **32,768**. This is why contrastive learning needs massive compute.

```
Trade-off visualization:

Batch=32:   Easy — only 31 negatives. Model might "cheat" with simple features.
Batch=4096: Hard — 4095 negatives. Must learn fine-grained distinctions.
Batch=32K:  Very hard — 32767 negatives. Learns robust, general features.
```

### Why Cosine Similarity?

**Dot product** of two vectors: `a · b = |a| × |b| × cos(θ)`

The magnitude `|a|` and `|b|` conflate with the angle `θ`. A vector could have high dot product because it's long, not because it's aligned.

**Cosine similarity** normalizes this: `cos(θ) = (a · b) / (|a| × |b|)`

Range: [-1, 1]
- 1.0: Identical direction (semantically identical)
- 0.0: Orthogonal (unrelated)
- -1.0: Opposite directions (semantically opposite)

**In practice**, we L2-normalize embeddings once, then dot product = cosine similarity:
```python
a_norm = a / a.norm()
b_norm = b / b.norm()
similarity = a_norm @ b_norm  # This is cosine similarity
```

---

## 2.2 CLIP Architecture Details

### Image Encoder Options

**ResNet Variant (RN50, RN101)**
```
Standard ResNet with modifications:
- AttentionPool2d instead of global average pool
- Outputs [batch, 1024] or [batch, 2048]
- Projection layer → [batch, 512]
```

**Vision Transformer (ViT-B/32, ViT-L/14)**
```
ViT-B/32: Base model, 32×32 patches
ViT-L/14: Large model, 14×14 patches (more patches = finer detail)

Process:
1. Split image into patches (e.g., 224÷14 = 16×16 = 256 patches)
2. Linear projection of each patch → patch embedding
3. Add [CLS] token + positional embeddings
4. Transformer encoder (12 or 24 layers)
5. [CLS] token output → projection → [512]
```

### Text Encoder

```
Transformer (not BERT, custom trained):
- 12 layers, 8 heads, 512 dim
- BPE tokenization (49152 vocab)
- Max 77 tokens

Process:
1. Tokenize text
2. Add [SOS] and [EOS] tokens
3. Transformer encoder with causal masking
4. Take [EOS] token output → projection → [512]
```

### Zero-Shot Classification

This is CLIP's superpower. Without any training on a new dataset:

```python
# Suppose you want to classify images into ["dog", "cat", "bird"]

# Step 1: Create text prompts
prompts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]

# Step 2: Encode prompts (do this once, cache the result)
text_embeddings = clip.encode_text(prompts)  # [3, 512]

# Step 3: Encode image
image_embedding = clip.encode_image(image)  # [1, 512]

# Step 4: Compute similarities
similarities = image_embedding @ text_embeddings.T  # [1, 3]

# Step 5: Classify
prediction = similarities.argmax()  # Index of highest similarity
```

**Prompt engineering matters:**
- "dog" → 54% accuracy
- "a photo of a dog" → 61% accuracy  
- "a photo of a dog, a type of pet" → 64% accuracy

The prompt should match how captions appeared in training data.

---

## 2.3 PyTorch Tensor Operations for Contrastive Learning

### `repeat_interleave` vs `repeat`

These are used when you need to align tensors of different shapes.

**`repeat_interleave(repeats, dim)`**: Repeat **each element** individually

```python
x = torch.tensor([1, 2, 3])

x.repeat_interleave(2)
# → tensor([1, 1, 2, 2, 3, 3])

# Think: "stutter" — each element stutters/repeats before moving on
```

**`repeat(*sizes)`**: Tile **the entire tensor**

```python
x = torch.tensor([1, 2, 3])

x.repeat(2)
# → tensor([1, 2, 3, 1, 2, 3])

# Think: "echo" — the whole sequence echoes
```

### When You Use Each

**Scenario:** You have 4 images and want to compare each against 4 different text prompts (16 pairs total).

```python
images = torch.randn(4, 512)  # 4 image embeddings
texts = torch.randn(4, 512)   # 4 text embeddings

# To compute all 16 pairwise similarities:

# Option A: Matrix multiplication (elegant)
similarities = images @ texts.T  # [4, 4]

# Option B: Explicit expansion (illustrative)
images_expanded = images.repeat_interleave(4, dim=0)  # [16, 512]
# [img0, img0, img0, img0, img1, img1, img1, img1, ...]

texts_expanded = texts.repeat(4, 1)  # [16, 512]  
# [txt0, txt1, txt2, txt3, txt0, txt1, txt2, txt3, ...]

# Now images_expanded[i] pairs with texts_expanded[i]
similarities = (images_expanded * texts_expanded).sum(dim=-1)  # [16]
```

---

# Part 3: Vision Language Models and Cross-Modal Projection

## 3.1 The VLM Challenge

### Why Can't We Just Concatenate?

LLMs process **token sequences**. They expect:
```
Input: [token_1, token_2, ..., token_n]
Each token: an integer ID from vocabulary (e.g., 0-50256 for GPT-2)
Embedded: vocabulary embedding → [768] or [4096] vector per token
```

Images aren't tokens. You could:
1. Describe the image in text → loses visual detail
2. Feed raw pixels → LLM has no idea what pixels mean
3. **Use a vision encoder to create "visual tokens"** → VLM approach

### The Projection Problem

**Vision encoder output:** `[batch, num_patches, vision_dim]`
- CLIP ViT-L/14: `[1, 257, 1024]` (256 patches + 1 CLS token)

**LLM embedding space:** `[batch, seq_len, llm_dim]`
- LLaMA-7B: `[1, *, 4096]`

**Problems:**
1. **Dimension mismatch:** 1024 ≠ 4096
2. **Semantic mismatch:** Vision features encode visual patterns; LLM expects "word-like" meanings
3. **Distribution mismatch:** Vision encoder wasn't trained to produce LLM-compatible representations

### The Projection Layer

A learned transformation that bridges vision → language space:

```python
class ProjectionLayer(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        # Simple linear
        self.proj = nn.Linear(vision_dim, llm_dim)
        
        # Or 2-layer MLP (LLaVA uses this)
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    
    def forward(self, vision_features):
        # vision_features: [batch, num_patches, vision_dim]
        return self.proj(vision_features)
        # Output: [batch, num_patches, llm_dim]
```

**Intuition:** The projection learns to "translate" visual concepts into the LLM's language of thought. A patch showing "grass" gets projected near where the LLM represents the concept "grass".

### Why MLP > Linear?

A single linear layer can only rotate/scale/shear the embedding space. It's an **affine transformation**.

But the vision and language spaces may have different **topologies** — concepts that are far apart in vision space might be near in language space, or vice versa.

An MLP with nonlinearity (GELU/ReLU) can **warp** the space:
```
Linear only: Preserves straight lines
MLP: Can bend, fold, stretch the space
```

Think of it like this: Vision space might represent "color" and "shape" as orthogonal axes. Language space might cluster by "animal vs vehicle". The MLP learns the nonlinear mapping.

---

## 3.2 LLaVA Architecture

### Overview

```
                    ┌──────────────────┐
                    │ Vision Encoder   │
                    │ (CLIP ViT-L/14)  │
Image ──────────────│     Frozen       │──── Vision Features [257, 1024]
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Projection     │
                    │     (MLP)        │──── Visual Tokens [256, 4096]
                    │   Trainable      │
                    └────────┬─────────┘
                             │
                             ▼
            ┌────────────────┴────────────────┐
            │                                 │
            ▼                                 ▼
    [Visual Tokens]                    [Text Tokens]
    [256 × 4096]                       [N × 4096]
            │                                 │
            └─────────── Concatenate ─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │      LLM         │
                    │    (Vicuna)      │──── Generated Response
                    │  Frozen or LoRA  │
                    └──────────────────┘
```

### Two-Stage Training

**Stage 1: Feature Alignment (Pretraining)**
- Vision encoder: Frozen
- Projection: **Trained**
- LLM: Frozen

Data: Simple image-caption pairs
Goal: Learn to project visual features so LLM can "understand" them

Think of it as teaching the projector to speak the LLM's language.

**Stage 2: Visual Instruction Tuning**
- Vision encoder: Frozen
- Projection: Fine-tuned
- LLM: LoRA fine-tuning or full fine-tuning

Data: Instruction-following conversations with images
Goal: Teach the combined system to follow complex visual instructions

---

## 3.3 Document Processing for RAG

### PDF Chunking Strategies

**Fixed-Size Chunking**
```
Split every N characters (e.g., 1000 chars)

Pros: Simple, predictable chunk sizes
Cons: May split mid-sentence, mid-table, mid-paragraph
```

**Recursive/Hierarchical Chunking**
```
Try to split by: paragraph → sentence → word → character
Use the largest unit that keeps chunks under size limit

# LangChain example
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,  # Overlap prevents losing context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]  # Try each in order
)
```

**Semantic Chunking**
```
Use sentence embeddings to find natural breakpoints

1. Embed each sentence
2. Compute similarity between adjacent sentences
3. Split where similarity drops (topic change)
```

**Layout-Aware Chunking**
```
Use document structure:
- Split at section headers
- Keep tables intact
- Preserve list items together
- Treat figures + captions as units
```

### Identifying Page Elements

Document AI models detect element types:

```
Element Types:
- Title
- Section Header
- Paragraph / Narrative Text
- Table
- Figure
- List Item
- Caption
- Page Header / Footer
```

**Tools:**
- `unstructured` library (open source)
- LayoutLM / LayoutLMv3 (Microsoft)
- DocTR (document text recognition)
- NVIDIA DALI (GPU-accelerated)

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf("document.pdf")

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text[:100]}...")
```

---

# Part 4: Vector Search, RAG, and Knowledge Graphs

## 4.1 Vector Databases

### The Problem They Solve

You have 1 million document chunks, each embedded as a 768-dim vector. Given a query vector, find the 10 most similar chunks.

**Brute force:** Compare query to all 1M vectors. O(N) comparisons per query.
At 1ms per comparison → 1000 seconds per query. Unusable.

**Approximate Nearest Neighbor (ANN):** Trade slight accuracy for massive speedup.

### HNSW (Hierarchical Navigable Small World)

The most popular ANN algorithm. Think of it as a multi-resolution graph:

```
Layer 3 (sparse):     A ─────────────────── B
                          
Layer 2:              A ───── C ───── B
                          
Layer 1:              A ── D ── C ── E ── B

Layer 0 (dense):      A─D─F─G─C─H─I─E─J─B

Search algorithm:
1. Start at Layer 3 (sparse), find approximate region
2. Drop to Layer 2, refine search
3. Continue dropping and refining
4. Layer 0: Final precise search in local neighborhood
```

**Complexity:** O(log N) vs O(N) for brute force

**Parameters:**
- `M`: Edges per node (higher = more accurate, more memory)
- `ef_construction`: Search depth during index building
- `ef_search`: Search depth during query (trade-off: speed vs accuracy)

### Using Vector DBs

```python
# Milvus example
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# Connect
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
]
schema = CollectionSchema(fields, "Document chunks")
collection = Collection("documents", schema)

# Insert
collection.insert([ids, embeddings, texts])

# Create HNSW index
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}
collection.create_index("embedding", index_params)

# Search
collection.load()
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=10
)
```

---

## 4.2 RAG (Retrieval-Augmented Generation)

### The Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INDEXING (Offline)                           │
│                                                                     │
│  Documents → Chunking → Embedding → Vector DB                       │
│     │           │          │           │                            │
│     ▼           ▼          ▼           ▼                            │
│   PDFs      1000 char   CLIP/BGE    Milvus                          │
│   HTML      chunks      model       Pinecone                        │
│   etc.                                                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL (Online)                           │
│                                                                     │
│  Query → Embed → Vector Search → Top-K Chunks → Augmented Prompt    │
│    │       │          │              │                │             │
│    ▼       ▼          ▼              ▼                ▼             │
│  "What    [768]     HNSW         [chunk1,          "Context:        │
│   is X?"   vec      search        chunk2,          chunk1...        │
│                                   chunk3]          Question: X"     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        GENERATION                                   │
│                                                                     │
│  Augmented Prompt → LLM → Answer                                    │
│        │             │       │                                      │
│        ▼             ▼       ▼                                      │
│  "Context: ...     GPT-4   Grounded                                 │
│   Question: X"     Llama   response                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Why RAG Helps

**Without RAG:**
- LLM only knows training data (cutoff date)
- Hallucinates when uncertain
- Can't cite sources

**With RAG:**
- Fresh knowledge from your documents
- Grounded in retrieved context
- Can cite specific chunks

---

## 4.3 Graph RAG

### The Limitation of Vector RAG

Vector similarity finds semantically similar text. But some questions require **relational reasoning**:

"What companies has the CEO of Acme Corp previously worked for?"

This requires:
1. Find Acme Corp's CEO (relationship: works_at)
2. Find that person's previous employers (relationship: previously_worked_at)

Vector search might find chunks mentioning the CEO, but won't reliably chain relationships.

### Knowledge Graph Structure

```
Nodes: Entities (people, companies, concepts)
Edges: Relationships (works_at, located_in, acquired_by)

Example:
(John Smith) ─[CEO_of]─→ (Acme Corp)
(John Smith) ─[previously_worked_at]─→ (Big Tech Inc)
(Acme Corp) ─[headquartered_in]─→ (San Francisco)
```

### Cypher Query Language

```cypher
// Find the CEO of Acme Corp
MATCH (p:Person)-[:CEO_of]->(c:Company {name: "Acme Corp"})
RETURN p.name

// Find all previous employers of Acme Corp's CEO
MATCH (p:Person)-[:CEO_of]->(c:Company {name: "Acme Corp"})
MATCH (p)-[:previously_worked_at]->(prev:Company)
RETURN prev.name

// Find all people within 2 relationship hops of John
MATCH (p:Person {name: "John Smith"})-[*1..2]-(connected)
RETURN connected
```

### Hybrid RAG: Vector + Graph

Best of both worlds:

```
Query: "What is Acme Corp's revenue strategy?"

1. Vector search: Find chunks mentioning Acme Corp + revenue
2. Graph traversal: Find related entities (CEO, products, competitors)
3. Combine: Augmented prompt includes chunks + entity context
4. Generate: LLM synthesizes answer with full context
```

---

## 4.4 NVIDIA AI Blueprint for VSS

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     VIDEO SEARCH & SUMMARIZATION                    │
│                                                                     │
│   Video Files                                                       │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────────┐                                               │
│   │ Data Plane      │  DeepStream: decode, sample frames            │
│   │ (Ingestion)     │  Chunk videos (10-60s segments)               │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │ CV NIMs         │  Object detection, tracking, segmentation     │
│   │ (Perception)    │  Extract structured scene info                │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐                                               │
│   │ VLM NIMs        │  Caption frames, embed with CLIP              │
│   │ (Understanding) │  Visual-to-text description                   │
│   └────────┬────────┘                                               │
│            │                                                        │
│            ▼                                                        │
│   ┌─────────────────┐     ┌─────────────────┐                       │
│   │ Vector DB       │     │ Graph DB        │                       │
│   │ (Milvus)        │     │ (Neo4j)         │                       │
│   │ Semantic search │     │ Entity relations│                       │
│   └────────┬────────┘     └────────┬────────┘                       │
│            │                       │                                │
│            └───────────┬───────────┘                                │
│                        │                                            │
│                        ▼                                            │
│   ┌─────────────────────────────────┐                               │
│   │ NeMo Retriever (RAG NIM)        │                               │
│   │ Retrieves relevant video chunks │                               │
│   └────────────────┬────────────────┘                               │
│                    │                                                │
│                    ▼                                                │
│   ┌─────────────────────────────────┐                               │
│   │ LLM NIM                         │                               │
│   │ Summarize, answer questions     │                               │
│   └─────────────────────────────────┘                               │
│                    │                                                │
│                    ▼                                                │
│              Natural Language Response                              │
│              "The forklift appears at timestamps..."                │
└─────────────────────────────────────────────────────────────────────┘
```

### Typical Query Flow

```
User: "Show me all instances of people not wearing hard hats"

1. EMBED: Convert query to vector
2. FILTER: (Optional) Restrict to "warehouse" camera, "this week"
3. VECTOR SEARCH: Find chunks with high similarity to safety violations
4. GRAPH QUERY: Find relationships (Person → not_wearing → HardHat)
5. RETRIEVE: Pull video timestamps + frame captions
6. SUMMARIZE: LLM generates report with timestamps

Response:
"Found 3 instances of PPE violations:
- Camera 2, 14:32:15 - Worker near forklift without hard hat
- Camera 5, 16:45:02 - Two workers in loading area without PPE
- Camera 2, 17:12:44 - Same worker from earlier incident
"
```

---

# Quick Reference: Common Student Questions

## Conceptual

**Q: "Why not just describe images in text and use a text-only LLM?"**
A: You lose information. "A dog running" doesn't capture breed, background, lighting, composition. VLMs preserve visual nuance while enabling language interface.

**Q: "What makes CLIP zero-shot work?"**
A: CLIP learns concepts from natural language descriptions, not fixed class labels. If training data included "a photo of a corgi puppy playing in snow", CLIP learned that concept — even if ImageNet never had that class.

**Q: "When should I use Graph RAG vs Vector RAG?"**
A: Vector for semantic similarity (find similar content). Graph for relational reasoning (multi-hop questions, entity relationships). Often: use both.

## Implementation

**Q: "My CLIP similarities are all 0.2-0.3 and nothing stands out"**
A: Check: (1) embeddings normalized? (2) using correct output layer? (3) temperature applied? (4) text prompts match training distribution ("a photo of X")?

**Q: "RAG retrieves irrelevant chunks"**
A: Debug steps: (1) check embedding model quality (2) try smaller chunks (3) add metadata filtering (4) implement re-ranking (5) check if query and chunks use same embedding model.

**Q: "Why is batch size so important for contrastive learning?"**
A: More batch = more negatives. With 32 negatives, easy to distinguish correct pair. With 32K negatives, model must learn fine-grained features. This is why CLIP needed massive compute.

---

# Certification Checklist

Use this to verify you can explain each topic:

- [ ] CNN fundamentals: convolution, pooling, receptive field
- [ ] PyTorch basics: Module, forward, backward, DataLoader
- [ ] Vision data shapes: RGB, grayscale, CT (3D), point cloud
- [ ] Audio spectrograms: STFT, mel scale, why it enables vision models
- [ ] Color modes: RGB vs BGR vs RGBA and why it matters
- [ ] LiDAR vs camera: data formats, strengths/weaknesses, projection math
- [ ] Fusion types: early, late, intermediate — when to use each
- [ ] CLIP architecture: dual encoder, not cross-attention
- [ ] Contrastive training: InfoNCE loss, why labels = indices
- [ ] Cosine similarity vs dot product: formula, when to use each
- [ ] repeat_interleave vs repeat: element-wise vs tensor-wise
- [ ] VLM architecture: vision encoder → projection → LLM
- [ ] Cross-modal projection: why MLP, dimension alignment
- [ ] LLaVA: two-stage training, frozen components
- [ ] PDF chunking: fixed, recursive, semantic, layout-aware
- [ ] Page element identification: types, tools
- [ ] Vector databases: HNSW, ANN, complexity trade-offs
- [ ] RAG pipeline: index → retrieve → augment → generate
- [ ] Vector RAG vs Graph RAG: semantic vs relational
- [ ] Context-aware RAG: metadata filtering, re-ranking
- [ ] Cypher queries: MATCH, relationships, multi-hop
- [ ] VSS applications: surveillance, media, compliance
- [ ] NVIDIA Blueprint VSS: components, data flow

---

*End of Deep Dive Study Guide*
