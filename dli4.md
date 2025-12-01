# Building AI Agents with Multimodal Models

## NVIDIA Deep Learning Institute — Instructor Certification Study Guide

---

# Part 1: Foundations

---

## 1. Convolutional Neural Network Fundamentals

### High-Level Concept

Convolutional Neural Networks (CNNs) are neural architectures that use small learnable filters sliding over images to detect patterns. They progress from simple edges in early layers to complex objects in deeper layers. Weight sharing makes them efficient and translation-equivariant — a cat detected in one corner can be detected anywhere.

### Key Components

**Kernel (Filter):**
A small learnable matrix, typically 3×3 or 5×5 pixels. During the forward pass, the kernel is placed over an image region, multiplied element-wise with pixel values, and summed to produce one output value. The kernel then slides to the next position and repeats. A 3×3 kernel has only 9 learnable parameters but is reused across the entire image.

**Stride:**
How far the kernel moves between positions. Stride of 1 means the kernel moves one pixel at a time, roughly preserving spatial dimensions. Stride of 2 means the kernel moves two pixels, halving the output dimensions (downsampling). Modern architectures often use stride 2 instead of pooling for dimensionality reduction.

**Padding:**
Zeros added around image borders. Without padding ("valid"), output shrinks because the kernel cannot center on edge pixels. With "same" padding, zeros are added so the kernel can center on every pixel, preserving spatial dimensions. For a 3×3 kernel, padding of 1 preserves size. For 5×5, padding of 2 preserves size.

**Pooling:**
Reduces spatial dimensions by summarizing local regions. Max pooling takes the maximum value in each 2×2 window. Average pooling takes the mean. Pooling adds robustness to small spatial shifts — a cat shifted 2 pixels still activates the same pooled neuron.

**Receptive Field:**
How much of the original image one neuron can "see." A layer 1 neuron with a 3×3 kernel sees 3×3 pixels. A layer 2 neuron sees through layer 1, effectively seeing 5×5 pixels. Deeper layers have exponentially larger receptive fields, enabling whole-object recognition.

**Tensor Shape:**
CNNs expect input as `[Batch, Channels, Height, Width]`. For example, 32 RGB images at 512×512 resolution would be `[32, 3, 512, 512]`.

### Role in Multimodal Pipelines

CNN encoders transform raw images into dense feature tensors with preserved spatial structure. These features can then be fused with text, audio, Light Detection and Ranging (LiDAR), or other modalities. The output before the classification head is typically used for fusion.

### Notebook Connection

In Lab 01a, you modified VGG's first convolutional layer from 3 to 4 input channels for early fusion with depth maps.

---

## 2. PyTorch Framework Essentials

### High-Level Concept

PyTorch is the deep learning framework used throughout this course. It provides tensors (multi-dimensional arrays with GPU support), automatic differentiation (computing gradients without manual calculus), and modular building blocks for neural networks.

### Key Components

**nn.Module:**
Base class for all neural network components. You define layers in `__init__()` and specify the computation flow in `forward()`. All custom models inherit from this class.

**Autograd (Automatic Differentiation):**
PyTorch tracks all operations on tensors that have `requires_grad=True`. When you call `loss.backward()`, it automatically computes gradients for all parameters using the chain rule. No manual derivative calculation needed.

**Training Loop Pattern:**
The standard pattern is: forward pass (`output = model(input)`) → compute loss (`loss = criterion(output, target)`) → backward pass (`loss.backward()`) → update weights (`optimizer.step()`) → clear gradients (`optimizer.zero_grad()`).

**DataLoader:**
Wraps a Dataset object to handle batching, shuffling, and parallel data loading. Iterating over a DataLoader yields batches ready for training.

**GPU Usage:**
Move tensors and models to GPU with `.to("cuda")` or `.cuda()`. Always ensure model and data are on the same device. Check availability with `torch.cuda.is_available()`.

### Notebook Connection

All labs use PyTorch. You'll write training loops, modify model architectures, and move tensors between CPU and GPU.

---

## 3. Vision Data Types and Tensor Shapes

### High-Level Concept

Different sensors produce different tensor formats. Standard cameras output dense 2D images. Depth sensors add distance information. Light Detection and Ranging (LiDAR) produces sparse 3D point clouds. Medical scanners produce 3D volumes. Models expect specific shapes — mismatches cause errors or silent failures.

### Data Types

**RGB Images:**
Shape is `[Batch, 3, Height, Width]`. Three color channels (Red, Green, Blue) with values 0-255 (raw) or 0-1 (normalized). Example: `[32, 3, 512, 512]` for 32 images at 512×512 resolution.

**Grayscale Images:**
Shape is `[Batch, 1, Height, Width]`. Single intensity channel. Converted from RGB using: `gray = 0.2126×R + 0.7152×G + 0.0722×B`. The unequal weights match human eye sensitivity — we have more green-sensitive cones than red or blue.

**Depth Maps:**
Shape is `[Batch, 1, Height, Width]`. Each pixel value represents distance in meters rather than color intensity. Can be dense (from stereo cameras) or sparse (from projected LiDAR).

**Point Clouds:**
Shape is `[N, 3]` or `[N, 4]` for coordinates `(x, y, z)` or `(x, y, z, intensity)`. These are unordered, sparse, and variable-sized. Standard 2D convolutions cannot process them directly — requires specialized architectures like PointNet, voxelization into 3D grids, or projection to 2D views.

**Computed Tomography (CT) Scans:**
Shape is `[Batch, 1, Depth, Height, Width]`. The depth dimension represents spatial slices through the body — adjacent slices show adjacent anatomy. This is NOT time. Example: `[1, 1, 200, 512, 512]` for 200 slices at 512×512. Requires 3D convolutions to capture cross-slice patterns.

### Notebook Connection

Lab 01b loads and visualizes each data type. Always verify `.shape` before feeding tensors to models.

---

## 4. Audio Spectrograms

### High-Level Concept

Raw audio is a one-dimensional waveform (amplitude over time). Spectrograms transform this into a two-dimensional frequency-versus-time image, allowing Convolutional Neural Networks and Vision Transformers to process sound using the same architectures designed for images.

### Transformation Pipeline

**Step 1 — Raw Waveform:**
A 1D array of amplitude values sampled at a fixed rate. For example, 5 seconds of audio at 16,000 samples per second produces 80,000 values.

**Step 2 — Short-Time Fourier Transform (STFT):**
Slide a small time window (typically 25 milliseconds) over the audio. For each window position, compute the frequency content using the Fourier Transform. Stack results into a 2D matrix with shape `[frequency_bins, time_frames]`. This shows how frequency content changes over time.

**Step 3 — Mel Scale:**
Human hearing is not linear — we easily distinguish 100 Hz from 200 Hz, but 10,000 Hz and 10,100 Hz sound nearly identical. The mel scale warps frequencies to match human perception using the formula: `mel = 2595 × log₁₀(1 + frequency/700)`. This compresses high frequencies where our hearing is less sensitive.

**Step 4 — Log Amplitude:**
Human loudness perception is also logarithmic (we use decibels for this reason). Taking the log of spectrogram energy matches perceived loudness differences.

**Final Tensor:**
Shape is typically `[1, mel_bins, time_frames]` — treated as a single-channel grayscale image. Time is preserved along the horizontal axis (columns represent time frames).

### Why This Matters

Once audio becomes a 2D spectrogram, the same CNN/ViT architectures work. The same augmentations apply (random crop, masking, mixup). Audio processing becomes image processing.

### Notebook Connection

Lab 01b converts audio to mel spectrograms using the librosa library and visualizes with matplotlib.

---

## 5. Image Color Modes

### High-Level Concept

Different libraries store color channels in different orders. This seemingly minor detail causes catastrophic failures when mismatched — models see completely wrong colors.

### Modes

**RGB (Red-Green-Blue):**
Channel order is `[Red, Green, Blue]`. Used by Python Imaging Library (PIL), PyTorch, TensorFlow, and most pretrained deep learning models.

**BGR (Blue-Green-Red):**
Channel order is `[Blue, Green, Red]`. This is the default for OpenCV when loading images with `cv2.imread()`. Historical artifact from early camera hardware.

**RGBA (Red-Green-Blue-Alpha):**
Four channels including transparency. Most models expect only 3 channels. Convert using `image.convert("RGB")` which composites transparency onto white.

### The Critical Bug

If you load an image with OpenCV (BGR) and feed it directly to a model trained on RGB data:

- Red objects appear blue
- Blue objects appear orange/red
- Green stays roughly the same (middle channel)
- Accuracy drops 40-60%

The model sees a completely different image than intended.

### The Fix

Always convert explicitly when using OpenCV:
```python
image = cv2.imread("photo.jpg")  # Loads as BGR
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
```

Or use PIL which loads RGB by default:
```python
from PIL import Image
image = Image.open("photo.jpg")  # Already RGB
```

### Notebook Connection

Lab 01b checks channel ordering. Always sanity-check by verifying first pixel values match expected colors visually.

---

## 6. Computed Tomography Scan Structure

### High-Level Concept

Computed Tomography (CT) scans are three-dimensional medical images created by taking many X-ray slices through the body. Unlike video where the third dimension is time, in CT the third dimension is spatial — adjacent slices show adjacent anatomy.

### Tensor Shape

Shape is `[Batch, Channels, Depth, Height, Width]`. A typical scan might be `[1, 1, 200, 512, 512]` meaning 1 scan, 1 channel (grayscale X-ray), 200 slices, each 512×512 pixels.

### Hounsfield Units

CT intensity values are calibrated to a standardized density scale called Hounsfield Units (HU):

- Air: approximately -1000 HU
- Lung tissue: approximately -500 HU
- Water: 0 HU (calibration reference)
- Soft tissue: approximately +40 HU
- Bone: approximately +1000 HU
- Metal implants: approximately +3000 HU

### Windowing

Different HU ranges reveal different anatomy. "Windowing" maps a selected HU range to display values:

- Lung window (center=-600, width=1500): reveals air-filled lung structures
- Bone window (center=400, width=1800): reveals skeletal structures

The same CT data shows completely different anatomy with different window settings.

### Why 3D Convolutions

A tumor visible in slice k typically continues into slices k-1 and k+1. Standard 2D convolutions per slice miss this cross-slice context. 3D convolutions (`Conv3d`) have kernels that span across slices, capturing volumetric patterns.

### Notebook Connection

CT concepts appear in Lab 01b. The key insight is that depth is spatial, not temporal — requiring 3D spatial reasoning.

---

## 7. Light Detection and Ranging versus Camera Data

### High-Level Concept

Light Detection and Ranging (LiDAR) sensors fire laser pulses and measure return time to calculate precise distances. Cameras capture dense color images but lack depth information. These modalities are complementary — cameras tell you WHAT something is, LiDAR tells you WHERE it is.

### Camera Characteristics

**Tensor shape:** `[3, Height, Width]` — dense coverage with every pixel having an RGB value.

**Strengths:**

- Rich semantic information (color, texture, text, patterns)
- Dense coverage at every pixel
- Inexpensive sensors ($50-500)
- Well-understood processing pipelines

**Weaknesses:**

- No direct depth measurement (must infer from stereo or motion)
- Fails in darkness without illumination
- Affected by glare, reflections, fog

**Coordinate system:** X points right (image columns), Y points down (image rows), Z points forward into the scene.

### LiDAR Characteristics

**Tensor shape:** `[N, 3]` or `[N, 4]` for `(x, y, z)` or `(x, y, z, intensity)`. Sparse — typically 100,000 to 1,000,000 points with gaps between them.

**Strengths:**

- Precise depth measurement (centimeter accuracy)
- Works in complete darkness
- Accurate 3D geometry
- Unaffected by lighting conditions

**Weaknesses:**

- Sparse coverage with gaps between points
- No color or texture information
- Expensive sensors ($1,000 to $75,000)
- Affected by rain, fog, dust (scatters laser)

**Coordinate system:** X points forward (direction of travel), Y points left, Z points up. Note this differs from camera coordinates.

### Complementary Nature

Neither modality captures complete reality alone. Cameras see a red car clearly but cannot tell if it's 10 or 100 meters away. LiDAR knows exact distance but cannot tell car color or read license plates. Fusion combines both for robust perception.

### Notebook Connection

Lab 01b visualizes both modalities. You observe how LiDAR appears sparse when projected onto dense camera images.

---

## 8. LiDAR to Camera Projection

### High-Level Concept

To fuse LiDAR and camera data, you must project 3D LiDAR points onto the 2D camera image plane. This requires two transformations: extrinsic (accounting for sensor mounting positions) and intrinsic (accounting for camera optics). The mathematics is the same perspective projection that Renaissance painters discovered.

### Step 1: Extrinsic Transformation

The LiDAR and camera are mounted at different positions and angles on the vehicle. The extrinsic parameters account for this offset.

**Rotation matrix R (3×3):** Encodes the angle difference between sensors. If the camera is tilted 10 degrees relative to LiDAR, R captures this rotation.

**Translation vector t (3×1):** Encodes the physical distance offset. If the camera is 0.5 meters to the right and 0.3 meters above the LiDAR, t captures this.

**Formula:** `Point_camera = R × Point_lidar + t`

This transforms each LiDAR point from the LiDAR coordinate system to the camera coordinate system.

### Step 2: Intrinsic Projection

The camera intrinsic matrix K (3×3) encodes the camera's optical properties:

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

Where:
- fx, fy are focal lengths in pixels (how much the lens zooms)
- cx, cy are the principal point (where the optical axis intersects the sensor, usually near image center)

**Formula:** `[u, v, 1]ᵀ = (1/Z) × K × [X, Y, Z]ᵀ`

### Why Divide by Z?

The division by Z creates perspective — this is the key mathematical operation. Objects at Z=10 meters project to larger pixel offsets than identical objects at Z=100 meters. This is why:

- Distant objects appear smaller
- Parallel railroad tracks appear to converge at the horizon
- A ball thrown toward you appears to grow

This is identical to how pinhole cameras work and how Renaissance painters achieved realistic perspective.

### Filtering

Points with Z ≤ 0 are behind the camera or on the camera plane — these must be discarded as they cannot be projected meaningfully.

### Result

A sparse set of pixel coordinates where LiDAR points land on the image. Because LiDAR has far fewer points than the camera has pixels, the projected depth appears sparse with gaps.

### Notebook Connection

Lab 01b applies this projection using calibration matrices (R, t, K) from calibration files. You visualize the sparse depth overlay on camera images.

---

# Part 2: Sensor Fusion Approaches

---

## 9. Early Fusion (Input-Level)

### High-Level Concept

Early fusion combines modalities at the input by concatenating channels before any processing. A single shared network then processes the combined tensor. The model learns cross-modal features from the very first layer.

### How It Works

For RGB images plus depth maps:

- RGB tensor: `[Batch, 3, Height, Width]`
- Depth tensor: `[Batch, 1, Height, Width]`
- Concatenated: `[Batch, 4, Height, Width]`

The first convolutional layer must be modified to accept 4 input channels instead of 3. Each 3×3 kernel now has shape 3×3×4 instead of 3×3×3, learning to combine color AND depth from the very first convolution.

### Model Modification

```python
# Original expects 3 channels
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

# Modified for early fusion
model.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3)
```

The new fourth channel weights must be initialized:

- Copy weights from an existing channel (e.g., duplicate red channel)
- Random initialization
- Zero initialization (depth has no initial effect, learned during training)

### Advantages

- Simple architecture with no separate branches
- Captures low-level cross-modal interactions from the start
- Can learn subtle correlations between modalities

### Disadvantages

- Requires pixel-perfect spatial alignment between modalities
- Cannot reuse pretrained 3-channel weights directly
- If one modality is noisy or missing, it contaminates all processing
- Harder to pretrain on single-modality data

### Notebook Connection

In Lab 01a, you modified VGG's first layer from 3 to 4 input channels, initialized the depth channel by copying red channel weights, then trained on fused RGB+depth input.

---

## 10. Late Fusion (Decision-Level)

### High-Level Concept

Late fusion runs completely separate models on each modality, then combines only their final predictions. Each modality has its own specialized encoder. The encoders never see each other's data — they only interact at the decision stage through averaging, voting, or a small fusion network.

### How It Works

```python
# Separate forward passes
logits_rgb = model_rgb(rgb_image)      # [Batch, num_classes]
logits_depth = model_depth(depth_map)  # [Batch, num_classes]

# Combine at decision level
final_logits = (logits_rgb + logits_depth) / 2
```

### Combination Methods

**Simple Average:** Add logits and divide by 2. Treats both modalities equally.

**Learned Weights:** `w1 × logits_rgb + w2 × logits_depth` where weights are learned during training.

**Fusion Network:** Concatenate logits, pass through a small Multi-Layer Perceptron (MLP) to produce final prediction.

**Voting:** Each model makes a prediction, majority wins.

### Key Insight

The RGB model never sees depth data. The depth model never sees color. "Cat in RGB" and "cat-shaped depth blob" are processed by completely independent networks. They only interact when combining final predictions.

### Advantages

- Can reuse best pretrained model for each modality
- Robust if one modality is missing (just use the available one)
- Simple and modular architecture
- Each branch can be trained and tuned independently

### Disadvantages

- No cross-modal feature learning during encoding
- Interactions only at the final decision layer
- May miss correlations that require seeing both modalities together

### Notebook Connection

In Lab 01a, you ran separate forward passes through RGB and depth models, then averaged their probability outputs. You compared accuracy against early fusion.

---

## 11. Intermediate Fusion (Feature-Level)

### High-Level Concept

Intermediate fusion uses separate encoders for each modality but combines their learned feature representations mid-network — not at raw input, not at final output, but at a meaningful feature level. This enables deep cross-modal reasoning while still leveraging modality-specific pretrained encoders.

### How It Works

```
Image → Vision Encoder → image_features [Batch, 1024]
                              ↓
                        Fusion Layer
                              ↑
Text → Text Encoder → text_features [Batch, 768]
                              ↓
                        Fused Features
                              ↓
                         Classifier
```

### Projection to Common Dimension

When encoder output dimensions differ, project to a shared space:

```python
image_proj = nn.Linear(1024, 512)
text_proj = nn.Linear(768, 512)

image_features = image_proj(vision_encoder(image))  # [Batch, 512]
text_features = text_proj(text_encoder(text))       # [Batch, 512]
```

### Fusion Mechanisms

**Concatenation + MLP:**
Stack features and pass through a Multi-Layer Perceptron:
```python
fused = torch.cat([image_features, text_features], dim=1)  # [Batch, 1024]
output = fusion_mlp(fused)
```

**Element-wise Addition:**
Requires same dimensions. Simply add: `fused = image_features + text_features`

**Cross-Attention:**
One modality queries the other for relevant information. Text tokens attend to relevant image patches:

`Attention(Q, K, V) = softmax(QKᵀ / √d) × V`

Where Q (queries) come from text, K (keys) and V (values) come from image. This allows text tokens to "look at" whichever image regions are most relevant.

### Why Intermediate Fusion Wins

- Avoids early fusion's noise problem — each encoder processes clean single-modality data
- Enables deep cross-modal reasoning — features interact, not just predictions
- Attention weights are interpretable — you can visualize which image regions relate to which words
- Can still leverage pretrained encoders — just add fusion layers on top

### Notebook Connection

Lab 02a implements feature fusion with concatenation followed by an MLP classifier. This intermediate approach typically outperforms both early and late fusion.

---

# Part 3: Contrastive Learning and CLIP

---

## 12. Contrastive Language-Image Pretraining (CLIP) Architecture

### High-Level Concept

CLIP uses two completely separate encoders — one for images, one for text — trained so that matching image-text pairs have high similarity and non-matching pairs have low similarity in a shared embedding space. Critically, there is NO cross-attention between modalities during encoding.

### Architecture

**Vision Encoder:**
A Vision Transformer (ViT) or Convolutional Neural Network processes the image and outputs a fixed-size embedding (e.g., 512 dimensions).

**Text Encoder:**
A Transformer processes the text and outputs a fixed-size embedding (same dimension as vision encoder, e.g., 512).

**Embedding Processing:**
Both outputs are L2-normalized to unit length. Similarity is computed as dot product (which equals cosine similarity after normalization).

### Why No Cross-Attention Matters

Because the encoders don't interact during forward pass:

- You can pre-compute image embeddings offline
- Store embeddings for millions of images in a database
- When a new text query arrives, just encode the query and search stored embeddings
- No need to re-run the image encoder for each query

If cross-attention existed, every query would require running both encoders together — O(N×M) forward passes for N images and M queries. This would be impractical for large-scale search.

### Training Data

CLIP was trained on 400 million image-text pairs scraped from the internet (alt text, captions, descriptions). No manual labeling was required — the pairing itself provides supervision.

### Zero-Shot Classification

CLIP can classify images into categories it was never explicitly trained on:

1. Create text prompts for each category: "a photo of a dog", "a photo of a cat"
2. Encode all prompts to get text embeddings
3. Encode the image to get image embedding
4. Compare image embedding to all text embeddings
5. Select category with highest similarity

The model has never seen explicit "dog" or "cat" labels — it learned these concepts from natural language descriptions.

### Notebook Connection

Lab 02b loads `openai/clip-vit-base-patch32` for zero-shot classification. CLIP's power comes from scale (400M pairs) more than architecture complexity.

---

## 13. Contrastive Training

### High-Level Concept

Contrastive training teaches a model to distinguish matching pairs from non-matching pairs within a batch. The model learns to pull matching image-text pairs together (high similarity) and push non-matching pairs apart (low similarity). The batch structure provides automatic labels — no manual annotation needed.

### The Setup

A batch contains N image-text pairs:

```
(image_0, text_0) ← match
(image_1, text_1) ← match
(image_2, text_2) ← match
...
(image_{N-1}, text_{N-1}) ← match
```

**Positive pairs:** (image_i, text_i) for all i — the N correct matches (diagonal)

**Negative pairs:** (image_i, text_j) where i≠j — the N²−N incorrect combinations (off-diagonal)

### The Similarity Matrix

Compute embeddings for all images and texts, then multiply:

```python
similarity_matrix = image_embeddings @ text_embeddings.T  # [N, N]
```

- Diagonal entries `similarity[i, i]`: correct pairs → should be HIGH
- Off-diagonal entries `similarity[i, j]`: incorrect pairs → should be LOW

### How Pulling and Pushing Works

The cross-entropy loss treats each row as a classification problem: "For image_i, which of the N texts is correct?"

The loss pushes probability mass toward the diagonal:
- Similarity with the correct text (diagonal) increases
- Similarity with incorrect texts (off-diagonal) decreases

This is the "pulling together" and "pushing apart" mechanism.

### Why Batch Size Matters

Larger batch = more negatives = harder task = better representations

- Batch 32: Each image must distinguish from 31 wrong texts — relatively easy
- Batch 4,096: Each image must distinguish from 4,095 wrong texts — harder
- Batch 32,768 (what CLIP used): Each image must distinguish from 32,767 wrong texts — very hard, forces fine-grained feature learning

Larger batches require more memory and compute. Distributed training across many GPUs enables massive batch sizes.

### Notebook Connection

Lab 02b builds the contrastive training loop. You observe the similarity matrix diagonal getting brighter (higher values) during training as the model learns.

---

## 14. Cosine Similarity versus Dot Product

### High-Level Concept

Cosine similarity measures the angle between two vectors, ignoring their lengths. Dot product measures both angle and length combined. For comparing embeddings, cosine similarity is preferred because semantic similarity should not depend on arbitrary vector magnitudes.

### Cosine Similarity

Formula: `cosine(A, B) = (A · B) / (‖A‖ × ‖B‖)`

Range: [-1, +1]

- +1: Vectors point in exactly the same direction (maximum similarity)
- 0: Vectors are perpendicular (unrelated)
- -1: Vectors point in opposite directions (maximum dissimilarity)

### Dot Product

Formula: `A · B = Σ(Aᵢ × Bᵢ) = ‖A‖ × ‖B‖ × cos(θ)`

The problem: A very long vector has high dot product with everything, even with vectors pointing in different directions. Two semantically similar items might have low dot product just because one embedding happens to be shorter.

### Why Cosine for Embeddings

Embedding magnitude is arbitrary — it depends on model architecture, layer norms, and training dynamics, not semantic content. Only the direction encodes meaning.

"Dog" and "puppy" should have high similarity regardless of whether their embedding vectors happen to be long or short.

### Practical Implementation

L2-normalize embeddings to unit length once:

```python
A_normalized = A / torch.norm(A, dim=-1, keepdim=True)
B_normalized = B / torch.norm(B, dim=-1, keepdim=True)
```

After normalization, dot product equals cosine similarity:

```python
similarity = A_normalized @ B_normalized.T
```

This is faster than computing cosine directly because matrix multiplication is highly optimized.

### Notebook Connection

All similarity computations throughout the labs use normalized embeddings. Contrastive loss, vector database search, and zero-shot classification all rely on this.

---

## 15. Ground Truth Labels in Contrastive Learning

### High-Level Concept

In contrastive learning, labels are simply the index positions of matching pairs within the batch — no manual class labels like "dog" or "cat" are required. The batch structure itself provides supervision.

### The Key Insight

When computing cross-entropy loss on the similarity matrix:

**Question:** "For image_i, which text is correct?"

**Answer:** text_i — the text at the SAME index

Therefore: `labels = [0, 1, 2, ..., N-1]`

### Implementation

```python
batch_size = images.shape[0]
labels = torch.arange(batch_size, device=device)  # [0, 1, 2, ..., N-1]
```

### How Cross-Entropy Uses These Labels

For row i of the similarity matrix (all similarities for image_i with each text), the target is index i. The loss function pushes `similarity[i, i]` to be the highest value in that row.

### Why This Is Self-Supervised

No human ever labeled these images:

- Data was collected as image-caption pairs from the web
- image_i was paired with text_i because they appeared together on a webpage
- The co-occurrence IS the label

This scales to 400 million pairs without any manual annotation cost.

### Notebook Connection

Lab 02b uses `torch.arange(batch_size)` for labels. Common student confusion: these are pair indices within the batch, not category identifiers like "dog" or "cat."

---

## 16. Information Noise Contrastive Estimation (InfoNCE) Loss

### High-Level Concept

The loss function for contrastive learning is cross-entropy over the similarity matrix, applied symmetrically for both image-to-text and text-to-image directions. Temperature scaling controls how peaked or spread out the probability distribution is.

### Step-by-Step Computation

**Step 1:** Encode images and texts, then L2-normalize:
```python
image_embeddings = F.normalize(image_encoder(images), dim=-1)  # [N, 512]
text_embeddings = F.normalize(text_encoder(texts), dim=-1)      # [N, 512]
```

**Step 2:** Compute similarity matrix:
```python
logits = image_embeddings @ text_embeddings.T  # [N, N]
```

**Step 3:** Apply temperature scaling:
```python
logits = logits / temperature
```

**Step 4:** Create labels (pair indices):
```python
labels = torch.arange(N, device=device)
```

**Step 5:** Compute loss in both directions:
```python
loss_i2t = F.cross_entropy(logits, labels)      # image-to-text
loss_t2i = F.cross_entropy(logits.T, labels)    # text-to-image
```

**Step 6:** Average:
```python
loss = (loss_i2t + loss_t2i) / 2
```

### Temperature Parameter

Temperature (τ) controls the sharpness of the softmax distribution:

**Low temperature (e.g., τ = 0.07, CLIP default):**
Dividing by a small number makes logits larger. Softmax becomes very peaked — the model is very confident. Stronger gradients on hard negatives.

**High temperature (e.g., τ = 1.0):**
Logits stay moderate. Softer distribution — less confident. Weaker gradients.

Temperature is often learned during training, stored in log space for numerical stability.

### Why Symmetric Loss

Computing loss in both directions ensures:
- Image encoder learns to distinguish texts
- Text encoder learns to distinguish images
- Both encoders receive equal gradient signal

Without symmetry, one encoder might dominate training.

### Notebook Connection

Lab 02b implements this exact loss function. Temperature is crucial — too high loses training signal, too low causes numerical instability.

---

## 17. repeat_interleave versus repeat in PyTorch

### High-Level Concept

Both operations duplicate tensor elements, but in different patterns. Using the wrong one scrambles pair alignment in contrastive learning, causing training to fail.

### repeat_interleave (Stutter Pattern)

Each element repeats multiple times before moving to the next:

```python
x = torch.tensor([A, B, C])
x.repeat_interleave(3)
# Result: [A, A, A, B, B, B, C, C, C]
```

Think of it as "stuttering": A-A-A, B-B-B, C-C-C

### repeat (Echo Pattern)

The entire tensor tiles multiple times:

```python
x = torch.tensor([A, B, C])
x.repeat(3)
# Result: [A, B, C, A, B, C, A, B, C]
```

Think of it as "echoing": ABC, ABC, ABC

### When to Use Each

Scenario: 4 images, each needs to compare against 3 texts (12 total pairings)

**Images — use repeat_interleave (stutter):**
```python
images.repeat_interleave(3)
# [I₀, I₀, I₀, I₁, I₁, I₁, I₂, I₂, I₂, I₃, I₃, I₃]
```

**Texts — use repeat (echo):**
```python
texts.repeat(4)
# [T₀, T₁, T₂, T₀, T₁, T₂, T₀, T₁, T₂, T₀, T₁, T₂]
```

Now position k in images aligns with position k in texts:
- Position 0: I₀ with T₀
- Position 1: I₀ with T₁
- Position 2: I₀ with T₂
- Position 3: I₁ with T₀
- And so on...

### The Common Bug

Using repeat where repeat_interleave is needed (or vice versa) scrambles your positive pairs. The model learns nothing meaningful because "correct" pairs are randomly assigned.

### Notebook Connection

Lab 02b uses these operations during batch preparation. Wrong operation = misaligned pairs = broken training.

---

# Part 4: Retrieval and Vector Databases

---

## 18. Vector Databases

### High-Level Concept

A vector database stores embedding vectors and provides fast approximate nearest neighbor search. When you have millions of embeddings and need to find the most similar ones to a query, brute-force comparison is too slow. Vector databases use specialized data structures to achieve near-instant search.

### The Problem

Imagine 1 million document embeddings, each with 768 dimensions. A user query needs the 10 most similar documents.

Brute force requires 1 million dot products per query. At 1 microsecond each, that's 1 second per query. For real-time applications with many concurrent users, this is impractical.

### Hierarchical Navigable Small World (HNSW) Algorithm

The most popular approximate nearest neighbor algorithm builds a multi-layer graph:

```
Layer 3 (sparse):    A ─────────────────── B
Layer 2:             A ────── C ────── B
Layer 1:             A ── D ── C ── E ── B
Layer 0 (dense):     A─D─F─G─C─H─I─E─J─B
```

**Search process:**
1. Start at top layer (sparse, few nodes)
2. Greedily move toward query vector
3. Drop to next layer, continue moving toward query
4. Repeat until bottom layer (dense, many nodes)
5. Refine search in local neighborhood

**Complexity:** O(log N) instead of O(N) — massive speedup for large databases.

### Key Parameters

**M:** Number of edges per node. Higher M = more accurate but more memory.

**ef_construction:** Search depth during index building. Higher = better index quality but slower build time.

**ef_search:** Search depth during queries. Higher = more accurate results but slower queries.

### Common Vector Database Tools

- **Milvus:** Open source, distributed, highly scalable
- **Pinecone:** Managed service, easy to use
- **Weaviate:** Supports hybrid search (vector + keyword)
- **Qdrant:** Written in Rust, very fast
- **pgvector:** PostgreSQL extension, integrates with existing databases

### Notebook Connection

Lab 04a uses Milvus to store video frame embeddings and search with text queries. The database handles similarity search internally using HNSW.

---

## 19. Retrieval Augmented Generation (RAG)

### High-Level Concept

Retrieval Augmented Generation combines information retrieval with text generation. First, fetch relevant documents from a knowledge base. Then, include those documents as context when the Large Language Model (LLM) generates an answer. This reduces hallucination, enables access to private or recent information, and allows citing sources.

### The Pipeline

**Offline Phase (Indexing):**
1. Split documents into chunks (200-500 tokens each)
2. Embed each chunk using an embedding model (e.g., BGE, E5)
3. Store embeddings in vector database with metadata (source, page number)

**Online Phase (Query):**
1. User asks a question
2. Embed the question using the same embedding model
3. Search vector database for top-k most similar chunks
4. Build a prompt with retrieved context plus the question
5. LLM generates answer grounded in the context

### Prompt Template Example

```
Based on the following context, answer the question.

Context:
{chunk_1}
{chunk_2}
{chunk_3}

Question: {user_question}

Answer:
```

### Why RAG Helps

**Without RAG:**
- LLM only knows its training data (with a knowledge cutoff date)
- Hallucinates when uncertain
- Cannot access private or recent information
- Cannot cite specific sources

**With RAG:**
- Access to up-to-date documents
- Grounded in retrieved context
- Can cite specific sources
- Works on proprietary data the LLM was never trained on

### Design Decisions

- **Chunk size:** Too large dilutes relevance; too small loses context
- **Overlap:** Chunks typically overlap by 50-100 tokens to preserve context at boundaries
- **Embedding model:** Must match between indexing and query time
- **Top-k:** How many chunks to retrieve (more context vs. more noise)
- **Re-ranking:** Optional step to re-score retrieved chunks for higher precision

### Notebook Connection

Lab 03b builds the full RAG pipeline: extract PDF text, chunk, embed, store, retrieve, augment prompt, generate answer.

---

# Part 5: Vision-Language Models

---

## 20. Vision-Language Model Architecture

### High-Level Concept

Vision-Language Models (VLMs) accept both images and text as input, generating text output with visual grounding. They work by converting image features into "visual tokens" that the Large Language Model processes alongside text tokens — like teaching the LLM a new visual language.

### Architecture Components

**Vision Encoder:**
A pretrained Convolutional Neural Network or Vision Transformer extracts features from the input image. Output is typically a sequence of patch embeddings, e.g., `[256 patches, 1024 dimensions]`.

**Projection Layer:**
A learned transformation mapping vision features to the LLM's embedding dimension. If vision encoder outputs 1024 dimensions and LLM expects 4096, the projector bridges this gap.

**Large Language Model:**
Processes the combined visual and text tokens to generate output. The visual tokens are treated like words in a foreign language the LLM has learned to understand.

### Input Sequence Structure

```
[IMG_1, IMG_2, ..., IMG_256, User, :, What, is, this, ?]
└────── 256 visual tokens ──────┘ └──── text tokens ────┘
```

The LLM's attention mechanism can attend across both visual and text tokens, learning to ground responses in image content.

### Training Approach

**Stage 1 — Feature Alignment:**
- Train only the projection layer
- Freeze vision encoder and LLM
- Use image-caption pairs
- Goal: teach projector to "speak" the LLM's language

**Stage 2 — Instruction Tuning:**
- Train projection layer and LLM (often with LoRA for efficiency)
- Keep vision encoder frozen
- Use instruction-following conversations with images
- Goal: teach the system to follow complex visual instructions

### Notebook Connection

Lab 03a builds VLM components. You observe how visual tokens are concatenated with text tokens before LLM processing.

---

## 21. Cross-Modal Projection

### High-Level Concept

Cross-modal projection is the learned transformation mapping vision encoder outputs into the Large Language Model's embedding space. It solves two problems: dimension mismatch (different vector sizes) and semantic mismatch (visual patterns vs. word-like meanings).

### The Dimension Problem

Vision encoder might output 1024-dimensional vectors. LLM might expect 4096-dimensional embeddings. A simple linear layer can change dimensions:

```python
projection = nn.Linear(1024, 4096)
```

But dimension alignment alone is insufficient.

### The Semantic Problem

Vision and language embedding spaces have different structures:

- In vision space: two photos of different dogs might be far apart (different poses, lighting)
- In language space: "dog" is near "puppy", "canine", "pet"

A linear transformation can only rotate, scale, and translate — it preserves straight lines and ratios. But mapping vision topology to language topology requires warping the space.

### Why Multi-Layer Perceptron Beats Linear

A Multi-Layer Perceptron (MLP) with nonlinear activation can bend, fold, and stretch the space:

```python
projection = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),              # Nonlinearity enables space warping
    nn.Linear(4096, 4096)
)
```

The nonlinearity (GELU, ReLU, etc.) breaks linearity, allowing the transformation to map different topologies onto each other.

### Training Goal

After training, the projector should map:
- Visual features of a dog → near the LLM's embedding of "dog"
- Visual features of a red car → near the LLM's embedding of "red car"

The visual tokens become interpretable because they land in semantically meaningful regions of the LLM's embedding space.

### Parameter Count

For vision_dim=1024, llm_dim=4096:
- First linear: 1024 × 4096 ≈ 4.2 million parameters
- Second linear: 4096 × 4096 ≈ 16.8 million parameters
- Total: approximately 21 million parameters

This is tiny compared to the frozen vision encoder (400+ million) and LLM (7+ billion), making training efficient.

### Notebook Connection

Lab 03a trains the projection layer to minimize distance between projected visual features and corresponding text embeddings.

---

## 22. Large Language and Vision Assistant (LLaVA) Architecture

### High-Level Concept

LLaVA is a reference Vision-Language Model architecture using a frozen CLIP vision encoder, a small trainable projector, and a Vicuna LLM. Its two-stage training separates feature alignment from instruction tuning, achieving high quality with minimal compute.

### Components

**Vision Encoder: CLIP ViT-L/14 (frozen throughout)**
- 24 transformer layers
- Processes 14×14 pixel patches (224÷14 = 16 patches per side = 256 total patches)
- Output: 257 tokens × 1024 dimensions (256 patches + 1 CLS token)

**Projector: 2-Layer MLP (trained)**
- Maps 1024 dimensions to 4096 dimensions
- Approximately 21 million parameters
- The only trainable component in Stage 1

**LLM: Vicuna-7B or 13B**
- LLaMA fine-tuned on conversation data
- 32 transformer layers (7B version)
- 4096 embedding dimension
- Frozen in Stage 1, fine-tuned with LoRA in Stage 2

### Stage 1: Feature Alignment

**What trains:** Only the projector

**What freezes:** Vision encoder and LLM

**Data:** 558,000 image-caption pairs (filtered from CC3M dataset)

**Goal:** Teach the projector to output vectors the LLM can interpret as meaningful visual content

**Compute:** Approximately 4 hours on 8 A100 GPUs

**Result:** Model can describe images but doesn't follow instructions well

### Stage 2: Visual Instruction Tuning

**What trains:** Projector + LLM (using LoRA for efficiency)

**What freezes:** Vision encoder

**Data:** 158,000 instruction-following conversations generated by GPT-4 from images

**Goal:** Teach the combined system to follow complex visual instructions

**Result:** Model engages in multi-turn conversations about images, answers detailed questions, follows nuanced instructions

### Why Two Stages

**Stage 1 alone:** Can describe images but responses are bland, doesn't follow instructions well

**Stage 2 alone:** Unstable training because randomly initialized projector outputs garbage vectors that confuse the LLM

**Both stages:** Projector learns basics first (Stage 1), then full system is refined (Stage 2). More stable and efficient than end-to-end training.

### Notebook Connection

Slides detail the LLaVA architecture. Lab 03a replicates the pattern at smaller scale, demonstrating that frozen pretrained components plus small trainable connector enable efficient VLM development.

---

# Part 6: Document Processing

---

## 23. Document Chunking Strategies

### High-Level Concept

Before documents can be searched with Retrieval Augmented Generation, they must be split into smaller chunks. How you chunk dramatically affects retrieval quality — chunks too large dilute relevance, chunks too small lose context.

### Fixed-Size Chunking

Split every N characters (e.g., 1000 characters) with some overlap (e.g., 200 characters):

```python
def fixed_chunk(text, size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i+size])
    return chunks
```

**Problem:** May split mid-sentence or mid-table:
```
Chunk 1: "...The CEO stated that profitability"
Chunk 2: "will increase by 40% next quarter..."
```

The meaning is broken across chunks.

### Recursive Chunking

Try to split at natural boundaries, falling back to smaller units:

1. First try to split at paragraph breaks (`\n\n`)
2. If chunks still too big, split at line breaks (`\n`)
3. If still too big, split at sentences (`. `)
4. If still too big, split at words (` `)
5. Last resort: split at characters

This keeps semantic units intact when possible.

### Semantic Chunking

Use embeddings to find natural topic boundaries:

1. Embed each sentence
2. Compute similarity between adjacent sentences
3. Where similarity drops significantly, a topic change likely occurred
4. Split at these low-similarity boundaries

Most sophisticated approach for pure text, but computationally expensive.

### Layout-Aware Chunking

Respect document structure rather than just character counts:

- Keep tables as complete units (never split mid-row)
- Group headers with their following paragraphs
- Preserve list items together
- Keep figures with their captions

Best for structured documents with mixed content types. Libraries like `unstructured` detect and preserve these elements.

### Size Guidelines

- Most embedding models have 512 token context maximum
- Sweet spot: 200-500 tokens per chunk
- Overlap: 50-100 tokens to preserve context at boundaries

### Notebook Connection

Lab 03b uses `unstructured` for layout-aware chunking. You compare strategies and observe that layout-aware significantly outperforms fixed-size for structured documents.

---

## 24. Page Element Identification

### High-Level Concept

Before processing a document, identify what type of content is in each region: titles, body paragraphs, tables, figures, lists, headers, footers. This enables appropriate handling for each element type rather than treating everything as undifferentiated text.

### Element Types

- **Title:** Document or section headers
- **NarrativeText:** Body paragraphs
- **ListItem:** Bulleted or numbered list items
- **Table:** Structured rows and columns
- **Figure:** Images, charts, diagrams
- **Caption:** Descriptions for figures or tables
- **Header/Footer:** Page metadata (page numbers, document title repeated)
- **Formula:** Mathematical equations

### Detection Pipeline

1. **Input:** PDF or page image
2. **Layout Detection:** A model (YOLOX, LayoutLM, Detectron2) identifies bounding boxes and classifies each region (table, figure, text block)
3. **OCR:** Optical Character Recognition extracts text from each region (using Tesseract, DocTR, etc.)
4. **Reading Order:** Regions sorted top-to-bottom, left-to-right
5. **Output:** Structured elements with type labels and extracted text

### Special Handling by Element Type

**Tables:**
Keep as complete units — never split mid-row. Options:
- Preserve as markdown
- Convert to structured format (dataframe)
- Describe in natural language for better embedding

**Figures:**
Require image understanding:
- Caption with a Vision-Language Model
- At minimum, include the caption text

**Lists:**
Preserve item structure:
- Each item is a logical unit
- Include the list header for context

**Headers/Footers:**
Often filtered out to reduce noise (page numbers, repeated document titles don't help retrieval).

### Tools

- **unstructured:** Open source Python library
- **LayoutLM/LayoutLMv3:** Microsoft, state of the art
- **DocTR:** OCR plus layout detection
- **NVIDIA DALI:** GPU-accelerated processing

### Notebook Connection

Lab 03b uses `unstructured` to automatically tag elements. You filter by type for specialized processing — tables preserved whole while narrative text is chunked normally.

---

# Part 7: Video Search and Summarization

---

## 25. Video Search and Summarization Applications

### High-Level Concept

Video Search and Summarization (VSS) enables natural language queries over large video archives, returning relevant clips with timestamps and summaries. Instead of humans watching hundreds of hours of footage, the system finds relevant moments in seconds.

### The Value Proposition

**Without VSS:**
- Human watches 100 hours of surveillance footage
- Takes days of work
- Expensive ($15-30/hour × days of work)
- Error-prone (human attention fades over time)
- Does not scale

**With VSS:**
- Query in natural language: "Show all forklift near-misses"
- Results in seconds
- Timestamps and text summaries returned
- Human reviews only relevant clips (minutes instead of hours)

### Application Domains

**Surveillance and Safety:**
- "Show all Personal Protective Equipment violations this week"
- "Find vehicles entering the parking lot after 10 PM"
- "Locate the red truck from yesterday's incident"

**Media Production:**
- "Find all clips of the CEO mentioning quarterly earnings"
- "Locate B-roll footage of city skylines"
- "Find interviews discussing the product launch"

**Sports Analytics:**
- "Show all three-point shots in the fourth quarter"
- "Find defensive formations against zone offense"
- "Locate player injury incidents this season"

**Education:**
- "Find lecture segments explaining neural networks"
- "Locate lab demonstrations of titration"

**Compliance and Legal:**
- "Find all conversations where pricing was discussed"
- "Locate instances of policy violations"

### Scale Challenge

1000 cameras × 24 hours × 30 days = 720,000 hours per month

Manual review is physically impossible at this scale.

### Notebook Connection

Lab 04a queries a traffic video database. You experience finding relevant moments from hours of footage in seconds.

---

## 26. NVIDIA AI Blueprint for Video Search and Summarization

### High-Level Concept

The NVIDIA AI Blueprint provides a complete reference architecture for Video Search and Summarization, integrating video processing, computer vision, vision-language models, retrieval systems, and large language models. Each component is packaged as a NVIDIA Inference Microservice (NIM) for optimized GPU deployment.

### Architecture Layers

**Layer 1 — Data Plane:**

The DeepStream Software Development Kit handles video ingestion:
- Decode video formats (H.264, H.265)
- Extract frames
- GPU-accelerated preprocessing
- Chunk videos into 10-60 second segments
- Sample keyframes (1-5 per chunk)

**Layer 2 — Computer Vision NIMs:**

Multiple detection and tracking models:
- Object detection: Find people, vehicles, objects in each frame
- Tracking: Maintain identity across frames (person_1 stays person_1 throughout video)
- Segmentation: Pixel-level scene understanding (foreground vs. background)

**Layer 3 — Vision-Language Model NIMs:**

Multimodal understanding:
- Frame captioning: Generate natural language descriptions ("A forklift moves through the warehouse")
- CLIP embedding: Create 512-dimensional vectors for semantic similarity search
- Optical Character Recognition (OCR): Extract visible text (signs, labels, documents)

**Layer 4 — Storage:**

Multiple database types:
- Vector database (Milvus): Store embeddings for fast similarity search
- Graph database (Neo4j): Store entity relationships (person_1 INTERACTED_WITH forklift_2)
- Metadata store: Video IDs, timestamps, camera IDs, captions

**Layer 5 — NeMo Retriever:**

Orchestrates retrieval across backends:
- Query embedding
- Metadata filtering
- Hybrid search (vector + graph)
- Re-ranking for precision

**Layer 6 — Large Language Model NIM:**

Generates final output:
- Answer generation grounded in retrieved context
- Summarization across multiple retrieved chunks

### Output

Natural language response with timestamps and links to relevant video clips.

### Notebook Connection

Lab 04a interacts with Blueprint API endpoints. The complexity of a 5+ model pipeline is abstracted into simple natural language queries.

---

## 27. Video Search and Summarization Pipeline

### High-Level Concept

The operational pipeline: video → chunks → keyframes → embeddings and captions → storage → retrieval → LLM answer. Each stage has design choices affecting latency, quality, and cost.

### Stage 1: Video Ingestion

Split videos into manageable chunks:
- Duration: 10-60 seconds per chunk
- Shorter chunks = more precise retrieval but more storage
- Sweet spot: 15-30 seconds for most applications

### Stage 2: Keyframe Sampling

Extract representative frames from each chunk:
- 1 frame per chunk: Fast but may miss events
- 5 frames per chunk: Good balance of coverage and cost
- Every frame: Expensive, diminishing returns

### Stage 3: Embedding

Encode keyframes using CLIP-style visual encoder:
- Output: 512-dimensional vector per frame
- Options for combining frames within chunk:
  - Average embeddings (simple, loses temporal info)
  - Concatenate (preserves order, larger vectors)
  - Temporal pooling (best quality, most complex)

### Stage 4: Captioning

Generate text description for each chunk using Vision-Language Model:
- Example: "A worker in a yellow vest walks past a forklift"
- Enables text-based search even when visual search misses
- Provides human-readable context for retrieved results

### Stage 5: Indexing

Store in vector database with metadata:
- Embedding vector
- Video ID, start time, end time
- Caption text
- Camera ID, location
- Create HNSW index for fast search

### Stage 6: Query Processing

When user queries:
1. Embed query using same CLIP text encoder
2. Search vector database for similar chunk embeddings
3. Optionally apply metadata filters (camera, date, time)
4. Re-rank for precision

### Stage 7: Response Generation

Pass retrieved chunk captions and user query to LLM:
- Generate natural language answer
- Include timestamps
- Cite specific video clips

### Notebook Connection

Lab 04a implements the retrieval and summarization loop.

---

## 28. Vector-Based Retrieval (Bi-Encoder Approach)

### High-Level Concept

Embed queries and documents independently using the same model, then retrieve by similarity search. Fast and scalable because all document embeddings can be precomputed.

### How It Works

1. Embed user query: `query_vec = encoder(query)` → [768 dimensions]
2. Search vector database using HNSW for top-k similar chunks
3. Return chunks ranked by cosine similarity

### Bi-Encoder versus Cross-Encoder

**Bi-Encoder (used for retrieval):**
- Encode query and document separately
- Similarity = dot product of embeddings
- Fast: can precompute document embeddings
- Used for initial retrieval of many candidates

**Cross-Encoder (used for re-ranking):**
- Encode query AND document together: `[CLS] query [SEP] document [SEP]`
- Scores the pair jointly with full attention between query and document
- More accurate but slower (cannot precompute)
- Used to re-rank top candidates from bi-encoder

**Typical pattern:** Bi-encoder retrieves top-100 candidates → Cross-encoder re-ranks to find top-10.

### Strengths of Vector Retrieval

- Semantic matching: "safety incidents" finds "near-miss events" without shared keywords
- No keyword engineering needed
- Works across paraphrases and synonyms
- Fast: query embedding + approximate nearest neighbor search

### Limitations of Vector Retrieval

- May miss exact identifier matches: "Show clip 47" might return visually similar clips instead of exact clip 47
- No relational reasoning: "Who talked to the person in red?" requires entity tracking, not just similarity
- No temporal reasoning: "What happened after the alarm?" needs time understanding

### Notebook Connection

Labs 04a and 04b use vector search as the foundation. You observe its strengths for semantic queries and limitations for relationship-based questions.

---

## 29. Graph-Based Retrieval

### High-Level Concept

Use knowledge graphs to answer relationship-heavy or multi-hop questions by traversing edges between entities. Solves problems that vector similarity cannot.

### When Graph Retrieval Is Needed

Query: "Which companies has the CEO of Acme previously worked for?"

**Vector search finds:**
- Chunks mentioning "CEO"
- Chunks mentioning "Acme"
- Chunks mentioning "companies"

**But cannot chain:** CEO of Acme → identify that specific person → find their previous employers

**Graph query traverses directly:**
```cypher
MATCH (c:Company {name: "Acme"})<-[:CEO_OF]-(person:Person)
MATCH (person)-[:PREVIOUSLY_WORKED_AT]->(previous:Company)
RETURN previous.name
```

### Knowledge Graph Structure

**Nodes (Entities):**
- Person: {name, role}
- Company: {name, industry}
- Location: {name, type}
- Event: {type, timestamp}
- Object: {type, id}

**Edges (Relationships):**
- (Person)-[:WORKS_AT]->(Company)
- (Person)-[:LOCATED_IN]->(Location)
- (Event)-[:INVOLVES]->(Person)
- (Person)-[:INTERACTED_WITH]->(Object)

### Building Graphs from Video

From video caption: "John Smith drives forklift near warehouse entrance"

1. Extract entities: [(John Smith, Person), (forklift, Object), (warehouse entrance, Location)]
2. Extract relationships: [(John Smith, DRIVES, forklift), (John Smith, NEAR, warehouse entrance)]
3. Create graph nodes and edges with timestamps

### Hybrid Approach

Best results combine both:
1. Vector search finds semantically relevant chunks
2. Graph traversal finds relationship-connected entities
3. Merge and deduplicate results
4. Re-rank by combined relevance
5. Generate answer with full context

### Notebook Connection

Lab 04b uses Neo4j. You visualize entity graphs from video analysis and see how graph queries answer relational questions that vector search cannot.

---

## 30. Context-Aware Retrieval

### High-Level Concept

Improve retrieval precision with metadata filtering before search, context expansion after search, and cross-encoder re-ranking. Not just "find similar" but "find relevant given constraints."

### Metadata Filtering

Query: "Show safety incidents from Camera 3 yesterday"

**Pure vector search returns:** Incidents from any camera, any date — many irrelevant results.

**Context-aware approach:**
1. Extract filters from query: camera="3", date="yesterday"
2. Apply filters to restrict search space BEFORE vector similarity
3. Vector search only within filtered subset
4. Results are precise and relevant

### Filter Types

**Temporal:** date, time_range, "last 7 days", before/after timestamps

**Spatial:** camera_id, location name, zone, building

**Categorical:** event_type, severity, status

### Context Expansion

After retrieving chunk k, also fetch neighboring chunks:
- Chunk k-1 (what happened before)
- Chunk k+1 (what happened after)

This provides temporal context — what led up to an event and what followed.

### Re-Ranking with Cross-Encoder

Initial bi-encoder retrieval returns candidates ranked by embedding similarity. A cross-encoder can re-score these more accurately:

1. Bi-encoder retrieves top-100 candidates (fast, broad recall)
2. Cross-encoder scores each candidate with the full query (slow, high precision)
3. Re-rank by cross-encoder scores
4. Return top-10

### Router Pattern

Use an LLM to parse user queries and decide retrieval strategy:
- Extract metadata filters
- Determine if query needs vector search, graph traversal, or both
- Route to appropriate retrieval pipeline

### Notebook Connection

Lab 04a passes filter arguments to the search API. You observe how filtering dramatically improves precision for specific queries.

---

## 31. Cypher Query Language

### High-Level Concept

Cypher is the declarative query language for Neo4j graph databases. It uses ASCII-art-style patterns to describe nodes and relationships. Essential for graph-based retrieval.

### Basic Syntax

**Nodes:** Enclosed in parentheses
```cypher
(variable:Label {property: value})

Examples:
(p:Person)                        -- Any person
(p:Person {name: "John"})         -- Person named John
```

**Relationships:** Enclosed in brackets with arrows
```cypher
-[r:RELATIONSHIP_TYPE]->          -- Directed relationship
-[r:RELATIONSHIP_TYPE]-           -- Either direction

Examples:
(p)-[:WORKS_AT]->(c)              -- p works at c
(a)-[:KNOWS]-(b)                  -- a and b know each other (either direction)
```

### Core Commands

**MATCH:** Find patterns in the graph
```cypher
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WHERE c.name = "Acme"
RETURN p.name
```

**CREATE:** Add new nodes and relationships
```cypher
CREATE (p:Person {name: "Jane", role: "engineer"})
```

**MERGE:** Create only if not exists (upsert)
```cypher
MERGE (c:Company {name: "Acme"})
```

**WHERE:** Filter results
```cypher
MATCH (p:Person)
WHERE p.age > 30
RETURN p
```

### Multi-Hop Queries

**Friends of friends (2 hops):**
```cypher
MATCH (me:Person {name: "Alice"})-[:KNOWS]->()-[:KNOWS]->(friend_of_friend)
WHERE friend_of_friend <> me
RETURN DISTINCT friend_of_friend.name
```

**Variable length paths (1 to 3 hops):**
```cypher
MATCH (start)-[:CONNECTED*1..3]-(end)
RETURN end.name
```

### Aggregation

```cypher
MATCH (p:Person)-[:WORKED_AT]->(c:Company)
RETURN p.name, count(c) as num_companies
ORDER BY num_companies DESC
```

### Natural Language to Cypher

LLMs can translate natural language questions into Cypher queries:

Input: "Who works at Acme?"

Output:
```cypher
MATCH (p:Person)-[:WORKS_AT]->(c:Company {name: "Acme"})
RETURN p.name
```

### Notebook Connection

Lab 04b executes Cypher queries against Neo4j. You write queries to traverse relationships and observe results.

---

## 32. NeMo Retriever

### High-Level Concept

NeMo Retriever is NVIDIA's retrieval microservice handling the full Retrieval Augmented Generation pipeline: ingestion, embedding, indexing, query processing, and hybrid retrieval. It provides a unified interface for vector and graph backends.

### Ingestion Pipeline

**Accepts:** Text, PDF, HTML, video captions

**Chunking:** Configurable strategies (fixed size, recursive, custom functions)

**Embedding:** Supports multiple models:
- E5 (various sizes)
- BGE (various sizes)
- NV-Embed (NVIDIA optimized)
- Custom models

**Output:** Vectors plus metadata stored in databases

### Indexing

**Vector database integration:**
- Milvus, Weaviate, Pinecone, others
- Creates HNSW indexes for fast approximate nearest neighbor search
- Supports incremental updates without full re-indexing

**Graph database integration:**
- Neo4j for entity relationships
- Links graph nodes to vector chunks

**Metadata indexing:**
- Filterable attributes (date, source, category)
- Enables pre-search filtering

### Query Processing

1. Parse query for keywords, entities, filters
2. Embed query using same model as documents
3. Apply metadata filters
4. Run vector search and/or graph traversal
5. Merge and deduplicate results

### Re-Ranking

Cross-encoder models score query-candidate pairs for precision:
- More accurate than bi-encoder similarity
- Applied to top-k candidates from initial retrieval

### LLM Integration

**Output formatting:** Returns ranked contexts plus metadata ready for prompt injection

**Streaming:** Supports streaming responses

**Citation tracking:** Records which chunks contributed to each answer

### Notebook Connection

Labs use NeMo Retriever functionality. The unified interface handles vector search, filtering, and result formatting for LLM input.

---

# Quick Reference Checklist

| Topic | Key Points |
|-------|------------|
| CNN Components | Kernel slides and multiplies, stride controls step size, padding preserves dimensions, pooling reduces size |
| Tensor Shapes | RGB `[B,3,H,W]`, Grayscale `[B,1,H,W]`, Point cloud `[N,3+]`, CT `[B,1,D,H,W]` |
| Audio Pipeline | Waveform → STFT → Mel scale → Log amplitude → 2D image |
| Color Modes | OpenCV loads BGR, PyTorch expects RGB, always convert explicitly |
| CT Structure | Hounsfield Units: air=-1000, water=0, bone=+1000; depth is spatial not temporal |
| LiDAR vs Camera | Camera=what (dense color), LiDAR=where (sparse depth), complementary |
| Projection | Extrinsic R,t transforms coordinates, Intrinsic K projects to pixels, divide by Z for perspective |
| Early Fusion | Concatenate input channels, modify first conv layer, single network |
| Late Fusion | Separate encoders, combine predictions only, average or vote |
| Intermediate Fusion | Separate encoders, combine features, cross-attention or concatenation |
| CLIP | Dual encoders, no cross-attention, enables precomputation, 400M training pairs |
| Contrastive Training | Pull matching pairs (diagonal), push non-matching (off-diagonal), batch size matters |
| Cosine Similarity | Normalize vectors, then dot product, ignores magnitude |
| Contrastive Labels | `torch.arange(N)` — pair indices, self-supervised |
| InfoNCE Loss | Symmetric cross-entropy, temperature controls sharpness |
| repeat vs repeat_interleave | repeat=echo (tile tensor), repeat_interleave=stutter (repeat each element) |
| Vector Database | HNSW enables O(log N) search, parameters: M, ef_construction, ef_search |
| RAG Pipeline | Chunk → Embed → Store → Retrieve → Augment prompt → Generate |
| VLM Architecture | Vision encoder → Projection → LLM, visual tokens like foreign words |
| Cross-Modal Projection | MLP can warp space, linear cannot, bridges dimension and semantic gaps |
| LLaVA | Stage 1 aligns projector only, Stage 2 instruction tunes with LoRA |
| Document Chunking | Layout-aware best, 200-500 tokens, preserve structure |
| Page Elements | Detect types, keep tables whole, caption figures |
| VSS Applications | Surveillance, sports, media, compliance — search video archives |
| NVIDIA Blueprint | Data plane → CV → VLM → Storage → Retriever → LLM |
| Vector vs Graph | Vector for semantic similarity, Graph for relationship traversal |
| Context-Aware RAG | Filter metadata before search, expand context after, re-rank for precision |
| Cypher | MATCH patterns, WHERE filters, variable-length paths with `*1..3` |
| NeMo Retriever | Unified retrieval: ingest, embed, index, search, re-rank, format for LLM |

---

*End of Study Guide*
