# Overview of all Explainable AI methods:

## Taxonomy

| Method                           | Description                                     |
|----------------------------------|-------------------------------------------------|
| Model agnostic vs model specific | Can be applied to different model architectures |
| Global vs Local                  |                                                 |
| Surrogate model                  |                                                 |
| Path attribution                 |                                                 |
| Gradient based                   |                                                 |


## Interpretable model:
- Linear models
    - Weights
    - $R^2$
- Decision trees
    - Feature importance
    - Path attribution
    - Leaf Node Analysis


## Methods for Exaplainable AI:

### **1. Partial Dependence Plots (PDPs)**
- **Purpose**: Visualizes how one or two input features affect the model's predictions, marginalizing over the other features.
- **How It Works**:
  - Fix one or two features (e.g., `x1`) and vary them across their range.
  - Keep other features constant (averaging over their distribution).
  - Plot the average model prediction for the varying features.
- **Best for**: Interpreting the global impact of features in tabular models.

---

### **2. Individual Conditional Expectation (ICE) Plot**
- **Purpose**: Similar to PDPs but visualizes the relationship between a feature and the model prediction at the level of individual data instances.
- **How It Works**:
  - For each instance, vary the target feature while keeping other features constant.
  - Plot the prediction for each instance separately.
- **Key Difference from PDP**: PDP shows global trends, while ICE highlights heterogeneity in feature effects.

---

### **3. Sliding Window Occlusion**
- **Purpose**: Identifies important regions in images by systematically occluding parts of the image and observing how the model's prediction changes.
- **How It Works**:
  - Slide a fixed-size window (e.g., `15x15`) across the image.
  - Replace the pixels in the window with a baseline value (e.g., black or mean pixel value).
  - Measure the change in prediction for each occluded region.
- **Best for**: Localizing regions critical for the model's decision.

---

### **4. Adaptive Occlusion**
- **Purpose**: Similar to Sliding Window Occlusion but uses adaptive window sizes and positions to refine important regions.
- **How It Works**:
  - Starts with a coarse grid for occlusion.
  - Refines the occlusion process iteratively, focusing on the regions that have the most significant impact on the predictions.
- **Benefit**: Reduces computational cost while providing more focused explanations.

---

### **5. Randomized Occlusion**
- **Purpose**: Tests model robustness by randomly occluding parts of the input to observe prediction stability.
- **How It Works**:
  - Randomly select regions of the input (images, text, etc.) to occlude.
  - Measure prediction changes and identify sensitive regions.
- **Benefit**: Useful for analyzing sensitivity and robustness in a stochastic manner.

---

### **6. Class Activation Mapping (CAM)**
- **Purpose**: Highlights regions of an image that contribute to a specific class prediction in convolutional neural networks (CNNs).
- **How It Works**:
  - Uses the output of the last convolutional layer.
  - Applies a weighted average of feature maps, with weights derived from the fully connected layer for the target class.
- **Limitation**: Requires model architecture modifications (e.g., global average pooling before the fully connected layer).

---

### **7. Gradient-Weighted Class Activation Mapping (Grad-CAM)**
- **Purpose**: Extends CAM to work with pre-trained models and localizes important regions in an image for a prediction.
- **How It Works**:
  - Computes gradients of the target class score with respect to feature maps in the last convolutional layer.
  - Uses these gradients to weigh the feature maps, highlighting influential regions.
- **Benefit**: Does not require architecture changes and works with a wide range of CNNs.

---

### **8. Guided Grad-CAM**
- **Purpose**: Combines Grad-CAM with Guided Backpropagation to provide more focused visual explanations.
- **How It Works**:
  - Grad-CAM highlights important regions for the target class.
  - Guided Backpropagation refines the explanation to focus on features relevant to the prediction.
- **Best for**: Producing high-resolution and class-discriminative saliency maps.

---

### **9. Gradient x Input**
- **Purpose**: Evaluates the importance of each input feature by combining the feature's gradient with its value.
- **How It Works**:
  - Multiply the gradient of the prediction with respect to the input by the input values.
  - Highlights the sensitivity of the prediction to small changes in the input.
- **Best for**: Understanding linear effects of inputs on predictions.

---

### **10. Integrated Gradients**
- **Purpose**: Provides a path-integrated measure of feature importance.
- **How It Works**:
  - Calculates the average gradients of the model output along a path from a baseline (e.g., zero) to the input.
  - Integrates these gradients to compute feature attribution.
- **Benefit**: Captures both linear and non-linear relationships, satisfying desirable properties like sensitivity and implementation invariance.

---

### **11. Explainable Region Aggregation for Images (XRAI)**
- **Purpose**: Improves upon occlusion methods by identifying contiguous, meaningful regions instead of individual pixels.
- **How It Works**:
  - Aggregates explanations over image regions (e.g., superpixels) instead of sliding windows or gradients.
  - Produces a heatmap indicating which regions are most influential for the prediction.
- **Benefit**: More interpretable and robust to noise compared to pixel-wise methods.

---

### Comparison Summary

| **Method**                 | **Best For**                       | **Data Type**   | **Global/Local**  | **Key Strength**                         |
|----------------------------|------------------------------------|-----------------|-------------------|------------------------------------------|
| PDP                        | Global trends                     | Tabular         | Global            | Simplifies feature impact visualization. |
| ICE                        | Instance-specific feature effects | Tabular         | Local             | Captures heterogeneity of feature impact.|
| Sliding Window Occlusion   | Image region importance           | Images          | Local             | Simple, intuitive for vision tasks.      |
| Adaptive Occlusion         | Region importance refinement      | Images          | Local             | Reduces computational cost.              |
| Randomized Occlusion       | Robustness testing                | Any             | Local             | Tests model sensitivity to occlusions.   |
| CAM                        | Class localization                | Images          | Local             | Intuitive; class-specific localization.  |
| Grad-CAM                   | Class localization for CNNs       | Images          | Local             | Works with any CNN architecture.         |
| Guided Grad-CAM            | High-resolution class maps        | Images          | Local             | Fine-grained and class-specific maps.    |
| Gradient x Input           | Input-level feature attribution   | Any             | Local             | Linear relationships, fast computation.  |
| Integrated Gradients       | Path-integrated feature attribution | Any           | Local             | Captures non-linear relationships.       |
| XRAI                       | Aggregated region importance      | Images          | Local             | Contiguous, interpretable regions.       |

Let me know if you'd like further details or examples of these methods!
