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

- **Purpose**: Visualizes how one or two input features affect the model's predictions, marginalizing over the other
  features.
- **How It Works**:
    - Fix one or two features (e.g., `x1`) and vary them across their range.
    - Keep other features constant (averaging over their distribution).
    - Plot the average model prediction for the varying features.
- **Best for**: Interpreting the global impact of features in tabular models.

---

### **2. Individual Conditional Expectation (ICE) Plot**

- **Purpose**: Similar to PDPs but visualizes the relationship between a feature and the model prediction at the level
  of individual data instances.
- **How It Works**:
    - For each instance, vary the target feature while keeping other features constant.
    - Plot the prediction for each instance separately.
- **Key Difference from PDP**: PDP shows global trends, while ICE highlights heterogeneity in feature effects.

---

### **3. Sliding Window Occlusion**

- **Purpose**: Identifies important regions in images by systematically occluding parts of the image and observing how
  the model's prediction changes.
- **How It Works**:
    - Slide a fixed-size window (e.g., `15x15`) across the image.
    - Replace the pixels in the window with a baseline value (e.g., black or mean pixel value).
    - Measure the change in prediction for each occluded region.
- **Best for**: Localizing regions critical for the model's decision.

---

### **4. Adaptive Occlusion**

- **Purpose**: Similar to Sliding Window Occlusion but uses adaptive window sizes and positions to refine important
  regions.
- **How It Works**:
    - Starts with a coarse grid for occlusion.
    - Refines the occlusion process iteratively, focusing on the regions that have the most significant impact on the
      predictions.
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

- **Purpose**: Highlights regions of an image that contribute to a specific class prediction in convolutional neural
  networks (CNNs).
- **How It Works**:
    - Uses the output of the last convolutional layer.
    - Applies a weighted average of feature maps, with weights derived from the fully connected layer for the target
      class.
- **Limitation**: Requires model architecture modifications (e.g., global average pooling before the fully connected
  layer).

##### In more details:

Class Activation Maps (CAMs) are a popular explainability method used in Convolutional Neural Networks (CNNs) to
visualize the regions of an image that the network focuses on while predicting a specific class. CAMs help in
interpreting which parts of the image contribute most to the network's classification decision.

###### **How CAMs Work**

1. **Feature Maps in CNNs**:
    - In a CNN, intermediate layers (e.g., convolutional layers) extract spatial feature maps from an image.
    - These feature maps contain spatial information about various regions of the image.

2. **Global Average Pooling (GAP)**:
    - After the convolutional layers, CNN architectures often use a **Global Average Pooling (GAP)** layer to reduce the
      spatial dimensions of feature maps, creating a single scalar value for each feature map.
    - These scalars are then passed to the fully connected layers to compute class scores.

3. **Class-Weighted Sum**:
    - For each class, the fully connected layer uses weights to assign importance to each feature map. These weights
      indicate how much each feature map contributes to a particular class.

4. **Generating the CAM**:
    - The CAM is computed as a **weighted sum** of the feature maps from the last convolutional layer, where the weights
      are the learned class-specific weights from the fully connected layer.

   Mathematically:
   $M_c(x, y) = \sum_k w_k^c f_k(x, y)$
    - $M_c(x, y)$: Class activation map for class $c$ at spatial location $(x, y)$.
    - $w_k^c$: Weight for class $c$ corresponding to feature map $k$.
    - $f_k(x, y)$: Activation of the $k$-th feature map at location $(x, y)$.

   The CAM highlights regions in the input image that contribute most to the classification decision for class $c$.

###### **Steps to Generate CAMs**

1. **Obtain the Last Convolutional Feature Maps**:
    - Extract the feature maps from the last convolutional layer of the CNN for a given input image.

2. **Retrieve Class-Specific Weights**:
    - Get the weights corresponding to the predicted class from the fully connected layer.

3. **Compute Weighted Sum**:
    - Multiply each feature map by its corresponding weight and sum them to generate the class activation map.

4. **Overlay the CAM on the Input Image**:
    - Resize the CAM to match the input image size.
    - Overlay it on the original image to visualize which regions were important for the prediction.

###### **Example Use Case**

Suppose you input an image of a cat into a CNN trained on ImageNet. The CAM highlights the regions in the image (e.g.,
the cat's face or body) that contributed most to the prediction "cat."

###### **Limitations of CAMs**

1. **Architectural Dependency**:
    - CAMs require the network to use a GAP layer before the fully connected layer. This limits their applicability to
      certain architectures.

2. **Resolution**:
    - CAMs are often low-resolution because they are derived from the last convolutional layer. This can make the
      highlighted regions blurry.

###### **Applications of CAMs**

- **Explainability**: Understanding why a model made a particular prediction.
- **Error Analysis**: Identifying where the model might be focusing incorrectly.
- **Weak Supervision**: Using CAMs for tasks like localization or segmentation without requiring bounding boxes.

---

### **7. Gradient-Weighted Class Activation Mapping (Grad-CAM)**

- **Purpose**: Extends CAM to work with pre-trained models and localizes important regions in an image for a prediction.
- **How It Works**:
    - Computes gradients of the target class score with respect to feature maps in the last convolutional layer.
    - Uses these gradients to weigh the feature maps, highlighting influential regions.
- **Benefit**: Does not require architecture changes and works with a wide range of CNNs.

##### In more details:

Grad-CAM is an explainability technique used for Convolutional Neural Networks (CNNs) to visualize the areas of an input
image that contribute the most to a specific class prediction. Unlike Class Activation Maps (CAM), which require
specific architectures (like Global Average Pooling), Grad-CAM can be applied to any CNN architecture.

###### **How Grad-CAM Works**

Grad-CAM relies on **gradients** of the class score with respect to the feature maps in a convolutional layer. These
gradients indicate the importance of each spatial location in the feature maps for the target class. Using this
information, Grad-CAM generates a heatmap that highlights the regions in the input image most relevant for the class
prediction.

###### **Steps to Compute Grad-CAM**

1. **Forward Pass**:
    - Pass the input image through the CNN to compute the class scores.
    - Identify the target class (e.g., the class with the highest probability or any specific class of interest).

2. **Backward Pass**:
    - Perform a **backpropagation** step to compute the gradients of the target class score $ y^c $ with respect to the
      activations $ A^k $ of a specific convolutional layer:
      $
      \frac{\partial y^c}{\partial A^k}
      $
    - These gradients indicate how changes in the activation $ A^k $ at each spatial location affect the class score.

3. **Compute Importance Weights**:
    - Average the gradients over the spatial dimensions $(i, j$) to get the importance weights $ \alpha_k^c $ for each
      feature map $ k $:
      $
      \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}
      $
    - Here, $ Z $ is the total number of spatial locations (height × width) in the feature map.

4. **Weighted Combination**:
    - Compute a weighted sum of the feature maps using the importance weights:
      $L^c_{\text{Grad-CAM}} = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$
    - The ReLU operation ensures that only positive contributions are considered, as negative values typically represent
      areas irrelevant to the class.

5. **Overlay the Heatmap**:
    - Resize $ L^c_{\text{Grad-CAM}} $ to match the input image dimensions (using interpolation).
    - Overlay the heatmap on the original image to visualize the regions contributing to the class prediction.

###### **Intuition Behind Grad-CAM**

- **Gradients**: Tell us how sensitive the output is to changes in the activations of the feature maps.
- **Feature Maps**: Capture spatial information about the input.
- By combining these, Grad-CAM provides a class-specific localization map without requiring architectural constraints
  like CAM.

###### **Applications of Grad-CAM**

1. **Explainability**:
    - Highlight areas of an image that influence the model’s decision for a specific class.

2. **Error Analysis**:
    - Understand why the model made an incorrect prediction by identifying focus regions.

3. **Localization**:
    - Localize objects in an image without bounding box annotations (weak supervision).

4. **Model Debugging**:
    - Identify whether the model is focusing on irrelevant regions (e.g., background noise).

###### **Advantages of Grad-CAM**

- **Flexible**: Works with any CNN architecture.
- **Class-Specific**: Highlights regions specific to the class of interest.
- **Intuitive**: Easy to understand and visualize.

###### **Limitations of Grad-CAM**

1. **Resolution**:
    - Grad-CAM maps are often low-resolution because they use feature maps from deeper layers.

2. **Dependence on Final Layers**:
    - It focuses on the deeper convolutional layers, which may miss fine-grained details from earlier layers.

3. **Not Robust to Noise**:
    - The quality of heatmaps can degrade if the model is noisy or poorly trained.

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
- **Benefit**: Captures both linear and non-linear relationships, satisfying desirable properties like sensitivity and
  implementation invariance.

---

### **11. Explainable Region Aggregation for Images (XRAI)**

- **Purpose**: Improves upon occlusion methods by identifying contiguous, meaningful regions instead of individual
  pixels.
- **How It Works**:
    - Aggregates explanations over image regions (e.g., superpixels) instead of sliding windows or gradients.
    - Produces a heatmap indicating which regions are most influential for the prediction.
- **Benefit**: More interpretable and robust to noise compared to pixel-wise methods.

---

### Comparison Summary

| **Method**               | **Best For**                        | **Data Type** | **Global/Local** | **Key Strength**                          |
|--------------------------|-------------------------------------|---------------|------------------|-------------------------------------------|
| PDP                      | Global trends                       | Tabular       | Global           | Simplifies feature impact visualization.  |
| ICE                      | Instance-specific feature effects   | Tabular       | Local            | Captures heterogeneity of feature impact. |
| Sliding Window Occlusion | Image region importance             | Images        | Local            | Simple, intuitive for vision tasks.       |
| Adaptive Occlusion       | Region importance refinement        | Images        | Local            | Reduces computational cost.               |
| Randomized Occlusion     | Robustness testing                  | Any           | Local            | Tests model sensitivity to occlusions.    |
| CAM                      | Class localization                  | Images        | Local            | Intuitive; class-specific localization.   |
| Grad-CAM                 | Class localization for CNNs         | Images        | Local            | Works with any CNN architecture.          |
| Guided Grad-CAM          | High-resolution class maps          | Images        | Local            | Fine-grained and class-specific maps.     |
| Gradient x Input         | Input-level feature attribution     | Any           | Local            | Linear relationships, fast computation.   |
| Integrated Gradients     | Path-integrated feature attribution | Any           | Local            | Captures non-linear relationships.        |
| XRAI                     | Aggregated region importance        | Images        | Local            | Contiguous, interpretable regions.        |

Let me know if you'd like further details or examples of these methods!
