### **Shapley Values: Overview**

Shapley values come from cooperative game theory and are used to fairly attribute a contribution of each "player" (or feature) to the overall outcome of a system. In machine learning, Shapley values are used to explain a model's predictions by quantifying how much each feature contributes to the model's output.

- **Definition**: Shapley values calculate the marginal contribution of each feature $ i $ by considering all possible subsets of features $ S $ (excluding $ i $).
- **Formula**:
  $
  \phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (|N| - |S| - 1)!}{|N|!} \, \left( v(S \cup \{i\}) - v(S) \right),
  $
  where:
  - $ N $: Set of all features.
  - $ S $: Subset of features not including $ i $.
  - $ v(S) $: The value (e.g., prediction) when using subset $ S $.
  - $ \phi_i $: Shapley value for feature $ i $.

This formula essentially averages the marginal contributions of feature $ i $ across all possible subsets, weighted by the number of subsets of a given size.

---

### **Why Shapley Value Calculation is Hard**

The main challenge in calculating Shapley values is the **combinatorial explosion**:
- For $ n $ features, there are $ 2^n $ subsets $ S $.
- Calculating $ v(S) $ for all subsets becomes infeasible for large $ n $, especially for machine learning models where $ v(S) $ may involve expensive computations (e.g., retraining the model on subsets of features).

---

### **Monte Carlo Approximation of Shapley Values**

Monte Carlo methods approximate the Shapley values by randomly sampling subsets $ S $ instead of iterating over all possible subsets.

1. **Sampling Process**:
   - Randomly sample subsets $ S $ of features multiple times.
   - Compute the marginal contribution of feature $ i $ for each sampled subset:
     $
     \Delta v = v(S \cup \{i\}) - v(S).
     $
   - Store the contributions and average them.

2. **Formula for Approximation**:
   $
   \phi_i \approx \frac{1}{M} \sum_{m=1}^M \left( v(S_m \cup \{i\}) - v(S_m) \right),
   $
   where $ M $ is the number of random subsets sampled.

3. **Why Monte Carlo is Needed**:
   - Exact computation of Shapley values is computationally prohibitive.
   - Monte Carlo approximation balances accuracy and efficiency, providing good estimates with fewer computations.

---

### **Shapley Values for Images: Expected Gradients**

Shapley values for images often require specialized approaches because:
- Images have high-dimensional features (e.g., every pixel is a feature).
- Direct computation of $ v(S) $ for subsets of pixels is infeasible.

#### **Expected Gradients Method**

The **Expected Gradients** method adapts Shapley values for continuous input spaces like images:

1. **Marginal Contribution as Gradients**:
   - Instead of evaluating $ v(S) $ for discrete subsets, approximate the marginal contribution of each pixel using the model's gradients.
   - Gradients measure how much the prediction $ f(x) $ changes when pixel values $ x_i $ are perturbed.

2. **Expectation over Background**:
   - Define a background distribution $ p(z) $ over the input space (e.g., pixel values for "neutral" images).
   - Compute the gradient of the model output $ f(x) $ with respect to each pixel, averaged over samples from the background:
     $
     \phi_i = \mathbb{E}_{z \sim p(z)} \left[ x_i \cdot \nabla_i f(x) \right],
     $
     where $ x_i $ is the value of the $ i $-th pixel and $ \nabla_i f(x) $ is the gradient of $ f(x) $ w.r.t. $ x_i $.

3. **Connection to Shapley Values**:
   - This approach uses gradients to approximate marginal contributions, avoiding the need to evaluate $ v(S) $ for all subsets.
   - The expectation over background samples ensures that the contributions are averaged across a range of possible inputs.

4. **Implementation for Images**:
   - A common approach is to randomly sample background images, calculate the gradients for each pixel, and compute the Shapley values using the expected gradients formula.

---

### **Summary of the Steps for Images**
1. Choose a background dataset representing "neutral" input distributions.
2. For each pixel:
   - Sample background images.
   - Compute gradients of the model output w.r.t. the pixel value.
   - Average these gradients to get the Shapley value for that pixel.
3. Visualize the resulting Shapley values to explain the model's focus (e.g., using heatmaps).

---

### **Why Expected Gradients Work**
- Efficient: Avoids combinatorial subset evaluations.
- Continuous Inputs: Naturally handles continuous and high-dimensional data.
- Scalable: Applicable to deep learning models without retraining or extensive perturbations.

By combining Shapley values with gradient-based methods, the **Expected Gradients** approach provides a powerful tool for explainable AI in tasks like image classification and object detection.