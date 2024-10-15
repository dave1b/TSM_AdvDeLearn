## Pass an embedding through multi-headed self-attention

### **Problem Setup**

Imagine we have a sequence of three words (or tokens) in a sentence. Each word is represented as an embedding vector of size 4. We'll pass these embeddings through a **self-attention** mechanism.

### 1. **Input Embeddings**

Let’s assume that our input embeddings for the three tokens look like this (random values for simplicity):

$$
\text{Embeddings} = \begin{bmatrix} 
1.0 & 0.5 & 0.8 & 0.3 \\ 
0.2 & 0.9 & 0.4 & 0.7 \\ 
0.3 & 0.4 & 0.5 & 0.9 
\end{bmatrix}
$$
So each row represents a token’s embedding. There are 3 tokens, each with a 4-dimensional embedding.

- Token 1 embedding: `[1.0, 0.5, 0.8, 0.3]`
- Token 2 embedding: `[0.2, 0.9, 0.4, 0.7]`
- Token 3 embedding: `[0.3, 0.4, 0.5, 0.9]`

### 2. **Query (Q), Key (K), Value (V) Matrices**

In self-attention, for each token, we compute three vectors:
- Query (Q): Represents the current token that is "asking for information."
- Key (K): Represents the other tokens that the current token compares itself to.
- Value (V): Holds the actual information (embedding) that will be weighted by the attention mechanism.

These vectors are computed by multiplying the input embeddings by learned weight matrices:
- **W_Q**: Weight matrix for queries.
- **W_K**: Weight matrix for keys.
- **W_V**: Weight matrix for values.

Let’s assume that the weight matrices for Query, Key, and Value are:

$$
W_Q = \begin{bmatrix} 
0.1 & 0.2 & 0.3 & 0.4 \\ 
0.4 & 0.3 & 0.2 & 0.1 \\ 
0.5 & 0.5 & 0.5 & 0.5 \\ 
0.1 & 0.3 & 0.2 & 0.4 
\end{bmatrix}
\quad
W_K = \begin{bmatrix} 
0.2 & 0.1 & 0.4 & 0.3 \\ 
0.3 & 0.4 & 0.2 & 0.1 \\ 
0.5 & 0.3 & 0.3 & 0.5 \\ 
0.2 & 0.2 & 0.1 & 0.3 
\end{bmatrix}
\quad
W_V = \begin{bmatrix} 
0.1 & 0.3 & 0.5 & 0.2 \\ 
0.2 & 0.4 & 0.1 & 0.3 \\ 
0.4 & 0.1 & 0.3 & 0.4 \\ 
0.3 & 0.2 & 0.2 & 0.5 
\end{bmatrix}
$$

### 3. **Computing Query, Key, and Value Vectors**

For each token, we compute the **Query (Q)**, **Key (K)**, and **Value (V)** vectors by multiplying the embedding by the weight matrices:

#### Token 1: 
$$
Q_1 = \text{Embedding}_1 \times W_Q = [1.0, 0.5, 0.8, 0.3] \times W_Q
$$
$$
= [1.0 \times 0.1 + 0.5 \times 0.4 + 0.8 \times 0.5 + 0.3 \times 0.1, \; \dots] = [0.89, 0.94, 1.17, 0.69]
$$

$$
K_1 = \text{Embedding}_1 \times W_K = [1.0, 0.5, 0.8, 0.3] \times W_K = [0.61, 0.56, 1.02, 0.64]
$$
$$
V_1 = \text{Embedding}_1 \times W_V = [1.0, 0.5, 0.8, 0.3] \times W_V = [0.61, 0.79, 1.07, 0.77]
$$

Similarly, we compute the Query, Key, and Value vectors for the other tokens.

#### Token 2:
$$
Q_2 = \text{Embedding}_2 \times W_Q = [0.2, 0.9, 0.4, 0.7] \times W_Q = [0.66, 0.71, 0.88, 0.68]
$$
$$
K_2 = \text{Embedding}_2 \times W_K = [0.61, 0.86, 0.78, 0.72]
$$
$$
V_2 = \text{Embedding}_2 \times W_V = [0.53, 0.69, 0.77, 0.82]
$$

#### Token 3:
$$
Q_3 = \text{Embedding}_3 \times W_Q = [0.72, 0.68, 0.83, 0.77]
$$
$$
K_3 = \text{Embedding}_3 \times W_K = [0.74, 0.83, 0.96, 0.79]
$$
$$
V_3 = \text{Embedding}_3 \times W_V = [0.62, 0.74, 0.87, 0.81]
$$

### 4. **Calculating Attention Scores**

Next, we compute the attention scores by taking the dot product of each token’s Query vector with the Key vectors of every token, including itself. This gives us a matrix of scores:

$$
\text{Score}_{ij} = Q_i \cdot K_j^\top
$$

#### Example: Token 1 attending to Token 1:
$$
\text{Score}_{11} = Q_1 \cdot K_1^\top = [0.89, 0.94, 1.17, 0.69] \cdot [0.61, 0.56, 1.02, 0.64]^\top = 0.89 \times 0.61 + 0.94 \times 0.56 + \dots = 1.91
$$

#### Token 1 attending to Token 2:
$$
\text{Score}_{12} = Q_1 \cdot K_2^\top = [0.89, 0.94, 1.17, 0.69] \cdot [0.61, 0.86, 0.78, 0.72]^\top = 2.34
$$

And similarly for the rest of the tokens.

### 5. **Applying Softmax to Get Attention Weights**

Once we have the raw scores, we apply the softmax function to normalize them. This ensures that the attention weights sum to 1, and each token's relevance is represented as a probability.

For Token 1, assume after applying softmax, we get the attention weights:

$$
\text{Attention Weights}_1 = [0.2, 0.5, 0.3]
$$

These weights represent how much attention Token 1 pays to each of the tokens (including itself).

### 6. **Weighted Sum of Value Vectors**

Now, for Token 1, we compute the output by taking a weighted sum of the Value vectors of all tokens using the attention weights.

$$
\text{Output}_1 = 0.2 \times V_1 + 0.5 \times V_2 + 0.3 \times V_3
$$
$$
= 0.2 \times [0.61, 0.79, 1.07, 0.77] + 0.5 \times [0.53, 0.69, 0.77, 0.82] + 0.3 \times [0.62, 0.74, 0.87, 0.81]
$$
$$
= [0.55, 0.72, 0.90, 0.80]
$$

Similarly, we compute the output for the other tokens.

### 7. **Final Output**

The final output is a new set of embeddings for each token, where each token now contains information about the entire sequence (due to self-attention). These embeddings are then passed to the next layer of the model (e.g., another self-attention layer or a feed-forward layer).

### **Summary**:
1. Input embeddings are passed through trainable projection matrices to get Query, Key, and Value vectors.
2. Each token's Query vector interacts with