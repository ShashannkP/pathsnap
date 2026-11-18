# Reusable Machine Learning Technical Knowledge Base

> **HACKATHON READINESS NOTICE:**  
> This knowledge base consolidates theoretical principles, mathematical formulations, and engineering strategies extracted from the reference slide units (`UE24CS352A`, Units 1–4). It serves as an authoritative, modular repository of ML techniques ready for immediate application in future modeling challenges.

---

## 1. Data Preprocessing and Cleaning

### 1.1 Feature Scaling and Normalization (Unit 1, pp. 355–358; Unit 2, p. 156)
- **Mathematical Principles:** Distance-based algorithms (k-NN, K-Means), gradient-based optimizers (SGD, Adam in Neural Networks), and margin-based classifiers (SVM) are sensitive to feature scales. An unscaled feature with variance $[0, 100,000]$ will dominate Euclidean distance and distort gradient descent trajectories relative to a feature with variance $[0, 1]$.
- **Min-Max Scaling:**
  $$x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \in [0, 1]$$
  *Use Case:* Best for bounded input distributions where preservation of zero and non-negative boundaries is essential, or when data is fed into bounded activations (Sigmoid, Tanh).
- **Z-Score Standardization:**
  $$x_{\text{std}} = \frac{x - \mu}{\sigma}$$
  *Use Case:* Best for features with Gaussian-like distributions, SVMs, and deep neural networks. Centering at zero ($\mu = 0$) prevents systematic directional bias in gradient updates.
- **Robust Scaling:** Uses median and interquartile range (IQR) to scale data without being influenced by severe outliers:
  $$x_{\text{robust}} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$$

### 1.2 Categorical Feature Encoding (Unit 1, pp. 364–367)
- **Nominal / Unordered Features:**
  - *One-Hot Encoding:* Generates $K$ binary indicator columns. Essential for distance-based models (k-NN) to avoid introducing false numerical order.
  - *Frequency / Count Encoding:* Replaces categories with their training frequency; effective in high-cardinality scenarios where one-hot encoding would cause dimensionality explosion.
- **Ordinal Features:** Integer mapping according to intrinsic domain rank (e.g., Low: 0, Medium: 1, High: 2).

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Distributional Diagnostics & Skewness
- **Target Distribution Analysis:** Inspect whether classification labels are balanced or skewed (Unit 1, pp. 96–112; Unit 4, pp. 77–88). Extreme class skew triggers the "Accuracy Paradox" and mandates threshold-invariant metrics.
- **Log-Transform for Heavy Tails:** For right-skewed continuous targets or features (e.g., financial values, counts), apply $x \mapsto \log(1 + x)$ to normalize error variance and stabilize gradient updates.

### 2.2 Feature Correlation and Collinearity (Unit 3, pp. 37–42)
- **Pairwise Correlation ($r_{xy}$):** Highly correlated features ($|r| > 0.9$) degrade linear model coefficient stability and cause redundant tree splits.
- **Multicollinearity in Random Forests:** When two features are collinear, tree ensembles split arbitrarily between them, diluting individual feature importance scores without improving predictive accuracy.

---

## 3. Feature Selection and Engineering

### 3.1 Information-Theoretic Feature Selection (Unit 1, pp. 199–216)
- **Shannon Entropy and Information Gain:**
  $$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
- Rank all features by Information Gain (or Mutual Information) relative to the target label to rapidly filter out uninformative noise features prior to training.

### 3.2 Text Feature Representations (Unit 3, pp. 184–216)
- **Bag-of-Words (BoW):** Translates documents into sparse word count vectors.
- **Unseen Words & Out-of-Vocabulary (OOV) Handling:** Using Laplace smoothing ($+1$) or background pseudo-counts ($m$-estimate) ensures zero-probability multiplication does not destroy predictions.
- **Numerical Underflow Prevention:** In high-dimensional sparse representations, always evaluate joint probabilities in log-space:
  $$\ln P(y | x) \propto \ln P(y) + \sum_{j=1}^d \ln P(x_j | y)$$

### 3.3 Spatial Representation Learning (Unit 4, pp. 141–175)
- **Spatial Convolutions as Feature Extractors:** Replacing hand-crafted spatial filters (Sobel, Prewitt) with learnable convolutional kernels that automatically discover multi-scale edge, corner, and semantic texture representations.

---

## 4. Classification Algorithms

### 4.1 Decision Trees (ID3 / CART) (Unit 1, pp. 191–266)
- **Core Mechanism:** Top-down greedy recursive partitioning using Information Gain (Entropy) or Gini Impurity.
- **Decision Boundaries:** Axis-aligned, step-wise orthogonal hyperplanes.
- **When to Use:** When feature interpretability is paramount; when data contains mixed types (categorical and continuous); when non-linear feature interactions exist.
- **Mitigating Overfitting:** Early stopping (minimum samples per leaf, maximum depth) and post-pruning (Reduced Error Pruning, Rule Post-Pruning).

### 4.2 Support Vector Machines (SVM) (Unit 2, pp. 153–272)
- **Core Mechanism:** Finds the optimal separating hyperplane that maximizes geometric margin $2/\|w\|$ between classes. Solved via Lagrangian dual formulation over support vectors:
  $$\max_\alpha \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j) \quad \text{s.t. } 0 \le \alpha_i \le C, \sum_{i=1}^N \alpha_i y_i = 0$$
- **Kernel Selection:**
  - *Linear:* High-dimensional text classification ($d \gg N$).
  - *RBF (Gaussian):* Complex non-linear boundaries in low-to-medium dimensions ($N < 50,000$).
- **Key Hyperparameters:** Regularization $C$ (margin rigidity vs. error tolerance) and $\gamma$ (RBF kernel bandwidth / curvature).

### 4.3 Naive Bayes Classifiers (Unit 3, pp. 148–228)
- **Core Mechanism:** Applies Bayes' theorem with the assumption that features are conditionally independent given the class label.
- **When to Use:** Baseline text classification, spam filtering, and streaming high-dimensional data. Extremely fast training ($O(Nd)$).

### 4.4 k-Nearest Neighbors (k-NN) (Unit 1, pp. 299–373)
- **Core Mechanism:** Assigns the majority vote of the $k$ closest neighbors in feature space according to a distance metric (Euclidean, Manhattan).
- **Distance-Weighted Voting:**
  $$w_i = \frac{1}{d(x_q, x_i)^2}$$
- **When to Use:** Low-dimensional datasets ($d < 20$) with irregular decision boundaries where no parametric assumption holds.

---

## 5. Regression Algorithms

### 5.1 Ordinary Least Squares (OLS) & Gradient Descent (Unit 1, pp. 267–290; Unit 3, pp. 104–109)
- **Theoretical Foundation:** Minimizing Mean Squared Error is mathematically equivalent to Maximum Likelihood Estimation under the assumption that residuals are independent and identically distributed (IID) Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
- **Cost Function & Gradient:**
  $$J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2, \quad \nabla_\theta J(\theta) = \frac{1}{m} X^T (X\theta - y)$$

### 5.2 k-NN Regression (Unit 1, pp. 311–322, 343–347)
- Computes local continuous prediction as the unweighted or distance-weighted average of the target values of the $k$ nearest training samples:
  $$\hat{y}(x_q) = \frac{\sum_{i \in N_k(x_q)} w_i y_i}{\sum_{i \in N_k(x_q)} w_i}$$

---

## 6. Clustering and Dimensionality Reduction

### 6.1 Partitional Clustering: K-Means & Bisecting K-Means (Unit 4, pp. 37–74)
- **Objective:** Minimize Within-Cluster Sum of Squares (Inertia):
  $$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
- **Bisecting K-Means Workflow:** Divisive hierarchical splitting of the cluster with maximum sum of squared errors; drastically lowers sensitivity to random initial centroid seeding.
- **Determining $K$:** Elbow method on inertia curve; Silhouette analysis (maximizing average silhouette width $s(i) \in [-1, 1]$).

### 6.2 Hierarchical Clustering: AGNES (Unit 4, pp. 14–36)
- **Bottom-Up Agglomeration:** Iteratively merges the two closest clusters according to linkage criteria (Single, Complete, Average, Centroid).
- **Output:** Interpretable tree hierarchy (Dendrogram). Horizontal cut through dendrogram yields target cluster partition.

### 6.3 Dimensionality Reduction & The Curse of Dimensionality (Unit 1, pp. 359–363)
- In high dimensions ($d > 50$), distance metrics degenerate as the ratio between the distance to the nearest point and the distance to the farthest point approaches 1:
  $$\lim_{d \to \infty} \frac{\text{dist}_{\max} - \text{dist}_{\min}}{\text{dist}_{\min}} = 0$$
- **Remedies:** Feature selection via information gain; pooling layers in spatial data (Unit 4); PCA or truncated SVD for dense projection.

---

## 7. Deep Learning Architectures

### 7.1 Multi-Layer Perceptrons (MLP) (Unit 2, pp. 45–147)
- **Universal Approximation:** A feedforward network with a single non-linear hidden layer can approximate any continuous function on compact subsets of $\mathbb{R}^n$.
- **Backpropagation via Multivariate Chain Rule:** Propagates output error gradients backwards to update layer weights:
  $$\delta^{[l]} = \left( (W^{[l+1]})^T \delta^{[l+1]} \right) \odot g'^{[l]}(z^{[l]}), \quad \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$

### 7.2 Convolutional Neural Networks (CNN) (Unit 4, pp. 128–176)
- **Key Inductive Biases:** Local connectivity (sparse interactions) and parameter sharing (weight sharing), rendering representations equivariant to spatial translation.
- **Spatial Geometry Formulas:**
  - Output spatial dimension:
    $$O = \left\lfloor \frac{N - F + 2P}{S} \right\rfloor + 1$$
  - Filter parameters per conv layer:
    $$\text{Params} = K \times \left( F \times F \times C_{\text{in}} + 1 \right)$$
- **Pooling (Max vs. Average):** Reduces spatial resolution, provides translational invariance, and enforces zero learnable parameter overhead.

---

## 8. Ensemble Learning and Boosting

### 8.1 Bagging (Bootstrap Aggregation) (Unit 3, pp. 9–14)
- **Theory:** Averages predictions of $B$ independently trained models on bootstrap samples (drawn with replacement).
- **Variance Reduction:**
  $$\text{Var}(\bar{f}) = \frac{1}{B} \sigma^2 \quad (\text{for perfectly uncorrelated models})$$
- Reduces variance without increasing bias. Ideal for high-variance, unstable base learners (deep unpruned decision trees).

### 8.2 Random Forest (Unit 3, pp. 37–65)
- **Feature Bagging / Random Subspaces:** At each split in each tree, samples a random subset of $m \approx \sqrt{d}$ candidate features.
- Decorrelates individual trees, suppressing the inter-tree correlation coefficient $\rho$ and driving ensemble variance down to its theoretical minimum:
  $$\text{Var}(\text{RF}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2 \to \rho \sigma^2$$
- Provides built-in feature importance via Mean Decrease in Impurity (MDI) and out-of-bag validation without separate split.

### 8.3 Gradient Boosting (Unit 3, pp. 21–36)
- **Sequential Optimization:** Fits weak learners (shallow trees with depth 3–6) to the negative gradient (pseudo-residuals) of the loss function:
  $$r_{im} = - \left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)}$$
- Shrinkage learning rate $\nu \in (0.01, 0.2]$ acts as L2 regularization in function space, dampening individual step sizes and preventing overshooting.

---

## 9. Transfer Learning and Pretrained Models (Unit 4, pp. 141–176)
- **Feature Extraction:** Freezing early convolutional layers (which capture universal low-level features such as Gabor-like edge detectors and color blobs) and training a new classifier head on top.
- **Fine-Tuning:** Unfreezing the final convolutional blocks and training with a very small learning rate ($\alpha \approx 10^{-5}$) to adapt high-level semantic representations to the target task.

---

## 10. Model Optimization and Hyperparameter Tuning

### 10.1 Optimization Algorithms (Unit 2, pp. 129–152)
- **SGD with Momentum:**
  $$v_t = \beta v_{t-1} + (1 - \beta) \nabla_\theta J(\theta), \quad \theta := \theta - \alpha v_t$$
  Accelerates through flat plateaus and dampens perpendicular oscillations.
- **Adam (Adaptive Moment Estimation):**
  Combines 1st moment (momentum) and 2nd moment (RMSprop scaling). Robust default optimizer for neural networks ($\alpha = 0.001, \beta_1 = 0.9, \beta_2 = 0.999$).
- **Learning Rate Schedules:** Step decay, exponential decay, or cosine annealing:
  $$\alpha_t = \alpha_0 \cdot e^{-kt}$$
  Allows rapid exploration in early epochs and precise convergence to the minimum in later epochs.

### 10.2 Support Vector Machine Regularization (Unit 2, pp. 264–272)
- Soft margin penalty parameter $C$ controls the trade-off:
  - High $C$: Strict penalty on errors; narrow margin; risks overfitting.
  - Low $C$: High tolerance for slack $\xi_i$; wide margin; risks underfitting.
- RBF Kernel bandwidth $\gamma$:
  - High $\gamma$: Narrow Gaussian bell; points must be very close to influence classification; complex decision boundaries.
  - Low $\gamma$: Broad Gaussian bell; smooth, generalized boundaries.

---

## 11. Validation Strategies and Evaluation Metrics

### 11.1 Cross-Validation and Out-of-Bag (OOB) Validation
- **K-Fold Stratified Cross-Validation:** Mandatory for tabular datasets to preserve class proportions across folds.
- **Out-of-Bag (OOB) Validation (Unit 3, pp. 12–14, 42):** Because $\approx 36.8\%$ of data points are excluded from each bootstrap sample, evaluating each tree on its out-of-bag samples yields an unbiased cross-validation score with zero extra training overhead!

### 11.2 Evaluation Metrics Selection (Unit 1, pp. 96–160; Unit 4, pp. 77–127)
- **Accuracy:** Suitable only when classes are perfectly balanced and misclassification costs are symmetric.
- **Precision:** $\frac{\text{TP}}{\text{TP} + \text{FP}}$. Optimize when False Positives are expensive (e.g., spam detection, content moderation).
- **Recall (Sensitivity):** $\frac{\text{TP}}{\text{TP} + \text{FN}}$. Optimize when False Negatives are catastrophic (e.g., medical diagnosis, defect detection).
- **F1-Score:** Harmonic mean balancing Precision and Recall:
  $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Macro-F1 vs. Micro-F1 (Unit 1, pp. 115–124; Unit 4, pp. 95–104):**
  - *Macro-F1:* Unweighted average of per-class F1-scores. Treats all classes equally; critical for highlighting performance on rare minority classes.
  - *Micro-F1:* Aggregates global TP, FP, FN before computing F1. Dominated by majority classes.
  - *Weighted-F1:* Averages per-class F1 weighted by class support.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
  Measures classification threshold-independent discriminative ability. Equivalent to the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example.
- **Precision-Recall (PR) AUC:** Superior to ROC-AUC when evaluating on severe class imbalance (e.g., positive class $< 5\%$), where a high number of True Negatives can artificially inflate ROC-AUC.

---

## 12. Handling Class Imbalance, Missing Values, and Noisy Data

### 12.1 Class Imbalance Solutions
- **Class-Weighted Loss Functions:** Scale the loss of minority class samples inversely proportional to their class frequency:
  $$w_c = \frac{N}{K \cdot N_c}$$
- **Threshold Tuning via ROC Analysis (Unit 1, pp. 128–146; Unit 4, pp. 105–112):** Adjust the classification decision threshold $\tau \ne 0.5$ along the ROC curve to maximize the desired metric (e.g., F1 or Recall at fixed Precision).
- **Laplace Smoothing and $m$-Estimate (Unit 3, pp. 172–179):** Prevents zero probabilities from wiping out class posterior estimates in sparse settings.

### 12.2 Missing Value Strategies (Unit 1, pp. 264–265)
- **Tree-Based Strategies:**
  - Assign the most common value at the current split node.
  - Assign the most common value among samples sharing the same target class label.
  - Fractional sample weighting across child branches.
- **General Tabular Strategies:** Median imputation for skewed numerical features; mode/constant imputation for categorical features; indicator flags (`is_missing`) to preserve missingness signals.

### 12.3 Robustness Against Noise and Outliers
- **Soft Margin Slack Variables ($\xi_i$) in SVM (Unit 2, pp. 264–272):** Allows margin violations without distorting the global orientation of the optimal hyperplane.
- **Reduced Error Pruning & Rule Post-Pruning (Unit 1, pp. 248–257):** Prunes noisy overfitted leaf branches by verifying accuracy improvements on an independent validation set.
- **Random Forest Averaging (Unit 3, pp. 37–52):** Random bootstrapping and feature subsampling naturally cancel out zero-mean random noise across ensemble members.
