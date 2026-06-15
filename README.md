# Data Mining Analysis: Comparative Classification, Clustering and Outlier Detection

**Author:** Ibaale Eric
**Student No:** 230070288
**Reg No:** 23/U/08288/PS
**Course Unit:** Data Mining
**Institution:** Makerere University, School of Statistics and Planning

## Project Overview

This project is a comprehensive data mining analysis submitted as coursework for the Data Mining course unit. It is organised into two parts, each addressing a distinct family of machine learning techniques applied to different real-world datasets.

**Part 1** focuses on supervised classification. Five machine learning algorithms are trained and evaluated on the Adult Income dataset to predict whether an individual earns more than USD 50,000 per year. The models are compared using accuracy, sensitivity, specificity, balanced accuracy, and ROC/AUC analysis.

**Part 2** focuses on unsupervised learning. K-Means and agglomerative hierarchical clustering are applied to the Iris dataset to assess how well the algorithms recover the known species groupings without using label information. Three outlier detection methods are then compared to identify anomalous observations.

All analyses are contained in a single knit-ready R Markdown document and are fully reproducible using a fixed random seed.

## Live Report: https://github.com/ibel123/classification-clustering-outlier-analysis/

## Repository Contents

```
project/
├── Comparative_Classification_Models.Rmd   # Main consolidated analysis document
├── adult.data                              # Adult Income dataset (UCI Repository)
├── adult.names                             # Column name reference for adult.data
└── README.md                              # This file
```

The Iris dataset is built into R and does not require an external file.

## Datasets

### Adult Income Dataset
**Source:** UCI Machine Learning Repository — 1994 U.S. Census Bureau database
**Target variable:** `income` — binary, either `<=50K` or `>50K`
**Features:** 14 demographic and employment attributes including age, workclass, education, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country
**Size after cleaning:** 30,139 observations

### Iris Dataset
**Source:** Built into R via `data(iris)`
**Target variable:** `Species` — three classes: setosa, versicolor, virginica
**Features:** Sepal length, sepal width, petal length, petal width (all continuous, in cm)
**Size:** 150 observations, 50 per species

## Analyses Performed

### Part 1: Comparative Classification Modeling (Adult Income Dataset)

**Data Preparation**
- Missing value detection and replacement (`"?"` → `NA`)
- Duplicate removal
- Outlier identification via IQR method and boxplot visualisation
- Factor encoding for categorical variables
- 70/30 train-test split with `set.seed(123)`

**Exploratory Data Analysis**
- Income class distribution
- Education level vs income cross-tabulation and visualisation
- Gender vs income comparison
- Hours worked per week vs income (boxplot)
- Age distribution vs income (boxplot)

**Classification Models**

| Model | Key Parameters |
|---|---|
| Decision Tree | `rpart`, `method = "class"` |
| Naïve Bayes | `naiveBayes`, Gaussian/multinomial distributions |
| Logistic Regression | `glm`, `family = binomial` |
| K-Nearest Neighbors | `knn`, k = 11, Euclidean distance, scaled features |
| Artificial Neural Network | `nnet`, 10 hidden neurons, L2 decay = 0.001, 300 iterations |

**Evaluation**
- Confusion matrix per model
- Accuracy, sensitivity, specificity, and balanced accuracy
- Combined ROC curve with AUC annotations for all five models

### Part 2: Clustering and Outlier Analysis (Iris Dataset)

**K-Means Clustering**
- Features scaled to zero mean and unit variance
- Elbow method (k = 1 to 8) to determine optimal k
- Final model: k = 3, nstart = 25, `set.seed(123)`
- `fviz_cluster()` visualisation with convex hulls in PCA space
- Comparison of cluster assignments to true species labels

**Agglomerative Hierarchical Clustering**
- Euclidean distance matrix
- Three linkage methods compared: Ward's D2, complete, and average
- Colour-coded dendrograms with horizontal cut-point reference lines
- Ward's dendrogram cut at k = 3 with `cutree()` and `rect.hclust()`
- Cluster-species comparison table

**Outlier Detection**
- IQR method applied to Sepal.Length (univariate)
- Local Outlier Factor (LOF) via `dbscan::lof()`, k = 5, all four features
- Isolation Forest via `solitude`, 100 trees, top 5% anomaly score threshold
- Scatter plot of Sepal.Length vs Petal.Length coloured by continuous LOF score
- Bar chart of method agreement across all three detectors

## R Package Requirements

Install all required packages before knitting by running the following in your R console:

```r
install.packages(c(
  "tidyverse",
  "reshape2",
  "ggplot2",
  "knitr",
  "kableExtra",
  "rpart",
  "rpart.plot",
  "e1071",
  "class",
  "nnet",
  "caret",
  "pROC",
  "factoextra",
  "cluster",
  "dendextend",
  "dbscan",
  "solitude"
))
```

**R version used:** 4.4.2 (2024-10-31 ucrt) — "Pile of Leaves"

## How to Run

1. Clone or download this repository to your local machine.
2. Place `adult.data` and `adult.names` in the same directory as the `.Rmd` file.
3. Open `Comparative_Classification_Models.Rmd` in RStudio.
4. Update the `setwd()` path in the `load-data` chunk to point to your local directory.
5. Install all required packages listed above if not already installed.
6. Click **Knit** and select your preferred output format (HTML, PDF, or Word).

The document is configured to knit to all three formats via the YAML header. PDF output requires a LaTeX installation such as TinyTeX (`tinytex::install_tinytex()`).

## Reproducibility

All stochastic operations (K-Means initialisation, train-test splitting, KNN tie-breaking, ANN weight initialisation, and Isolation Forest sampling) use `set.seed(123)`. Results reported in the document are fully reproducible on any machine running R 4.x with the packages listed above.

## Key Results Summary

**Classification (Adult Income)**

The five models achieved the following overall accuracy on the 30% holdout test set:

| Model | Accuracy | AUC |
|---|---|---|
| Decision Tree | 84.09% | 0.8441 |
| Naïve Bayes | 81.75% | 0.8663 |
| Logistic Regression | 84.74% | 0.9028 |
| K-Nearest Neighbors | 81.72% | 0.8604 |
| Artificial Neural Network | 83.63% | 0.8865 |

Logistic Regression achieved the highest accuracy and AUC. The ANN ranked second on both metrics. Naïve Bayes, despite its simplicity, produced a competitive AUC due to its well-calibrated probability estimates. The key income predictors identified by the Decision Tree were relationship status, marital status, capital gain, and education level.

**Clustering (Iris)**

Both K-Means (k = 3) and Ward's hierarchical clustering recovered the three species groupings with approximately 83% assignment accuracy. Setosa was perfectly separated by both methods. The principal source of error was the overlap between versicolor and virginica in the petal dimension feature space.

**Outlier Detection (Iris)**

The IQR method was the most conservative detector, flagging only observations at the extreme ends of the Sepal.Length distribution. LOF and Isolation Forest flagged a broader and partially overlapping set of multivariate outliers concentrated at the versicolor/virginica boundary. Full three-method agreement was rare, reflecting the complementary nature of the three approaches.

## Academic Context

This project was submitted in partial fulfilment of the requirements for the Data Mining course unit at the School of Statistics and Planning, Makerere University. The analyses draw on techniques from supervised learning, unsupervised learning, and anomaly detection as covered in the course curriculum.

**Referees:**
Dr. Frank Namugera, Makerere University
