# Medical Text Clustering Project

This project aims to cluster text abstracts related to different fields of medical technology and healthcare, including: 
* Electronic Health Records (EHR)
* Healthcare Robotics
* Medical Imaging
* Precision Medicine
* Telemedicine

The project employs various Natural Language Processing (NLP) techniques and Unsupervised Machine Learning models ranging from traditional Bag-of-Words models to modern Transformer-based architectures like Sentence-BERT.

## Project Structure

The workspace is organized into data processing pipelines, analysis notebooks, and clustering model implementations.

### Scripts and data files
*   **`data_process.ipynb`**: The initial data pipeline. It loads raw CSV datasets from the `datasets/` folder, samples documents, performs text cleaning (lowercasing, removing URLs/special characters, lemmatization, stopword removal), and consolidates them into `processed_data.csv`.
*   **`processed_data.csv`**: The final preprocessed dataset used for clustering models.

### Directories
*   **`analysis/`**: Contains notebooks for analyzing clustering results and identifying divergence.
    *   `Word Frequency Analysis.ipynb`: Analyzes top keywords in correctly classified vs. misclassified samples to identify noise words.
    *   `Cluster Divergence Identification.ipynb`: Uses Word2Vec and K-means to identify and visualize cluster divergence.
*   **`K-means/`**: Dedicated notebooks for K-means clustering experiments.
    *   `K-means.ipynb`: Baseline K-means using TF-IDF.
    *   `K-means(Custom Stopwords+LSA).ipynb`: K-means using custom stopwords and Latent Semantic Analysis (LSA) for dimensionality reduction.
    *   `K-means(word2vec).ipynb`: K-means using Word2Vec document embeddings.
*   **`EM/`**: Dedicated notebooks for Expectation-Maximization (Gaussian Mixture Models) clustering.
    *   `em_clustering.ipynb`: GMM clustering using Word2Vec embeddings.
    *   `em_clustering_tfidf.ipynb`: GMM clustering using TF-IDF and PCA.
*   **`HC/`**: Dedicated notebooks for Hierarchical Clustering.
    *   `HC_LSA.ipynb`: Hierarchical clustering using TF-IDF and LSA.
    *   `HC_SBERT.ipynb`: Hierarchical clustering using Sentence-BERT (SBERT) embeddings.
*   **`datasets/`**: Source data folder containing the original CSV files for each medical field.

## Environment & Requirements

This project works with Python 3.12+ and requires the following major libraries:

*   **Data Processing**: `pandas`, `numpy`
*   **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
*   **Machine Learning**: `scikit-learn`, `scipy`
*   **NLP Tools**: `nltk`, `gensim`, `sentence-transformers`

### Installation

To set up the environment, install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy nltk gensim sentence-transformers wordcloud
```

## Getting Started

1.  **Data Preparation**: Ensure your raw CSV files are in the `datasets/` folder.
2.  **Preprocessing**: Run `data_process.ipynb` to generate `processed_data.csv`.
3.  **Clustering**: Open any notebook in `K-means/`, `EM/`, or `HC/` to run clustering models and view performance metrics (Silhouette Score, Kappa Score, NPMI Coherence) and visualizations (t-SNE, Confusion Matrix).
4.  **Analysis**: Use notebooks in the `analysis/` folder to dive deeper into the clustering results and error analysis.
