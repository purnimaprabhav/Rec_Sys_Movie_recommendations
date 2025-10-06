***

# Movie Recommendation System

An extensible, modular movie recommendation framework integrating **Collaborative Filtering**, **Content-Based Filtering**, and **Hybrid recommendation strategies** designed for comprehensive experimentation and real-time evaluation. Developed with **Streamlit** to provide an intuitive and immersive interface for users and researchers alike.

***

### Core Features

- **Neural Collaborative Filtering (PyTorch/Tez)**  
  - A tailor-made deep learning model that learns latent embeddings for both users and movies, optimizing predictive accuracy for user ratings.  
- **Content-Based Recommendation**  
  - Employs TF-IDF vectorization on movie synopses combined with cosine similarity metrics to deliver contextually relevant suggestions.  
- **Hybrid Model (LightFM)**  
  - Integrates collaborative signals with content metadata in a unified matrix factorization framework trained under WARP loss, enhancing recommendation diversity and precision.  
- **Streamlit Interactive Web Application**  
  - Enables dynamic exploration of recommendation outputs, model evaluation, and side-by-side comparison of methodologies through a user-friendly interface.  
- **Comprehensive Evaluation Suite**  
  - Tools for quantitative assessment of model performance including standard metrics and customizable configurations.

***

### Dataset Summary

This repository utilizes publicly available data from the renowned [The Movies Dataset by Rounak Banik (Kaggle)], which must be placed within the `datasets/` directory for seamless pipeline execution.  

- `movies_metadata.csv`: Rich movie metadata including genres, release details, and descriptive overviews.  
- `keywords.csv`: Curated set of keywords tagging movie content.  
- `links_small.csv`: External identifiers and links, ancillary to recommendation logic.  
- `ratings_small.csv`: Sparse user rating matrix, central to collaborative algorithms.

***

### Model Architecture & Methods

- **Collaborative Filtering Neural Model (`RecSysModel`)**  
  - Employs a PyTorch/Tez-based architecture to jointly embed users and items into a latent feature space, facilitating personalized rating predictions with deep representation learning.  

- **Content-Based Filtering Pipeline**  
  - Implements TF-IDF vectorization over textual movie descriptions followed by cosine similarity computations to identify nearest neighbors in content space.  

- **Hybrid Approach Using LightFM**  
  - Combines user-item interaction data with side information in a hybrid model optimized via WARP loss for effective ranking. Executed in `lightfm_recommender.py`.  

***

### Installation & Setup

1. Clone the repository and navigate into the project directory:  
   ```bash
   git clone <repo_url>
   cd Recommender_System-main
   ```
2. Create a virtual environment (highly recommended) and install project dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

***

### Usage Instructions

- Launch the interactive recommendation interface:  
  ```bash
  streamlit run app.py
  ```
- Train and evaluate models independently:  
  - Collaborative filtering model:  
    ```bash
    python recommender.py
    ```
  - Hybrid LightFM model:  
    ```bash
    python lightfm_recommender.py
    ```

***

### Project Structure

```
Recommender_System-main/
├── app.py                  # Streamlit web application for immersive interaction
├── recommender.py          # Neural collaborative and content-based filtering methods
├── lightfm_recommender.py  # Hybrid LightFM model implementation
├── datasets/               # Required datasets, sourced externally
│   ├── movies_metadata.csv
│   ├── keywords.csv
│   ├── links_small.csv
│   └── ratings_small.csv
├── models/                 # Persisted model artifacts and checkpoints
│   ├── recsysmodel.pth
│   └── lightfm_model.pkl
└── README.md
```

***

### Technical Requirements

- Python version 3.7 or higher  
- Core dependencies include PyTorch, Tez, LightFM, Streamlit, pandas, scikit-learn, numpy (detailed in `requirements.txt`).

***

### Additional Notes

- The system relies on `ratings_small.csv` for all training and validation steps, requiring no custom data splits by default.  
- The hybrid LightFM component is structured as a robust foundation and can be further enhanced with advanced feature engineering and model tuning.  
- Ensure all datasets reside in the `/datasets` folder to guarantee operational integrity.

***

### References & Resources

- Dataset: [The Movies Dataset (Kaggle)](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)  
- Model Documentation: [LightFM Official Documentation](https://making.lyst.com/lightfm/docs/home.html)

***


***

Would you like me to craft that architecture diagram or workflow explanation to add further sophistication?
