# **Movie Recommendation System with NCF**

This project implements a movie recommendation system utilizing a custom-trained Neural Collaborative Filtering (NCF) deep learning model and a user-friendly Streamlit dashboard. It leverages SQL databases to store movie data, user ratings, and recommendations, with the NCF model used for efficient recommendation generation. To enhance scalability, Apache Spark is employed for training the deep learning model across large datasets.

---

## **Overview: Neural Collaborative Filtering (NCF)**

Neural Collaborative Filtering (NCF) is a deep learning-based approach used for building recommendation systems. Unlike traditional collaborative filtering methods (e.g., matrix factorization), NCF captures non-linear user-item interactions using neural networks. The model is composed of two primary components:

1. **Embedding Layers**: These map users and items (e.g., movies) into low-dimensional vectors.
2. **Neural Network Layers**: These learn complex interactions between the user and item embeddings, producing a final prediction score indicating user preferences for an item.

NCF is particularly powerful in capturing hidden patterns in user preferences and item characteristics, making it ideal for personalized movie recommendations. The architecture is scalable, adaptable to large datasets, and has been shown to outperform traditional methods in many cases.

<p align="center">
<img src="NCF.png" alt="drawing" width="50%"/>
</p>
<p align="center">
NCF Architecture
</p>

---

## **MovieLens Dashboard**

### **MovieLens Dataset Analysis**
<p align="center">
<img src="images/movielens_analysis.png" alt="drawing" width="75%"/>
</p>
<p align="center">
MovieLens Dataset Analysis
</p>

### **Movie Recommendation System**
<p align="center">
<img src="images/recommendation_system.png" alt="drawing" width="75%"/>
</p>
<p align="center">
Movie Recommendation System
</p>

---

## **Technologies and Libraries Used**

The project utilizes the following technologies and libraries:

### **Programming Languages**
- **Python**: Core language for implementing the recommendation system, data processing, and dashboard.

### **Data Analysis and Visualization**
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations and array handling.
- **Matplotlib**: For creating static, animated, and interactive visualizations.
- **Seaborn**: For statistical data visualization.

### **Machine Learning and Deep Learning**
- **PyTorch**: For building and training the Neural Collaborative Filtering (NCF) model.
- **PyTorch Lightning**: For simplifying PyTorch training loops and improving scalability.
- **Scikit-learn**: For dataset splitting and pre-processing.

### **Big Data**
- **Apache Spark**: For handling and processing large-scale datasets using Spark SQL and DataFrames.

### **Web Frameworks**
- **Streamlit**: For building an interactive, web-based dashboard.

### **Database**
- **SQLite**: For storing user data, movie data, and precomputed recommendations.

### **Utilities**
- **TQDM**: For displaying progress bars during computations.
- **OS Module**: For file and path manipulations.

---

## **Project Structure**

- **`ml-1m/`**:
  - Directory containing the MovieLens 1M dataset.
  - Refer to the `README` file in this folder for dataset details.
- **`BigData_Project_v4.ipynb`**:
  - Jupyter notebook for training the Neural Collaborative Filtering (NCF) model.
  - The trained model weights are saved as `mrs-v4.pkl`.
- **`movie_recommendation.db`**:
  - SQLite database storing the `movies`, `ratings`, and `recommendations` tables.
- **`movie_dashboard_v3.py`**:
  - Streamlit dashboard script for interacting with the recommendation system.
- **`load_data_to_sql.py`**:
  - Script for loading the dataset into the SQLite database.
- **`create_views.py`**:
  - Script for creating reusable SQL views to speed up dashboard queries.
- **`precompute_recommendations.py`**:
  - Script to precompute and store the top 10 movie recommendations for all users.

---

These technologies and libraries work together to create a robust and scalable recommendation system and data analysis dashboard.

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/HARSHALK2598/BigDataProject.git
cd BigDataProject
```

### **2. Install Dependencies**
Ensure you have Python 3.8+ installed. Then, use the `requirements.txt` file to install dependencies:

```bash
pip install -r requirements.txt
```

### **3. Create SQL Views**
Run the `create_views.py` script to create views that optimize dashboard queries:

```bash
python create_views.py
```

### **4. Precompute Recommendations**
Run the `precompute_recommendations.py` script to generate and store the top 10 recommendations for all users in the database:
```bash
python precompute_recommendations.py
```

---

## **5. Run the Streamlit Dashboard**
Launch the dashboard to explore movie recommendations and data analysis:
```bash
streamlit run movie_dashboard_v3.py
```

---

## **Features**

### **MovieLens Analysis**
- Visualize rating distributions.
- View top 10 rated movies and most reviewed movies.
- Analyze average ratings by genre and review count.

### **Recommendation System**
- Get personalized movie recommendations for any user.
- View the top 10 highest-rated movies based on user ratings.

---

## **Usage Notes**
- The database (`movie_recommendation.db`) stores all tables and views used by the dashboard.
- Ensure the dataset (`ml-1m`) and pre-trained model (`mrs-v4.pkl`) are present in the project directory.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

