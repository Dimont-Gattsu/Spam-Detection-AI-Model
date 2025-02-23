# **AI Models for Spam and Phishing Detection**  

This repository contains machine learning models to detect spam messages and phishing URLs using **Perceptron** and **Multilayer Perceptron (MLP) classifiers**.  

## **Datasets**  

### **1. Spam Detection Dataset (`spam.csv`)**  
This dataset is designed for **binary classification**, where the goal is to distinguish between **spam** and **ham (non-spam)** messages based on keyword occurrences.  

- Each row represents an **email or message**.  
- Contains **six numerical features**, each representing the frequency of specific spam-related keywords.  
- The final column (`labels`) indicates whether the message is **spam** or **ham**.  
- This dataset is used to train a **Perceptron model**, which learns a linear decision boundary to classify messages.  

### **2. Phishing Detection Dataset (`phishing_dataset.csv`)**  
This dataset is used for training an **MLP classifier** to detect phishing attempts based on various numerical features extracted from URLs.  

- Each row represents a **URL**, with multiple extracted features.  
- Features include:  
  - **URL structure** (e.g., number of dots, hyphens, slashes).  
  - **Domain attributes** (e.g., `domain_length`, `tld_present_params`).  
  - **Network indicators** (e.g., `asn_ip`, `qty_ip_resolved`).  
  - **Security signals** (e.g., `tls_ssl_certificate`, `domain_spf`).  
  - **Search engine indexing status** (e.g., `url_google_index`).  
- The final column (`phishing`) is the **target label**:  
  - `1` → **Phishing URL**  
  - `0` → **Legitimate URL**  
- The dataset is used to train an **MLP classifier** for phishing detection.  

---

## **Scripts**  

### **1. Perceptron-Based Spam Detection (`spam-detection-perceptron.py`)**  
This script trains a **Perceptron classifier**, a single-layer neural network for **binary classification**.  

**Steps:**  
1. Loads `spam.csv` and converts it into a **NumPy array**.  
2. Extracts **features** and **labels**.  
3. Splits the dataset into **80% training** and **20% testing**.  
4. Trains a **Perceptron model** from `sklearn.linear_model`.  
5. Evaluates the model using:  
   - **Accuracy**  
   - **Precision**  
   - **Recall**  
   - **F1 Score**  
   - **Classification Report**  
6. Saves the trained model using **joblib** for future use.  

💡 *Note:* The Perceptron is effective for **linearly separable data** but may struggle with **non-linear patterns**.  

### **2. MLP-Based Phishing Detection (`spam-detection-mlp.py`)**  
This script trains a **Multilayer Perceptron (MLP) classifier**, a type of **Artificial Neural Network (ANN)**, to detect **phishing attempts**.  

**Steps:**  
1. Loads and processes `phishing_dataset.csv`.  
2. Normalizes features to a **0-1 scale** to avoid biases from large feature values.  
3. Splits the dataset into **training and testing sets**.  
4. Initializes an **MLP classifier** with three hidden layers (`20, 20, 10`).  
5. Trains the model on the dataset.  
6. Evaluates performance using:  
   - **Accuracy**  
   - **Precision**  
   - **Recall**  
   - **F1 Score**  
   - **Classification Report**  
7. Saves the trained model as `trained_model.pkl` for future use.  
