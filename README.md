AI Models to detect spam given a dataset.

spam.csv :
    The dataset spam.csv is structured for a binary classification problem, where the goal is to distinguish between spam and ham (non-spam) messages based on specific keywords.
    Each row represents an email or message, with six numerical features corresponding to the occurrence of spam words in the message.
    The final column, labeled labels, indicates whether the message is "spam" or "ham".
    The numerical values in the feature columns represent how many times each keyword appears in the message.
    This dataset is used to train a Perceptron model to identify spam based on these keyword occurrences, learning a linear decision boundary to classify new messages as either spam or ham.


spam-detection-perceptron.py:
    This script implements a Perceptron classifier, a simple single-layer neural network used for binary classification.
    It starts by loading a dataset (spam.csv), converting it to a NumPy array, and extracting features and labels.
    The data is then split into training and testing sets using an 80/20 split.
    The Perceptron model from sklearn.linear_model is trained on the training data, learning a linear decision boundary.
    The model’s performance is evaluated using accuracy, precision, recall, F1-score, and a classification report.
    Finally, the trained model is saved to disk using joblib for future use.
    This approach is effective for linearly separable data but may struggle with more complex, non-linear patterns.


phishing_dataset.csv:
    The file phishing_dataset.csv contains a dataset used for training and evaluating a Multilayer Perceptron (MLP) classifier to detect phishing URLs based on various numerical features. 
    Each row represents a URL, and the columns contain extracted features such as the number of dots (qty_dot_url), hyphens (qty_hyphen_url), slashes (qty_slash_url),
    and other special characters in different parts of the URL (domain, directory, file, and parameters).
    Additional features include domain-related attributes (e.g., domain_length, tld_present_params), network indicators (e.g., asn_ip, qty_ip_resolved), security-related signals (e.g., tls_ssl_certificate, domain_spf),
    and search engine indexing status (e.g., url_google_index).
    The final column (phishing) serves as the label, where 1 likely indicates a phishing URL and 0 a legitimate one.
    This dataset is used to train and test the MLP classifier to distinguish phishing from legitimate URLs based on these characteristics.
    

spam-detection-mlp.py:
    This script trains a Multilayer Perceptron (MLP) classifier, a type of Artificial Neural Network (ANN), to detect phishing attempts based on numerical features from the dataset phishing_dataset.csv.
    The dataset is first loaded and converted into a NumPy array, where the last column represents class labels (e.g., phishing or non-phishing), and the rest are feature values.
    The features are normalized to scale between 0 and 1, preventing larger values from disproportionately influencing the model.
    The dataset is then split into training and testing sets, ensuring a fair evaluation of the model’s performance.
    The MLP classifier is initialized with three hidden layers of sizes (20, 20, 10) and trained on the processed data.
    The script evaluates the model using accuracy, precision, recall, and F1 score, generating a classification report to assess its predictive performance.
    Finally, the trained model is saved to a file (trained_model.pkl) for future use.
