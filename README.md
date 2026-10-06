# Task 4 - Classification with Logistic Regression

## AI & ML Internship

This project completes **Task 4: Classification with Logistic Regression** using the **Breast Cancer Wisconsin (Diagnostic)** dataset.

### Objective
Build a binary classification model using Logistic Regression and evaluate it using:
- Confusion Matrix
- Precision
- Recall
- ROC-AUC
- Threshold tuning
- Sigmoid function visualization

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Project Structure

```text
Task_4_Logistic_Regression_Classification/
├── data/
│   └── breast_cancer_wisconsin.csv
├── notebooks/
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sigmoid_curve.png
│   ├── threshold_comparison.csv
│   └── RESULTS.md
├── src/
│   └── logistic_regression.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
python3 src/logistic_regression.py
```

The script trains the model, evaluates the default 0.50 threshold, tunes the threshold for high recall, and saves plots/results in `outputs/`.

## Methodology

1. Load the Breast Cancer Wisconsin dataset.
2. Separate features and binary target.
3. Split data into training and testing sets using an 80/20 split.
4. Standardize features with `StandardScaler`.
5. Train a Logistic Regression classifier.
6. Generate class probabilities with `predict_proba`.
7. Evaluate the model using accuracy, precision, recall, confusion matrix, and ROC-AUC.
8. Tune the classification threshold and compare the effect on precision/recall.
9. Visualize the sigmoid function and ROC curve.

## Key Concepts

### Logistic Regression vs Linear Regression
Linear Regression predicts a continuous numerical value. Logistic Regression predicts the probability of a class and maps that probability between 0 and 1 using the sigmoid function.

### Sigmoid Function
The sigmoid function is:

`σ(z) = 1 / (1 + e^(-z))`

It converts a linear score into a probability between 0 and 1.

### Precision vs Recall
- **Precision:** Of the samples predicted positive, how many were actually positive?
- **Recall:** Of all actual positive samples, how many did the model identify?

### ROC-AUC
The ROC curve plots True Positive Rate against False Positive Rate across classification thresholds. AUC summarizes the model's ability to rank positive samples above negative samples. Higher AUC is generally better.

### Confusion Matrix
A confusion matrix contains:
- True Positive
- True Negative
- False Positive
- False Negative

### Class Imbalance
If one class is much more common than the other, accuracy can become misleading. Precision, recall, F1-score, ROC-AUC, PR-AUC, class weights, resampling, and threshold tuning can provide a better evaluation.

### Choosing a Threshold
The default threshold is commonly 0.50, but it should depend on the application's cost of false positives versus false negatives. This project demonstrates threshold tuning to prioritize high recall.

### Multi-Class Logistic Regression
Yes. Logistic Regression can be extended to multi-class classification using strategies such as one-vs-rest or multinomial logistic regression.

## Interview Questions

1. **How does logistic regression differ from linear regression?**  
   Linear regression predicts continuous values. Logistic regression estimates class probabilities and is used for classification.

2. **What is the sigmoid function?**  
   It maps any real-valued input to a value between 0 and 1, allowing the output to be interpreted as a probability.

3. **What is precision vs recall?**  
   Precision measures how many predicted positives are correct. Recall measures how many actual positives are detected.

4. **What is the ROC-AUC curve?**  
   ROC shows the trade-off between true positive rate and false positive rate at different thresholds. AUC measures overall ranking performance.

5. **What is a confusion matrix?**  
   It summarizes predictions as TP, TN, FP, and FN.

6. **What happens if classes are imbalanced?**  
   Accuracy may hide poor minority-class performance. Use appropriate metrics, stratification, class weights, resampling, or threshold tuning.

7. **How do you choose the threshold?**  
   Choose it according to the business or application objective. For medical screening, for example, high recall may be prioritized to reduce false negatives.

8. **Can logistic regression be used for multi-class problems?**  
   Yes, with one-vs-rest or multinomial approaches.

## Result

See `outputs/RESULTS.md` for the generated metrics and confusion matrix. The generated PNG files provide the visual evidence required for the task.

