Task 4 - Logistic Regression Classification

Dataset: Breast Cancer Wisconsin (Diagnostic)
Train samples: 455
Test samples: 114
Number of features: 30

Default threshold: 0.50
Accuracy: 0.9825
Precision: 0.9861
Recall: 0.9861
ROC-AUC: 0.9954

Confusion Matrix at threshold 0.50:
[[41  1]
 [ 1 71]]

Tuned threshold: 0.36
Tuned accuracy: 0.9912
Tuned precision: 0.9863
Tuned recall: 1.0000

Classification Report at threshold 0.50:
              precision    recall  f1-score   support

           0       0.98      0.98      0.98        42
           1       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

