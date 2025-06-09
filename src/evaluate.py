import matplotlib.pyplot as plt

accuracies = {
    'Naive Bayes': 69.75,
    'RNN': 60.00,
    'LSTM': 63.75,
    'CNN': 69.25,
    'SVM': 83.75,
    'KNN': 60.00
}

plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/model_accuracy_comparison.png')
