import matplotlib.pyplot as plt
import numpy as np

# Classifier names
classifiers = ['Decision Tree', 'Gradient Boosting', 'K Neighbors', 'Linear SVC', 'Random Forest']

# Validation accuracies
validation_accuracies = [0.7573715949261801, 0.7089207735495945, 0.7353711790393013, 0.6791848617176128, 0.7594094406321481]

# Test accuracies
test_accuracies = [0.7664795175712207, 0.7141609482220836, 0.7409024745269287, 0.6869203576627158, 0.7671865252651279]

# Width of the bars
bar_width = 0.15

# Index for x-axis
index = np.arange(len(classifiers))

# Plotting
plt.figure(figsize=(10, 6))

bar1 = plt.bar(index - bar_width/2, validation_accuracies, bar_width, color='b', label='Validation Accuracy')
bar2 = plt.bar(index + bar_width/2, test_accuracies, bar_width, color='r', label='Test Accuracy')

plt.title('Accuracy of Different Classifiers')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.xticks(index, classifiers, rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
