import matplotlib.pyplot as plt
import numpy as np
"""
vector_label1 = [-0.09789313,  0.12861505, -0.11870613, -1.74425781,  0.01419329,  1.27106166,  0.03139492, -0.11368603, -0.39989181]
vector_prediction1 = [-0.0968,  0.1351, -0.1257, -1.7621,  0.0285,  1.2618,  0.0436, -0.0481,  -0.3633]








vector_label1 = np.array(vector_label1)
vector_prediction1 = np.array(vector_prediction1)
diffrence = []
diffrence = vector_label1 - vector_prediction1
print("difference", diffrence)
"""

vectors_label = [
    [-0.09789313, 0.12861505, -0.11870613, -1.74425781, 0.01419329, 1.27106166, 0.03139492, -0.11368603, -0.39989181],
    [-0.03260346, 0.09387283, -0.05966107, -1.814043, 0.00570274, 1.235922, 0.2044418, -0.05439788, -0.14984419],
    [0.17292964, 0.48420027, -0.19589409, -2.0507977, 0.1656408, 0.6168301, 1.55783486, -0.00403172, 0.00973947],
    [-0.46920693, 0.8098488, -0.3878336, -1.4043442, 0.32884854, 0.9927318, 0.47299942, -0.04916036, -0.19325758],
    [0.10801413, 0.52824676, -0.01746686, -2.02584147, -0.04512902, 0.2964487, 0.17703225, 0.06681892, 0.02339216]
]

vectors_prediction = [
    [-0.0968, 0.1351, -0.1257, -1.7621, 0.0285, 1.2618, 0.0436, -0.0481, -0.3633],
    [-0.0236, 0.1176, -0.0466, -1.8105, 0.0144, 1.2535, 0.2335, -0.0650, -0.1175],
    [0.1743, 0.4811, -0.1770, -2.1091, 0.1338, 0.6482, 1.5720, 0.0049, 0.0262],
    [-0.4643, 0.7884, -0.3902, -1.4573, 0.3239, 1.0106, 0.4715, -0.0589, -0.1926],
    [0.1193, 0.5856, -0.0030, -2.0399, -0.0511, 0.3198, 0.1828, 0.0889, 0.0444]
]



# Calculate and print the differences using a for loop
for i in range(len(vectors_label)):
    label = np.array(vectors_label[i])
    prediction = np.array(vectors_prediction[i])
    difference = label - prediction
    print(f"Difference for vector {i + 1}: {difference}")


absolute_differences = [np.abs(np.array(label) - np.array(prediction)) for label, prediction in zip(vectors_label, vectors_prediction)]


plt.figure(figsize=(10, 6))

for i, diff in enumerate(absolute_differences):
    plt.plot(diff, label=f"Vector {i + 1}")

plt.title('Differences between Label and Prediction Vectors')
plt.xlabel('Element Index')
plt.ylabel('Difference Value')
plt.legend()
plt.grid(True)
plt.show()    









