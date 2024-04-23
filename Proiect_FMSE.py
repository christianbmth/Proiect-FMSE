#Power of data in Quantum ML

from qiskit import QuantumCircuit, Aer, transpile, assemble # type: ignore
from qiskit.visualization import plot_histogram # type: ignore
import numpy as np

# Define the classical data
classical_data = np.array([0.2, 0.4, 0.6, 0.8])

# Define the quantum circuit
qc = QuantumCircuit(2, 2)

# Encode the classical data into quantum states
for i, data_point in enumerate(classical_data):
    angle = 2 * np.arcsin(np.sqrt(data_point))
    qc.ry(angle, i)

# Measure the qubits to retrieve classical information
qc.measure([0, 1], [0, 1])

# Simulate the quantum circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
qobj = assemble(compiled_circuit)
result = simulator.run(qobj).result()
counts = result.get_counts(qc)

# Plot the measurement results
plot_histogram(counts)

#Kernel Trick

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the Iris dataset (a simple example dataset)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with a radial basis function (RBF) kernel
# The RBF kernel is commonly used for non-linear classification tasks
classifier = SVC(kernel='rbf', gamma='scale')

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#SVM Algorithm

from sklearn import svm
import numpy as np

# Define the dataset
X_train = np.array([[5, 25], [7, 30], [9, 35], [11, 40]])  # Features: weight and height
y_train = np.array(['Cat', 'Cat', 'Dog', 'Dog'])           # Labels: Cat or Dog

# Create an SVM classifier
classifier = svm.SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(X_train, y_train)

# Define new data points for prediction
X_new = np.array([[6, 28], [10, 38]])

# Make predictions on the new data points
predictions = classifier.predict(X_new)

# Output the predictions
for i, pred in enumerate(predictions):
    print(f"Prediction for data point {i+1}: {pred}")
	
