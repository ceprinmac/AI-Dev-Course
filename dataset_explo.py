import pickle  
import numpy as np  
from sklearn.model_selection import train_test_split  
import tarfile  
import matplotlib.pyplot as plt

# Extracting the Dataset
tar = tarfile.open(r"C:/AIDev/DataExplo/cifar-10-python.tar.gz")
tar.extractall()
tar.close()

# Define a function to load the batch file
def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

# Load dataset batch files
data_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle('cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle('cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle('cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle('cifar-10-batches-py/data_batch_5')

# Combine the loaded batches into a single dataset
X_train = np.concatenate([
    data_batch_1[b'data'],
    data_batch_2[b'data'],
    data_batch_3[b'data'],
    data_batch_4[b'data'],
    data_batch_5[b'data']
])
y_train = np.concatenate([
    data_batch_1[b'labels'],
    data_batch_2[b'labels'],
    data_batch_3[b'labels'],
    data_batch_4[b'labels'],
    data_batch_5[b'labels']
])

# Load the test batch
test_batch = unpickle('cifar-10-batches-py/test_batch')
X_test = test_batch[b'data']
y_test = np.array(test_batch[b'labels'])

# Reshape the data
X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Verify the dataset extraction.
print("Dataset extracted successfully!")

# Check the dataset shape
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

label_names = {
    0: "Frog",
    1: "Bird",
    2: "Dog",
    3: "Dog",
    4: "Dog",
    5: "Frog",
    6: "Bird",
    7: "Dog",
    8: "Dog",
    9: "Dog",
    10: "Frog",
    11: "Bird",
    12: "Dog",
    13: "Dog",
    14: "Dog",
}

# Visualize the images
fig, axes = plt.subplots(3, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i])
    ax.set_title(f"Label: {y_train[i]} \n {label_names[y_train[i]]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

#Verify Class Labels
unique_labels = np.unique(y_train)
print("Unique class labels: ", unique_labels)



# Explanation of output:

# X_train has a shape of (40000, 32, 32, 3), which means it contains 40,000 images for training. Each image has a size of 32x32 pixels with 3 color channels (RGB).
# y_train has a shape of (40000,), which corresponds to the labels for the 40,000 training images. Each label represents the class/category of the corresponding image in X_train.
# X_val has a shape of (10000, 32, 32, 3), indicating it contains 10,000 images used for validation. The dimensions of each image are the same as in X_train.
# y_val has a shape of (10000,) and contains the corresponding labels for the validation images in X_val.
# X_test has a shape of (10000, 32, 32, 3), representing 10,000 images that are part of the test set. Similar to X_train and X_val, each image in X_test has dimensions of 32x32 pixels with 3 color channels.
# y_test has a shape of (10000,) and provides the labels for the test images in X_test.
# Unique class labels: [0 1 2 3 4 5 6 7 8 9]: As a reminder, in the CIFAR-10 dataset, there are a total of 10 different classes or categories, represented by the numbers 0 to 9.

# This code performs several tasks in the following sequence to handle the CIFAR-10 dataset:

# Extracts the dataset from a compressed file.
# Loads and combines the individual batch files (individual subsets of data within a dataset) into a single dataset for training. The test batch is loaded separately.
# Reshapes the data into the desired format to prepare it for further processing.
# Splits the dataset into training and validation sets.
# Confirms the successful extraction and verifies the shapes of the datasets.
# Visualizations are created to display a subset of the training images along with their labels using matplotlib.
# Prints the unique class labels present in the training set.