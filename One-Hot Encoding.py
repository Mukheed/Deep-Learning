#exp-12
from sklearn.preprocessing import OneHotEncoder
import numpy as np
# Sample categorical sequence data
categorical_data = ['cat', 'dog', 'cat', 'bird', 'dog']
# Reshape data to fit OneHotEncoder requirements
categorical_data = np.array(categorical_data).reshape(-1, 1)
# Initialize OneHotEncoder
encoder = OneHotEncoder()
# Fit and transform the data
one_hot_encoded_data = encoder.fit_transform(categorical_data).toarray()
print("One Hot Encoded Data:")
print(one_hot_encoded_data)
