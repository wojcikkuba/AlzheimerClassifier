import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importing the dataset
df = pd.read_csv('alzheimer.csv')

# Filling the values of NaN with the medians of the corresponding columns
ses_median = np.median(df['SES'][~np.isnan(df['SES'])])
mmse_median = np.median(df['MMSE'][~np.isnan(df['MMSE'])])
df['SES'].fillna(ses_median, inplace=True)
df['MMSE'].fillna(mmse_median, inplace=True)

# Transforming a qualitative feature into a logical value
mask = (df['Group'].values == 'Nondemented') | (df['Group'].values == 'Converted')
df['Group'][mask] = 0
df['Group'][~mask] = 1

# Splitting the dataset
y = df['Group'].values.astype(int)
X = df.drop(columns=['Group','M/F']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Applying PCA
pca = PCA(n_components=4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Training SVM with PCA data
classifier = SVC(kernel = 'linear')
classifier.fit(X_train_pca, y_train)

# Predicting the test set result with PCA data
y_pred_pca = classifier.predict(X_test_pca)
print('Accuracy score with default linear kernel and PCA:')
print(metrics.accuracy_score(y_test, y_pred_pca))

