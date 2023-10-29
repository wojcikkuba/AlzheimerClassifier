from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.options.mode.chained_assignment = None

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

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)

# Training SVM with PCA data
classifier = SVC(kernel='linear')
classifier.fit(X_pca, y)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = np.array(float_features)
    prediction_input = scaler.transform([final_features])
    prediction_input_pca = pca.transform(prediction_input)
    prediction = classifier.predict(prediction_input_pca)
    output = int(prediction[0])
    return render_template('index.html', prediction_text='Class prediction: {}'.format(output))

if __name__ == '__main__':
    app.run(host='0.0.0.0')
