# AlzheimerClassifier
Python project for classification of Alzheimer's disease using SVM

The classifier uses machine learning techniques (**SVM**, **PCA**) to predict whether a patient has Alzheimer's disease or not
based on various features extracted from clincal data, such as the volume of different brain regions and clinical dementia rating.

Added simple Flask interface for easy to use GUI. 

**To run the app:**

1. Build the Docker image:
   `docker build -t alzheimer-classifier .`

2. Run the container:
   `docker run -p 5000:5000 alzheimer-classifier`

The application will be available at `localhost:5000`
