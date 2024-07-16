# Resume Screening App

This application is a web-based tool for screening resumes using machine learning models. It classifies resumes into various job categories based on their content.
![image](https://github.com/user-attachments/assets/6cb225ba-7ff2-44c8-a09d-0389c5ad93b4)
## Overview

The application uses the following key components:
- **Natural Language Processing (NLP)**: Used for cleaning and processing the text from resumes.
- **Machine Learning Model**: A trained classifier (K-Nearest Neighbors) to predict the job category of the resume.
- **TF-IDF Vectorizer**: Used to transform the resume text into numerical features suitable for model prediction.
- **Streamlit**: A framework for building and deploying the web app.

## Steps Performed

### 1. Import Necessary Libraries

```python
import nltk  # Natural Language Processing Toolkit
import streamlit as st  # Framework for creating the web app
import pickle  # For loading the pre-trained models
import re  # For regular expressions and text cleaning
```

### 2. Download NLTK Data

Ensure that the necessary NLTK data is available for text processing.

```python
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Load Pre-trained Models

Load the pre-trained classifier and TF-IDF vectorizer from serialized files.

```python
clf = pickle.load(open('clf.pkl', 'rb'))  # Classifier model
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # TF-IDF vectorizer
```

### 4. Define Text Cleaning Function

A function to clean the resume text by removing URLs, special characters, and unnecessary whitespace.

```python
def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
```

### 5. Web Application

#### a. Title of the App

Set the title of the web application.

```python
st.title("Resume Screening App")
```

#### b. File Upload

Allow users to upload a resume file (either in `.txt` or `.pdf` format).

```python
uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])
```

#### c. Read and Process Uploaded File

Read the uploaded file and decode it to a readable text format. If UTF-8 decoding fails, use 'latin-1'.

```python
if uploaded_file is not None:
    try:
        resume_bytes = uploaded_file.read()  # Read the file as bytes
        resume_text = resume_bytes.decode('utf-8')  # Decode bytes to text using UTF-8
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')  # Fallback decoding with 'latin-1'
```

#### d. Clean Resume Text

Clean the text of the uploaded resume.

```python
cleaned_resume = clean_resume(resume_text)
st.write(cleaned_resume)
```

#### e. Transform Text Using TF-IDF

Transform the cleaned resume text into numerical features using the TF-IDF vectorizer.

```python
input_features = tfidf.transform([cleaned_resume])
```

#### f. Predict Job Category

Use the classifier to predict the job category of the resume.

```python
prediction_id = clf.predict(input_features)[0]
st.write(prediction_id)
```

#### g. Map Prediction to Category Name

Map the predicted category ID to a human-readable category name and display it.

```python
category_mapping = {
    15: "Java Developer",
    23: "Testing",
    8: "DevOps Engineer",
    20: "Python Developer",
    24: "Web Designing",
    12: "HR",
    13: "Hadoop",
    3: "Blockchain",
    10: "ETL Developer",
    18: "Operations Manager",
    6: "Data Science",
    22: "Sales",
    16: "Mechanical Engineer",
    1: "Arts",
    7: "Database",
    11: "Electrical Engineering",
    14: "Health and Fitness",
    19: "PMO",
    4: "Business Analyst",
    9: "DotNet Developer",
    2: "Automation Testing",
    17: "Network Security Engineer",
    21: "SAP Developer",
    5: "Civil Engineer",
    0: "Advocate",
}

category_name = category_mapping.get(prediction_id, "Unknown")
st.write("Predicted Category:", category_name)
```

### 6. Run the Application

Ensure the web app runs when the script is executed.

```python
if __name__ == "__main__":
    main()
```

This README file provides a step-by-step description of the code and its functionality, helping users understand and utilize the Resume Screening App effectively.
