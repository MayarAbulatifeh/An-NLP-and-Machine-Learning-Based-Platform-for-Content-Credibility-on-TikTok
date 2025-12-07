# NLP & Machine Learning Platform for TikTok Content Credibility Classification

This repository contains an end-to-end NLP and Machine Learning pipeline designed to classify TikTok video transcripts into three categories: **Claim**, **Supported Claim**, and **Opinion**.  
The project covers the complete workflow from data collection and curation to model development, evaluation, and deployment with a Flask API.

---

## 📌 Project Overview

The goal of this project was to explore whether linguistic patterns in short-form content can be used to assess the credibility of information.  
To achieve this, I built a structured pipeline that includes:

- Data curation and exploratory analysis  
- Data generation and dataset merging from differnet sources  
- Text preprocessing and feature engineering  
- Model selection and performance evaluation  
- Deployment of the final model through a Flask API  

The project demonstrates the practical implementation of NLP techniques and supervised learning methods in a realistic setting.

---

## 📂 Repository Structure
/Data Curation → Initial data exploration, inspection, and cleaning
/Data_Generation → Dataset merging, augmentation, and template-based generation
/Preprocessing → Text cleaning, normalization, tokenization, and TF-IDF preparation
/Model Selection → Model training, comparison, and evaluation notebooks
/Deployment → Flask application for inference
Documentation → Final report and project poster


---

## Methodology

### **1. Data Collection & Curation**
- Selected a TikTok credibility dataset from Kaggle  
- Performed exploratory data analysis to understand structure, distribution, and quality  
- Cleaned missing and inconsistent values  

### **2. Data Generation & Merging**
- Applied template-driven data generation to enrich the dataset  
- Merged multiple datasets to create the final modeling dataset  

### **3. Preprocessing & Feature Engineering**
- Applied regex-based text cleaning  
- Performed tokenization, stopword removal, and lemmatization  
- Generated TF-IDF representations using uni-grams and bi-grams  

### **4. Model Development & Selection**
Trained several supervised learning models & Word Embedding Techniques, including:
- Linear Support Vector Machine (**best performance**)  
- Logistic Regression  
- Multinomial Naive Bayes
- TF-IDF
- BOW
- Word2Vec
- FastText
- 
Evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

### **5. Deployment**
Developed a simple Flask API to serve the model and enable prediction on raw text inputs.

Example request:
POST /predict
{
  "text": "According to a 2023 report from the World Health Organization (WHO), regular physical activity can reduce the risk of heart disease by up to 30%. The American Heart Association also recommends at least 150 minutes of moderate exercise per week, confirming that staying active has long-term health benefits for the heart and overall wellbeing."
}

---

## Model Availability

The trained model file is not included in this repository due to size limitations.  
It can be provided upon request for users who wish to run the Flask API locally.

---

## Technologies Used

**Languages:** Python  
**Libraries:** pandas, numpy, scikit-learn, nltk/spaCy, matplotlib, seaborn 
**Tools:** Jupyter Notebook, Flask  

---

## 📈 Core Skills Demonstrated

- Data cleaning and preprocessing  
- NLP feature engineering (TF-IDF, lemmatization)  
- Supervised model training and evaluation  
- Dataset merging and augmentation  
- Structuring machine learning projects  
- Deploying ML models with Flask  

---

## Documentation

The repository includes:
- **Final project report (PDF)**  
- **Project poster** summarizing the workflow and findings  

---

## Future Improvements

Potential extensions include:
- Using transformer-based architectures (e.g., BERT)  
- Credibility scoring instead of classification  
- API containerization with Docker  
- Deployment to a cloud environment (AWS/GCP/Render)

---

## Contact

For inquiries regarding the model file or implementation details, feel free to reach out.
Email: mayarabulatifeh@gmail.com
Phone: +962 792139173

