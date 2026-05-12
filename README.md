# SMART-CROP-RECOMMENDATION-SYSTEM-USING-ENSEMBLE-LEARNING-FUZZY-LOGIC-AND-EXPLAINABLE-AI


An AI-powered precision agriculture project that recommends the most suitable crops based on soil nutrients and environmental conditions using Ensemble Machine Learning, Fuzzy Logic, and Explainable AI.

Overview

Modern agriculture increasingly depends on intelligent decision-support systems to improve productivity and reduce uncertainty in farming practices. Traditional crop selection methods often rely on experience-based judgment, which may not fully account for variations in soil composition, climate conditions, and environmental factors.

This project proposes a Smart Crop Recommendation System that combines multiple machine learning models with fuzzy logic and explainable artificial intelligence to generate accurate, transparent, and practical crop recommendations.

The system accepts agricultural input parameters such as Nitrogen, Phosphorus, Potassium, temperature, humidity, pH, and rainfall, then predicts the most suitable crops along with confidence scores, soil health analysis, and feature-level explanations.

Features
Ensemble Machine Learning

The system uses a Soft Voting Ensemble consisting of:

Random Forest
Gradient Boosting
K-Nearest Neighbors (KNN)
Naive Bayes

Combining multiple classifiers improves predictive accuracy and reduces model bias.

Top-3 Crop Recommendations

Instead of returning only one crop, the system provides:

Top three recommended crops
Confidence percentages
Ranked prediction results

This allows users to compare alternatives and make flexible decisions.

Fuzzy Logic-Based Soil Health Analysis

The project includes a fuzzy inference mechanism using trapezoidal membership functions to evaluate soil quality under uncertain environmental conditions.

The system computes a Soil Health Index (SHI) that reflects overall soil suitability.

Explainable AI using SHAP

SHAP (SHapley Additive Explanations) is integrated to improve transparency.

The system explains:

Which features influenced the prediction
Positive and negative feature contributions
Feature importance visualization

This transforms the model from a black-box system into an interpretable agricultural advisory tool.

Seasonal Crop Mapping

Each recommended crop is mapped to:

Kharif season
Rabi season
Zaid season

This ensures agronomic relevance of recommendations.

Input Validation and Alerts

The application validates all user inputs and generates warning messages for unrealistic values.

Example:

Excessive nitrogen level
Invalid rainfall range
Abnormal pH values
System Architecture

The system follows a multi-stage pipeline:

User Input Acquisition
Input Validation
Feature Preprocessing
Ensemble Prediction
Soft Voting Aggregation
Top-3 Crop Selection
Fuzzy Soil Health Evaluation
SHAP Explainability
Seasonal Mapping
Result Visualization
Technologies Used
Programming Language
Python
Framework
Flask
Machine Learning Libraries
Scikit-learn
SHAP
Data Processing
NumPy
Pandas
Visualization
Matplotlib
Frontend
HTML
CSS
Dataset

The project uses the Crop Recommendation Dataset from the UCI Machine Learning Repository.

Input Features
Nitrogen (N)
Phosphorus (P)
Potassium (K)
Temperature
Humidity
pH
Rainfall
Output
Recommended Crop Label

The dataset contains:

2,200 samples
22 crop classes
Balanced class distribution
Machine Learning Methodology
Soft Voting Ensemble

Each classifier predicts class probabilities independently.

The final prediction is calculated by averaging probabilities from all models.

Ensemble Formula

P(y | x) = (1/M) × Σ Pi(y | x)

Where:

Pi(y | x) = probability from each classifier
M = number of classifiers

The crop with the highest aggregated probability is selected.

Fuzzy Logic Methodology

The project uses trapezoidal membership functions to model uncertainty in agricultural inputs.

The fuzzy logic module:

Handles gradual transitions
Avoids rigid thresholding
Produces realistic soil quality evaluation

The final Soil Health Index is calculated using weighted membership scores.

Explainable AI

SHAP values are generated for every prediction.

The explainability module helps users understand:

Why a crop was recommended
Which parameters influenced the result
Feature contribution strength
Results

The proposed ensemble achieved higher accuracy than individual classifiers.

Model	Accuracy
Random Forest	98.00%
Gradient Boosting	99.00%
KNN	96.00%
Naive Bayes	95.00%
Proposed Ensemble	99.55%
Project Structure
Smart-Crop-Recommendation-System/
│
├── app_fixed.py
├── model.pkl
├── rf_model.pkl
├── encoder.pkl
├── features.pkl
├── requirements.txt
├── README.md
└── dataset/
Installation
Clone Repository
git clone https://github.com/your-username/smart-crop-recommendation-system.git
Navigate to Project Directory
cd smart-crop-recommendation-system
Install Dependencies
pip install -r requirements.txt
Run Application
python app_fixed.py
Future Enhancements

Possible future improvements include:

IoT sensor integration
Real-time weather API support
Mobile application deployment
Deep learning integration
Market-price aware recommendations
Multilingual support
Edge device deployment
Federated learning for privacy preservation
Applications
Precision Agriculture
Smart Farming
Agricultural Decision Support
Soil Health Monitoring
Sustainable Crop Planning
Conclusion

The Smart Crop Recommendation System demonstrates how Ensemble Learning, Fuzzy Logic, and Explainable AI can be combined to create an accurate, transparent, and practical agricultural decision-support platform.

By integrating predictive intelligence with interpretability and uncertainty handling, the project provides a meaningful contribution toward next-generation precision agriculture systems.

Author

Hemanth Kumar
VIT-AP University

License

This project is developed for academic and research purposes.
