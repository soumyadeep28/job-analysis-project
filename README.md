# Job Skill Analyzer Project

A Flask-based web application that analyzes job descriptions to predict job scores and extract relevant skills using machine learning and NLP.

## 📋 Table of Contents
- [About](#about)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running with Docker](#running-with-docker)
- [Running Locally](#running-locally)
- [Project Structure](#project-structure)

## 📖 About
This application takes a job description as input and:
1.  Predicts a "job score" using a trained Machine Learning model.
2.  Identifies key skills and their importance based on a configuration file.
3.  Extracts additional skills using NLP techniques (spaCy).

## ⚠️ Prerequisites (Required Files)
The following files are **required** for the application to run but are **NOT** included in the git repository (due to size or file type). You must ensure these files are present in the project root directory:

1.  `job_all_best_model.pkl`: The trained ML model.
2.  `job_all_scaler.pkl`: The scaler object fitted on training data.
3.  `job_all_columns.json`: JSON file containing the list of columns used during training.
4.  `skills_config.json`: Configuration for skill importance mapping.

**Note:** If you are building the models from scratch, you will also need the raw dataset files (e.g., `job_skills.csv`), but they are not required for running the prediction app itself.

## 🐳 Running with Docker
The easiest way to run the application is using Docker.

### 1. Build the Docker Image
Run the following command in the project root:
```bash
docker build -t jd-skill-analyzer .
```

### 2. Run the Container
Run the container, mapping port 8000 (host) to 5000 (container):
```bash
docker run -p 8000:5000 jd-skill-analyzer
```
*(Note: The internal app runs on port 5000 or 8000 depending on configuration. If the above doesn't work, try mapping 8000:8000)*

### 3. Access the App
Open your browser and navigate to:
http://localhost:8000

## 💻 Running Locally

### 1. Create a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Run the Application
```bash
python app.py
```
The application will start on `http://0.0.0.0:8000`.

## 📂 Project Structure
- `app.py`: Main Flask application entry point.
- `jobanalysis.py`: Contains helper functions for feature engineering and skill extraction.
- `Dockerfile`: Configuration for building the Docker image.
- `requirements.txt`: Python package dependencies.
- `templates/`: HTML templates for the web interface.
- `*.pkl / *.json`: Model and configuration files (see Prerequisites).