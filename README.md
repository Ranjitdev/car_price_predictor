# Car Price Predictor

## Overview of Amazon sales data analysis: -  
**Before sellling a second hand car it is necessary to know the market price of second hand cars depending on its credentials like the age of the car or how long it drived. it is very painful to find a good source of second hand car price predictor. so i made this application for the same.**

## Data location: -  
- description: [Here](data\EDA.ipynb)  
- data:
  - test data: [Here](artifacts\test.csv)
  - train data: [Here](artifacts\train.csv)
  - raw data: [Here](artifacts/raw_data.csv)
  - preprocessor: [Here](artifacts\preprocessor.pkl)
  - model: [Here](artifacts\model.pkl)
  - train test report: [Here](artifacts\training_report.csv)
- project notebook: [notebook](data\EDA.ipynb)  
- data details: [report](data/report.html)  


## Setup: -
  - Run for create virtual environment
    - > conda create -p venv python=3.10 -y
  - Run before start application: -
    - > pip install -r requirements.txt
  - Run for create docker image: -
    - > docker build -t ranjitkundu/car_price_predictor:v1 .
  - Show docker images: -
    - > docker images
  - Run the docker image in container: - 
    - > docker run -p 8501:8501 ranjitkundu/car_price_predictor:v1
  - Initiate GitHub: -
    - > git init
  - Add all files in GitHub: -
    - > git add .
  - Commit code in GitHub: -
    - > git commit -m "message"
  - Push code in GitHub: -
    - > git push -u origin main