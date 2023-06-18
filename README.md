Created setup tools for automate requirements    
created requirements.txt file for listing all requirements    
created main app.py file    
created src folder for all resources files    
created venv local virtual environment    
connected to git and uploaded the basic structure    
    
cleaned the data in jupyter notebook by analysing null values duplicate rows    
found some unimportant columns in data so i dropped those columns    
found some outliers in data which can affect badly in model    
after data cleaning done some analysis in data for inspecting correlations    
    
after doing that created a model dictionary and parameters tuning dictionary    
with those dictionaries trained the data and tested the score   

now created data ingestion file for injecting data in artifacts
data transformation file for transforming the data with onehotencoder and used standard scaler for scaling   
then model trainer file will train the model and saves the trainer in artifacts

then created custom exception for handling the errors and logger for logging
utils file had the files used many times in various scripts

after that all component files joins together in init components

pipeline handles the input data to predict the final price   

streamlit used to create simple web interface
