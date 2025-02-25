
Step-1 
>>git clone https://github.com/yourusername/Automated-Model-Selection.git

>>cd Automated-Model-Selection

Step-2 | Install Dependecies
>>pip install -r requirements.txt

Step-3 | Train and Save the Model
>>python app/model_training.py

Step-4 | Run the Flask API
>>python app/model_api.py

Step-5 |  Test the API (Using cURL or Postman)
>>curl -X POST "http://127.0.0.1:5000/predict" -H "Content-Type: application/json" -d "{\"features\": [0.1, 0.2, 0.3, ..., 6.4]}"

If successful, it will return: {"prediction": 2}

Step-6 | Run the Streamlit Web Interface
>>streamlit run app/streamlit_app.py

This will open a web page where you can enter values and get predictions.
