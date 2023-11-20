# DeKUTECPredictor
This is an ML model for energy consumption prediction for Dedan Kimathi University of Technology.

# Install requirements
pip install -r requirements.txt

# Edit the input_data.json file, specifically the number of students and the features option, according to the number of students in the school.

# Start the server
python ECPredictor.py

# You can either choose to run a POST request in the web browser with an API extension like Talend API, or use the terminal.
curl -X POST -H "Content-Type: application/json" -d @input_data.json http://localhost:5000/predict
