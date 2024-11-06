from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__,template_folder='C:\\Users\\Inbaraj\\OneDrive\\Desktop\\spam Email project\\template')


# Load the trained model
model = joblib.load('spam_model.pkl')

@app.route('/')
def home():
    # Render the index.html file located in the templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    message = data['message']
    prediction = model.predict([message])
    result = "It is spam" if prediction[0] == 1 else "It is ham"
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True) 