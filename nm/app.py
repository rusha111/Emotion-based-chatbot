from flask import Flask, render_template, request
import joblib
import re
import cohere  # Cohere package for generating replies

app = Flask(__name__)

# Load the trained sentiment analysis model
model = joblib.load('chatbot_model.pkl')  # Make sure your model is trained on customer support-related data

# Cohere API key and client initialization
COHERE_API_KEY = 'OMmqpuHXJVBoQJnm6s27Hbbtj9k77MvehJCREhxU'
cohere_client = cohere.Client(COHERE_API_KEY)

# Utility function to clean and preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Customer support-related responses
def get_support_response(sentiment, user_message):
    # Depending on the sentiment, form a support-focused prompt
    if sentiment == 'happy':
        prompt = f"The customer seems happy. Provide a positive response to their inquiry: {user_message}"
    elif sentiment == 'sad':
        prompt = f"The customer seems sad. Offer sympathy and support to their issue: {user_message}"
    elif sentiment == 'angry':
        prompt = f"The customer seems angry. Acknowledge their frustration and offer solutions: {user_message}"
    elif sentiment == 'neutral':
        prompt = f"The customer seems neutral. Provide an informative response: {user_message}"
    else:
        prompt = f"Respond empathetically to the customer based on the following message: {user_message}"
    
    # Use Cohere to generate a response based on the sentiment and message
    response = cohere_client.generate(
        model="command",  # You can use a different model if needed
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    # Get the generated text from Cohere's response
    return response.generations[0].text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    user_message = ""
    bot_response = ""
    sentiment = ""

    if request.method == "POST":
        user_message = request.form.get("message")
        processed_message = preprocess_text(user_message)

        # Get the sentiment prediction for the user's message
        sentiment = model.predict([processed_message])[0]

        # Generate the support response based on sentiment and message
        bot_response = get_support_response(sentiment, user_message)

    return render_template("index.html", user_message=user_message, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
