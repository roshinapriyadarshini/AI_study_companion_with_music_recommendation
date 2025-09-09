# it directly call the __init__.py file

from flask import request, render_template, url_for, redirect, jsonify, Response, Flask
# from mlapp import app
import google.generativeai as genai
import speech_recognition as sr
import pytesseract as pt
from PIL import Image
import html
import re
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import time


# Specify the full path to the Tesseract executable
pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

load_dotenv()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html',title='Welcome to TuneScribe')

@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/faq")
def faq():
    return render_template('faq.html')

@app.route("/features")
def features():
    return render_template('features.html')

@app.route("/media")
def media():
    return render_template('media.html')

@app.route("/contact")
def contact():
    return render_template('contact.html')

# @app.route("/face_recognition")
# def face_recognition():
#     return render_template('face_recognition.html')


@app.route("/index", methods=["GET","POST"])
def index():
    if request.method == "POST":
        input_method = request.form.get("input_method")
        num_questions = request.form.get("num_questions", 10)  # Default to 10 if not specified
        try:
            num_questions = int(num_questions)
        except ValueError:
            num_questions = 25  # Fallback to 25 if input is invalid
        if input_method == "Text":
            user_input = request.form.get("user_input")
            qa = generate_paragraph(user_input, num_questions)
            return render_template("result.html", qa=qa)  # Directly render result without redirect
        elif input_method == "Image":
            uploaded_file = request.files.get("uploaded_file")
            if uploaded_file:
                image = Image.open(uploaded_file)
                user_input = get_text_from_image(image)
                qa = generate_paragraph(user_input, num_questions)
                return render_template("result.html", qa=qa)  # Directly render result without redirect

    return render_template("index.html")

    #     if input_method == "Text":
    #         user_input = request.form.get("user_input")
    #         return redirect(url_for("result", input_text=user_input, num_questions=num_questions))
    #     elif input_method == "Image":
    #         uploaded_file = request.files.get("uploaded_file")
    #         if uploaded_file:
    #             image = Image.open(uploaded_file)
    #             user_input = get_text_from_image(image)
    #             return redirect(url_for("result", input_text=user_input, num_questions=num_questions))
    # return render_template("index.html")
            

# @app.route("/result")
# def result():
#     input_text = request.args.get("input_text", "")
#     num_questions = request.args.get("num_questions", 10)  # Default to 10 if not specified
#     try:
#         num_questions = int(num_questions)
#     except ValueError:
#         num_questions = 10  # Fallback to 10 if input is invalid

#     if input_text:
#         qa = generate_paragraph(input_text, num_questions)
#         return render_template("result.html", qa=qa)
#     return render_template("result.html", error="No input provided.")

def get_text_from_image(image):
    input_text = pt.image_to_string(image)
    return input_text.strip()

# Function to generate questions and answers using Google Generative AI
def generate_paragraph(user_input, num_questions):
    genai.configure(api_key="AIzaSyA_KEb2yJDOsQS0ZuerUi-zSI72RCHV7L8")
    prompt = (
        f"Given the following text:\n\n{user_input}\n\n"
        f"Please extract {num_questions} meaningful questions that can be formed based on the content. "
        "For each question, provide a concise answer directly derived from the text. "
        "Format the output as a list of questions followed by their respective answers, ensuring clarity and relevance.\n\n"
        "Additionally, generate a concise summary of the text."
    )
    generation_config = {
        "temperature": 0.45,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 2500,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    response = model.generate_content(prompt)
    formatted_response = format_response(response.text)  # Format the response
    return formatted_response


def generate_response(user_input):
    """Generate a response using the Gemini API."""
    prompt = f"{user_input}"
    generation_config = {
        "temperature": 0.45,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 1500,
        "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
    )
    response = model.generate_content(prompt)
    formatted_response = format_response(response.text)  # Format the response
    return formatted_response

def format_response(text):
     # Remove markdown symbols like ##, **, and other unwanted characters
    cleaned_text = re.sub(r'(\*\*|##)', '', text)
    cleaned_text = cleaned_text.replace('*', '-')

    # Further clean up the text (optional)
    cleaned_text = html.escape(cleaned_text)  # Sanitize input for HTML safety

    # Format the cleaned text into paragraphs for better readability
    formatted_text = cleaned_text.replace('\n', '<br>')  # Replace newlines with HTML line breaks
    return formatted_text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages from the user."""
    user_input = request.form['message']
    response_text = generate_response(user_input)
    return jsonify({"response": response_text})

# import random
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np

# @app.route("/chatbot")
# def chatbot():
#     return render_template('chatbot.html')
# # Dummy music recommendation data
# music_recommendations = [
#     {"title": "Focus Beats", "artist": "Study Tunes", "genre": "Instrumental"},
#     {"title": "Calm Vibes", "artist": "Chillout Lounge", "genre": "Ambient"},
#     {"title": "Concentration Flow", "artist": "Mindful Music", "genre": "Lo-fi"},
# ]

# # Training data for chatbot responses
# training_data = [
#     ("hello", "Hello! How can I assist you with your studies today?"),
#     ("how are you", "I'm an AI companion, always here to help you with your study needs!"),
#     ("recommend music", "Sure! Here's a music recommendation to help you focus."),
#     ("thank you", "You're welcome! Happy studying!"),
#     ("what is ai", "AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines."),
#     ("recommend me some study tips", "Break your study sessions into smaller chunks, use flashcards, and take regular breaks!"),
#     ("what is machine learning", "Machine Learning is a branch of AI that allows computers to learn from data and make decisions."),
# ]

# # Extract features and labels for training
# texts, responses = zip(*training_data)
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(texts)
# y = np.arange(len(texts))

# # Train a simple K-Nearest Neighbors model
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(X, y)

# # Initialize NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# def clean_text(text):
#     """Cleans and preprocesses the input text."""
#     tokens = word_tokenize(text.lower())
#     tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
#     return " ".join(tokens)

# def generate_response(user_input):
#     """Generate a response using simple ML/NLP logic."""
#     cleaned_input = clean_text(user_input)
#     input_vector = vectorizer.transform([cleaned_input])
#     prediction = model.predict(input_vector)
#     response = responses[prediction[0]]
#     return response

# @app.route("/chat1", methods=["POST"])
# def chat1():
#     user_input = request.json.get("message")
#     response_text = generate_response(user_input)
#     return jsonify({"response": response_text})

# @app.route("/music", methods=["GET"])
# def recommend_music():
#     recommendation = random.choice(music_recommendations)
#     return jsonify(recommendation)


# Global variable to store current emotion
current_emotion = {'emotion': 'neutral', 'last_updated': time.time()}

# Load the pre-trained emotion detection model
try:
    with open('model/facialemotionmodel.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('model/facialemotionmodel.h5')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Spotify setup
client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')

if client_id and client_secret:
    spotify_credentials = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    sp = spotipy.Spotify(client_credentials_manager=spotify_credentials)
else:
    print("Warning: Spotify credentials not found")
    sp = None

def get_playlists(emotion):
    """Get multiple playlist recommendations based on emotion"""
    if not sp:
        return []
    
    mood_queries = {
        'happy': ['happy hits', 'feel good music', 'upbeat favorites'],
        'sad': ['sad songs', 'emotional healing', 'melancholy music'],
        'angry': ['rage beats', 'angry rock', 'intense music'],
        'fear': ['calming playlist', 'peaceful music', 'relaxing sounds'],
        'disgust': ['mood lifter', 'positive vibes', 'cheerful beats'],
        'surprise': ['party hits', 'exciting music', 'upbeat surprises'],
        'neutral': ['chill vibes', 'ambient music', 'focus playlist']
    }
    
    playlists = []
    queries = mood_queries.get(emotion, ['chill playlist'])
    
    for query in queries:
        try:
            results = sp.search(q=query, type='playlist', limit=1)
            if results and results['playlists']['items']:
                playlist = results['playlists']['items'][0]
                playlists.append({
                    'name': playlist['name'],
                    'url': playlist['external_urls']['spotify'],
                    'image': playlist['images'][0]['url'] if playlist['images'] else None
                })
        except Exception as e:
            print(f"Error fetching playlist for {query}: {e}")
    
    return playlists

@app.route('/get_current_emotion')
def get_current_emotion():
    """Get current emotion and recommended playlists"""
    emotion = current_emotion['emotion']
    playlists = get_playlists(emotion)
    return jsonify({
        'emotion': emotion,
        'playlists': playlists
    })

def preprocess_face(face):
    """Preprocess face image for model prediction."""
    try:
        # Ensure face dimensions are valid
        if face is None or face.shape[0] == 0 or face.shape[1] == 0:
            return None
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0  # Normalize pixel values
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        return face
    except Exception as e:
        print(f"Error during face preprocessing: {e}")
        return None

def generate_frames():
    """Generate frames with emotion detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)  # Open the camera feed
    
    if not camera.isOpened():
        print("Error: Could not open camera")
        return

    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame from camera.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        frame_emotion = None  # Initialize emotion for this frame
        for (x, y, w, h) in faces:
            try:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract the region of interest
                face_roi = gray[y:y + h, x:x + w]

                if model is not None:
                    processed_face = preprocess_face(face_roi)
                    if processed_face is not None:
                        prediction = model.predict(processed_face)
                        emotion_idx = np.argmax(prediction[0])
                        emotion = EMOTIONS[emotion_idx]
                        confidence = float(prediction[0][emotion_idx])

                        # Update frame emotion
                        frame_emotion = emotion

                        # Display emotion and confidence
                        text = f"Mood: {emotion.upper()} ({confidence:.2f})"
                        cv2.putText(frame, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face ROI: {e}")

        # Update current emotion if detected
        if frame_emotion:
            current_emotion['emotion'] = frame_emotion
            current_emotion['last_updated'] = time.time()

        # Encode the frame for video streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/face_recognition')
def face_recognition():
    """Render the face recognition template."""
    return render_template('face_recognition.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":    
    app.run(debug=True)