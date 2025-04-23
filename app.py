from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import datetime
import psycopg2
from psycopg2 import pool
import functools

app = Flask(__name__, 
            static_folder='static',  # Explicitly set static folder
            static_url_path='/static')

# Set a secret key for session management
app.secret_key = 'your_very_secure_secret_key_here'  # Change this to a random string in production

# Temporary folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define class labels

class_names = ['COVID-19','NORMAL', 'PNEUMONIA']
# Hard-coded admin credentials
ADMIN_USERNAME = "kit"
ADMIN_PASSWORD = "aiml"

# PostgreSQL Database Configuration
# Replace these with your actual database credentials
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "lungpredict"
DB_USER = "postgres"
DB_PASSWORD = "aiml"

# Create connection pool
connection_pool = psycopg2.pool.SimpleConnectionPool(
    1, 10,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

# Login required decorator
def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view


# Load the model at startup
MODEL_PATH = "respiratory_disease_classifier.keras"
model = load_model(MODEL_PATH, compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def preprocess_image(image_path, img_size=(224, 224)):
    # Load and preprocess the image
    img = Image.open(image_path)
    
    # Ensure image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Use EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to save prediction to database
def save_prediction(name, age, gender, prediction, disease_name, confidence, image_path):
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO predictions 
                (name, age, gender, prediction_result, disease_name, confidence, image_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (name, age, gender, prediction, disease_name, confidence, image_path))
            conn.commit()
    except Exception as e:
        print(f"Error saving prediction: {e}")
        conn.rollback()
    finally:
        connection_pool.putconn(conn)

# Function to get all predictions from database
def get_predictions():
    conn = connection_pool.getconn()
    predictions = []
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT name, age, gender, prediction_result, disease_name, 
                       confidence, image_path, prediction_date 
                FROM predictions 
                ORDER BY prediction_date DESC
            ''')
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                prediction = dict(zip(columns, row))
                # Convert the datetime object to string for template rendering
                prediction['prediction_date'] = prediction['prediction_date'].strftime("%Y-%m-%d %H:%M")
                predictions.append(prediction)
    except Exception as e:
        print(f"Error retrieving predictions: {e}")
    finally:
        connection_pool.putconn(conn)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/user_details')
def user_details():
    return render_template('user_details.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Get user details from the form
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')
    
    # Pass these details to the upload image page
    return render_template('upload_image.html', name=name, age=age, gender=gender)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check credentials against hardcoded values
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('history'))
        else:
            error = 'Invalid username or password. Please try again.'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    # Get prediction history from database
    history_data = get_predictions()
    return render_template('history.html', history_data=history_data, username=session.get('username'))

@app.route('/process_image', methods=['POST'])
def process_image():
    # Collect user details from hidden fields
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')

    # Check if an image is uploaded
    if 'image' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Real prediction logic using the deep learning model
        try:
            # Preprocess the uploaded image
            processed_image = preprocess_image(filepath)
            
            # Make prediction
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction[0])
            disease_name = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index]) * 100
            
            # Determine positive/negative result
            prediction_result = "Positive" if disease_name != "NORMAL" else "Negative"
            
            # Get relative image path for database storage
            relative_image_path = f'/static/uploads/{filename}'
            
            # Save the prediction to database
            confidence_str = f"{confidence:.2f}%"
            save_prediction(name, int(age), gender, prediction_result, disease_name, 
                           confidence_str, relative_image_path)
            
            return render_template('prediction.html', 
                                name=name, 
                                age=age, 
                                gender=gender, 
                                image_url=relative_image_path, 
                                prediction=prediction_result, 
                                disease_name=disease_name,
                                confidence=confidence_str)
        except Exception as e:
            return f"Error in prediction: {str(e)}", 500

    return "Invalid file format", 400

# Clean up database connections when app exits
@app.teardown_appcontext
def close_db_pool(error):
    if hasattr(app, 'connection_pool'):
        connection_pool.closeall()

if __name__ == '__main__':
    app.run(debug=True)