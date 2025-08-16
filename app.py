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
import cloudinary
import cloudinary.uploader
import tempfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, 
            static_folder='static',  # Explicitly set static folder
            static_url_path='/static')

# Set a secret key for session management
app.secret_key = os.getenv('SECRET_KEY', 'your_very_secure_secret_key_here')

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET'),
    secure=True
)

# Temporary folder to store uploaded images (for processing only)
UPLOAD_FOLDER = 'temp_uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure temp directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define class labels
class_names = ['COVID-19','NORMAL', 'PNEUMONIA']

# Hard-coded admin credentials (you can move these to env vars too)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'kit')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'aiml')

# Render PostgreSQL Database Configuration
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# Validate required environment variables
required_env_vars = ['DB_HOST', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Create connection pool for Render PostgreSQL
try:
    connection_pool = psycopg2.pool.SimpleConnectionPool(
        1, 10,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode='require'  # Render requires SSL
    )
    print("✅ Successfully connected to Render PostgreSQL!")
except Exception as e:
    print(f"❌ Error connecting to database: {e}")
    raise

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
try:
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ML model: {e}")
    raise

def preprocess_image(image_path, img_size=(224, 224)):
    """Load and preprocess the image for ML prediction"""
    img = Image.open(image_path)
    
    # Ensure image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Use EfficientNet preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_to_cloudinary(file_path, filename):
    """Upload image to Cloudinary and return secure URL"""
    try:
        # Upload to Cloudinary with folder organization
        response = cloudinary.uploader.upload(
            file_path,
            folder="Lungs X-ray Images",
            public_id=f"xray_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename.split('.')[0]}",
            resource_type="image",
            overwrite=True
        )
        return response.get('secure_url')
    except Exception as e:
        print(f"Error uploading to Cloudinary: {e}")
        return None

def save_prediction(name, age, gender, prediction, disease_name, confidence, image_url):
    """Save prediction result to Render PostgreSQL database"""
    conn = None
    try:
        # Check if connection pool is available
        if not connection_pool or connection_pool.closed:
            raise Exception("Database connection pool is not available")
            
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute('''
                INSERT INTO predictions 
                (name, age, gender, prediction_result, disease_name, confidence, image_path) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            ''', (name, age, gender, prediction, disease_name, confidence, image_url))
            conn.commit()
            print(f"✅ Prediction saved for {name}")
    except Exception as e:
        print(f"❌ Error saving prediction: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            try:
                connection_pool.putconn(conn)
            except:
                pass  # Pool might be closed

def get_predictions():
    """Get all predictions from Render PostgreSQL database"""
    conn = None
    predictions = []
    try:
        # Check if connection pool is available
        if not connection_pool or connection_pool.closed:
            print("❌ Database connection pool is not available")
            return predictions
            
        conn = connection_pool.getconn()
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
                
                # Validate image URL - if missing or invalid, use placeholder
                image_path = prediction.get('image_path', '')
                if not image_path or not image_path.startswith(('http://', 'https://')):
                    prediction['image_path'] = '/static/images/placeholder-xray.jpg'  # Default placeholder
                
                predictions.append(prediction)
    except Exception as e:
        print(f"❌ Error retrieving predictions: {e}")
    finally:
        if conn:
            try:
                connection_pool.putconn(conn)
            except:
                pass  # Pool might be closed
    return predictions

# Routes
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
        
        # Check credentials against environment variables
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('history'))
        else:
            error = 'Invalid username or password. Please try again.'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    """Display prediction history from Render PostgreSQL"""
    try:
        history_data = get_predictions()
        return render_template('history.html', 
                             history_data=history_data, 
                             username=session.get('username'))
    except Exception as e:
        flash(f'Error loading history: {str(e)}', 'error')
        return render_template('history.html', 
                             history_data=[], 
                             username=session.get('username'))

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process uploaded image, make ML prediction, upload to Cloudinary, save to Render PostgreSQL"""
    # Collect user details from hidden fields
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')

    # Validate user input
    if not all([name, age, gender]):
        flash('All user details are required.', 'error')
        return redirect(url_for('user_details'))

    # Check if an image is uploaded
    if 'image' not in request.files:
        flash('No file uploaded.', 'error')
        return redirect(url_for('upload_image'), code=307)
    
    file = request.files['image']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('upload_image'), code=307)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        
        # Use temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filename.rsplit(".", 1)[1].lower()}') as temp_file:
            file.save(temp_file.name)
            temp_filepath = temp_file.name

        try:
            # Preprocess the uploaded image for ML prediction
            processed_image = preprocess_image(temp_filepath)
            
            # Make prediction using the ML model
            prediction = model.predict(processed_image)
            predicted_class_index = np.argmax(prediction[0])
            disease_name = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index]) * 100
            
            # Determine positive/negative result
            prediction_result = "Positive" if disease_name != "NORMAL" else "Negative"
            
            # Upload image to Cloudinary
            print(f"🔄 Uploading image to Cloudinary...")
            cloudinary_url = upload_to_cloudinary(temp_filepath, filename)
            
            # Clean up temporary file
            os.unlink(temp_filepath)
            
            if not cloudinary_url:
                flash('Error uploading image to cloud storage. Please try again.', 'error')
                return redirect(url_for('upload_image'), code=307)
            
            print(f"✅ Image uploaded to Cloudinary: {cloudinary_url}")
            
            # Save the prediction to Render PostgreSQL database
            confidence_str = f"{confidence:.2f}%"
            save_prediction(name, int(age), gender, prediction_result, disease_name, 
                           confidence_str, cloudinary_url)
            
            flash(f'Prediction completed! Result: {prediction_result} ({disease_name})', 'success')
            
            return render_template('prediction.html', 
                                name=name, 
                                age=age, 
                                gender=gender, 
                                image_url=cloudinary_url, 
                                prediction=prediction_result, 
                                disease_name=disease_name,
                                confidence=confidence_str)
                                
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_filepath):
                os.unlink(temp_filepath)
            print(f"❌ Error in prediction process: {str(e)}")
            flash(f'Error processing image: {str(e)}', 'error')
            return redirect(url_for('upload_image'), code=307)

    else:
        flash('Invalid file format. Please upload PNG, JPG, or JPEG files only.', 'error')
        return redirect(url_for('upload_image'), code=307)

# Database health check route
@app.route('/health')
def health_check():
    """Check if database connection is working"""
    try:
        # Check if connection pool is available
        if not connection_pool or connection_pool.closed:
            return {'status': 'unhealthy', 'error': 'Connection pool is closed'}, 500
            
        conn = connection_pool.getconn()
        with conn.cursor() as cursor:
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
        connection_pool.putconn(conn)
        return {
            'status': 'healthy',
            'database': 'connected',
            'cloudinary': 'configured' if os.getenv('CLOUDINARY_CLOUD_NAME') else 'not configured'
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 500

# Test database connection route
@app.route('/test-db')
def test_db():
    """Test database connection and show recent predictions"""
    try:
        predictions = get_predictions()
        return {
            'status': 'success',
            'total_predictions': len(predictions),
            'recent_predictions': predictions[:3] if predictions else []
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}, 500

# Clean up database connections only when app exits (not on each request)
import atexit

def cleanup_connection_pool():
    """Close connection pool when app shuts down"""
    try:
        if connection_pool:
            connection_pool.closeall()
            print("🔌 Database connection pool closed")
    except:
        pass

# Register cleanup function to run when app exits
atexit.register(cleanup_connection_pool)

if __name__ == '__main__':
    # Print startup information
    print("🚀 Starting Flask Lung Disease Prediction App...")
    print(f"📊 Database: {DB_HOST}")
    print(f"☁️  Cloudinary: {'✅ Configured' if os.getenv('CLOUDINARY_CLOUD_NAME') else '❌ Not configured'}")
    
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))