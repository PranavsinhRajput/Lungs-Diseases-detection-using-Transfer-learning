# Flask Lung Disease Detection Web App

This is a Flask web application for lung disease detection using X-ray images. The app uses a pre-trained deep learning model to classify images of lungs and determine if the individual has a respiratory condition.

## Features
- Upload chest X-ray images for lung disease classification.
- The app predicts the likelihood of a disease such as Pneumonia or COVID-19.
- Displays results on the same page, along with relevant information.
- Stores images and metadata (age, gender) for record-keeping.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/PranavsinhRajput/Lungs-Diseases-detection-using-Transfer-learning
cd your-repo-name
```

### 2. Install Virtual Environment
Install the virtualenv package if you don’t have it already:

```
pip install virtualenv
```

### 3.Create a Virtual Environment
Create a new virtual environment inside your project folder:
```
virtualenv venv
```

### 4. Activate the Virtual Environment
Windows:
```
venv\Scripts\activate
```
macOS/Linux:
```
source venv/bin/activate
```

### 5. Install the Required Dependencies
Install all necessary Python packages from requirements.txt:
```
pip install -r requirements.txt
```

### 6. Run the Flask App
```
python app.py
```

