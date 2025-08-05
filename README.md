# Crop Care Application

A smart and sustainable web application designed to assist farmers with plant disease detection, crop and fertilizer recommendations, and blockchain-based farmer data storage.

## Features

- Plant Disease Detection using Deep Learning (ResNet9)
- Crop Recommendation based on soil type
- Fertilizer Suggestion based on crop type
- Farmer Registration stored securely via SQLite
- Clean, user-friendly HTML interface
- Powered by PyTorch for model inference

## Technologies Used

- Python & Flask
- PyTorch & torchvision
- SQLite (via SQLAlchemy)
- HTML5 & CSS3
- PIL (Image processing)
- Git for version control

## How It Works

- Upload a crop image to get real-time disease prediction
- Select soil type to get recommended crops
- Enter crop name to get suitable fertilizer
- Register a farmer and store data securely in the database

## Project Structure

crop-care-application/
│
├── app.py # Main Flask application
├── models/
│ └── plant_disease_model.pth # Pretrained ResNet9 model
├── utils/
│ └── model.py # ResNet9 model architecture
├── templates/
│ ├── index.html
│ ├── crop.html
│ ├── crop-result.html
│ ├── fertilizer.html
│ ├── fertilizer-result.html
│ ├── block_index.html
│ ├── disease-result.html
│ └── success.html
├── static/
│ └── background.jpg # Background image
├── farmers.db # SQLite database file
├── requirements.txt # Python dependencies
└── README.md # Project overview


## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/crop-care-application.git
   cd crop-care-application
Install dependencies:


pip install -r requirements.txt



python app.py



http://127.0.0.1:5000/
Purpose for This project was built to:

Empower farmers with smart, AI-driven decision-making tools

Demonstrate the integration of deep learning models in a web application

Promote sustainable agriculture and digital solutions in farming

