from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmers.db'
db = SQLAlchemy(app)

# Farmer registration model
class Farmer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    farm_type = db.Column(db.String(100), nullable=False)
    soil_type = db.Column(db.String(100), nullable=False)
    area = db.Column(db.String(100), nullable=False)
    region = db.Column(db.String(100), nullable=False)
    state = db.Column(db.String(100), nullable=False)
    district = db.Column(db.String(100), nullable=False)

# Load the plant disease detection model
disease_classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___healthy',
    'Potato___Late_blight', 'Tomato_Target_Spot',
    'Tomato_Tomato_mosaic_virus', 'Tomato_Tomato_YellowLeaf_Curl_Virus',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight',
    'Tomato_healthy', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu')))
disease_model.eval()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/block-index')
def block_index():
    return render_template('block_index.html')

@app.route('/register')  # <-- This is what was missing
def register():
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded.", 400
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = disease_model(img_tensor)
        _, predicted = torch.max(preds, 1)
        result = disease_classes[predicted.item()]
    return render_template('disease-result.html', prediction=result)

@app.route('/crop-result', methods=['POST'])
def crop_result():
    soil = request.form['soil'].lower()
    soil_map = {
        'loamy': 'Rice, Wheat',
        'sandy': 'Groundnut, Potato',
        'clay': 'Paddy, Sugarcane',
        'black': 'Cotton, Soybean'
    }
    recommended_crop = soil_map.get(soil, "No matching crop found.")
    return render_template('crop-result.html', crop=recommended_crop)

@app.route('/fertilizer-result', methods=['POST'])
def fertilizer_result():
    crop_name = request.form['crop'].lower()
    fert_map = {
        'rice': 'Urea + DAP',
        'wheat': 'Urea + MOP',
        'maize': 'NPK 20-20-20',
        'cotton': 'NPK + Potash'
    }
    fertilizer = fert_map.get(crop_name, "No fertilizer found.")
    return render_template('fertilizer-result.html', fertilizer=fertilizer)

@app.route('/register-farmer', methods=['POST'])
def register_farmer():
    data = request.form
    new_farmer = Farmer(
        name=data['name'],
        farm_type=data['farm_type'],
        soil_type=data['soil_type'],
        area=data['area'],
        region=data['region'],
        state=data['state'],
        district=data['district']
    )
    db.session.add(new_farmer)
    db.session.commit()
    return render_template("success.html", message="✅ Farmer registered successfully!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
