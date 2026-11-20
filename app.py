import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import tensorflow as tf

# ── App config ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY']          = 'dental_secret_key_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dental.db'
app.config['UPLOAD_FOLDER']       = 'static/uploads'
app.config['MAX_CONTENT_LENGTH']  = 16 * 1024 * 1024   # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

db           = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'   # redirect to login if not authenticated

# ── Database model ────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80),  unique=True, nullable=False)
    email    = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Load best model (VGG16) ───────────────────────────────────────────────────
MODEL_PATH = 'models/VGG16_best.keras'
IMG_SIZE   = (96, 96)
CLASSES    = ["Cavity", "Fillings", "Impacted Tooth", "Implant", "Normal"]

# Treatment suggestions per class
TREATMENTS = {
    "Cavity": {
        "description": "Dental cavity (caries) detected — bacterial decay has damaged tooth enamel.",
        "treatment": [
            "Schedule an appointment with your dentist immediately",
            "Dental filling or composite restoration for small-medium cavities",
            "Root canal treatment if decay has reached the pulp",
            "Dental crown for large structural damage"
        ],
        "maintenance": [
            "Brush twice daily with fluoride toothpaste",
            "Floss daily to remove plaque between teeth",
            "Limit sugary and acidic food/drinks",
            "Use fluoride mouthwash",
            "Visit dentist every 6 months for checkup"
        ],
        "urgency": "High — visit dentist within 1–2 weeks"
    },
    "Fillings": {
        "description": "Existing dental filling detected — previously treated tooth restoration visible.",
        "treatment": [
            "Monitor filling condition at regular checkups",
            "Replace filling if cracks, wear, or leakage detected",
            "Consider ceramic/composite replacement for old amalgam fillings",
            "Dental crown if filling is large and tooth is weakened"
        ],
        "maintenance": [
            "Avoid chewing hard foods on filled teeth",
            "Brush gently around filled areas",
            "Regular dental X-rays to monitor filling integrity",
            "Avoid teeth grinding (wear a night guard if needed)"
        ],
        "urgency": "Low — routine 6-month checkup sufficient"
    },
    "Impacted Tooth": {
        "description": "Impacted tooth detected — tooth unable to fully erupt through the gum line.",
        "treatment": [
            "Consult an oral surgeon for evaluation",
            "Surgical extraction if causing pain or crowding",
            "Orthodontic treatment to create space for eruption",
            "Regular monitoring if asymptomatic"
        ],
        "maintenance": [
            "Keep the area around the impacted tooth clean",
            "Use antiseptic mouthwash to prevent infection",
            "Watch for signs of infection: swelling, pain, bad taste",
            "Follow up every 3–6 months with panoramic X-ray"
        ],
        "urgency": "Medium — consult dentist within 2–4 weeks"
    },
    "Implant": {
        "description": "Dental implant detected — artificial tooth root replacement visible.",
        "treatment": [
            "Maintain regular implant check-ups every 6 months",
            "Professional cleaning around implant crown",
            "Bone density monitoring via periodic X-rays",
            "Replace implant crown if worn (typically every 10–15 years)"
        ],
        "maintenance": [
            "Brush implant twice daily with soft-bristle brush",
            "Use interdental brushes or water flosser around implant",
            "Avoid smoking — reduces implant longevity",
            "Wear night guard if you grind teeth"
        ],
        "urgency": "Low — routine 6-month checkup sufficient"
    },
    "Normal": {
        "description": "No significant dental pathology detected — teeth appear healthy.",
        "treatment": [
            "Continue current oral hygiene routine",
            "Preventive sealants if recommended by dentist",
            "Professional cleaning every 6 months"
        ],
        "maintenance": [
            "Brush twice daily with fluoride toothpaste for 2 minutes",
            "Floss daily before brushing",
            "Use antiseptic mouthwash",
            "Stay hydrated — saliva protects teeth",
            "Eat a balanced diet low in sugar"
        ],
        "urgency": "None — maintain routine checkups"
    }
}

print("Loading VGG16 model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ── Helper functions ──────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension
    img_array = img_array / 255.0                    # normalize

    predictions = model.predict(img_array, verbose=0)
    pred_idx    = np.argmax(predictions[0])
    confidence  = float(predictions[0][pred_idx]) * 100

    # All class probabilities for display
    all_probs = {
        CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
        for i in range(len(CLASSES))
    }
    return CLASSES[pred_idx], confidence, all_probs

# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('home'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/about')
@login_required
def about():
    return render_template('about.html')

@app.route('/methodology')
@login_required
def methodology():
    model_results = [
        {'name': 'VGG16',          'accuracy': 90.33, 'f1': 0.71, 'status': 'best'},
        {'name': 'CNN',            'accuracy': 86.34, 'f1': 0.69, 'status': ''},
        {'name': 'MobileNetV2',    'accuracy': 83.68, 'f1': 0.64, 'status': ''},
        {'name': 'MLP',            'accuracy': 72.48, 'f1': 0.35, 'status': ''},
        {'name': 'EfficientNetB0', 'accuracy': 72.26, 'f1': 0.35, 'status': ''},
        {'name': 'ResNet50',       'accuracy': 69.77, 'f1': 0.47, 'status': ''},
    ]
    return render_template('methodology.html', model_results=model_results)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename    = secure_filename(file.filename)
            filepath    = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Run prediction
            predicted_class, confidence, all_probs = predict_image(filepath)
            treatment_info = TREATMENTS[predicted_class]

            return render_template('result.html',
                filename        = filename,
                predicted_class = predicted_class,
                confidence      = round(confidence, 2),
                all_probs       = all_probs,
                treatment       = treatment_info
            )
        else:
            flash('Allowed file types: PNG, JPG, JPEG', 'danger')
            return redirect(request.url)

    return render_template('predict.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username').strip()
        email    = request.form.get('email').strip()
        password = request.form.get('password')
        confirm  = request.form.get('confirm_password')

        if password != confirm:
            flash('Passwords do not match', 'danger')
            return redirect(request.url)

        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(request.url)

        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(request.url)

        hashed_pw = generate_password_hash(password)
        new_user  = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password')
        user     = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()    # creates dental.db with User table
    app.run(debug=True)