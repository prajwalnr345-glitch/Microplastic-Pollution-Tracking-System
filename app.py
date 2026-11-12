import os, sqlite3, uuid
from datetime import datetime
from functools import wraps

import numpy as np
from flask import Flask, request, redirect, url_for, session, flash, jsonify, render_template_string
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Microplastic Pollution Tracking in Urban Water Bodies System - single-file Flask application.
# Replace your old app.py with this file. Keep your .keras model in the same folder.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "microplastic.db")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "pollution_classifier_final.keras")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-for-real-deployment")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

# IMPORTANT: confirm this order against the class order in your model training code.
CLASS_NAMES = ["High", "Low", "Moderate"]
IMG_SIZE = (224, 224)
model = None
model_load_message = "Classifier model has not been loaded."

def connect_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def initialize_database():
    with connect_db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS samples(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            location TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
            volume_liters REAL NOT NULL, particle_candidates INTEGER NOT NULL,
            concentration REAL NOT NULL, predicted_class TEXT, confidence REAL,
            image_name TEXT NOT NULL, created_at TEXT NOT NULL,
            location1 TEXT, latitude1 REAL, longitude1 REAL,
            location2 TEXT, latitude2 REAL, longitude2 REAL,
            location3 TEXT, latitude3 REAL, longitude3 REAL,
            location4 TEXT, latitude4 REAL, longitude4 REAL)""")
        # Add the four sampling-point fields to databases created by older versions.
        existing = {row["name"] for row in con.execute("PRAGMA table_info(samples)").fetchall()}
        for column, sql_type in [
            ("location1", "TEXT"), ("latitude1", "REAL"), ("longitude1", "REAL"),
            ("location2", "TEXT"), ("latitude2", "REAL"), ("longitude2", "REAL"),
            ("location3", "TEXT"), ("latitude3", "REAL"), ("longitude3", "REAL"),
            ("location4", "TEXT"), ("latitude4", "REAL"), ("longitude4", "REAL")
        ]:
            if column not in existing:
                con.execute(f"ALTER TABLE samples ADD COLUMN {column} {sql_type}")
        # Preserve older one-location records as Location 1.
        con.execute("""UPDATE samples SET location1=location, latitude1=latitude, longitude1=longitude
                       WHERE location1 IS NULL""")
initialize_database()

def require_login(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapped

def load_classifier():
    global model, model_load_message
    if model is not None:
        return model
    if not os.path.exists(MODEL_PATH):
        model_load_message = "Model file not found. Classification is unavailable; particle candidate counting can still run."
        return None
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        model_load_message = "Classifier model loaded."
        return model
    except Exception as exc:
        model_load_message = "Could not load model: " + str(exc)
        app.logger.exception("Model loading error")
        return None

def predict_class(image_path):
    m = load_classifier()
    if m is None:
        return None, None
    try:
        from tensorflow.keras.preprocessing import image as keras_image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        im = keras_image.load_img(image_path, target_size=IMG_SIZE)
        arr = keras_image.img_to_array(im)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        probabilities = np.asarray(m(arr, training=False))[0]
        if len(probabilities) != len(CLASS_NAMES):
            return None, None
        index = int(np.argmax(probabilities))
        return CLASS_NAMES[index], round(float(probabilities[index]) * 100, 2)
    except Exception:
        app.logger.exception("Prediction failed")
        return None, None

def count_image_candidates(image_path):
    # Approximate contour count only. It is NOT a validated microplastic detector.
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            return 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        min_area, max_area = max(4, int(h*w*0.000002)), int(h*w*0.08)
        return sum(1 for c in contours if min_area <= cv2.contourArea(c) <= max_area)
    except ImportError:
        return 0

BASE_HTML = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title or 'Microplastic Pollution Tracking in Urban Water Bodies System' }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--ink:#142b3b;--muted:#718395;--line:#e5edf3;--bg:#f4f7fb;--navy:#102b46;--blue:#4776f5;--cyan:#16b8c7;--purple:#8665e8;--pink:#e96ca8;--green:#16a887;--amber:#efa83d}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 90% 0%,#e8edff 0,transparent 28%),var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif}
header{height:74px;background:rgba(255,255,255,.92);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 max(calc((100% - 1320px)/2),24px);position:sticky;top:0;z-index:500}
.brand{font:800 13px Manrope;color:var(--navy);text-decoration:none;letter-spacing:0;max-width:78%;line-height:1.25}.brand b{color:var(--cyan)}
nav{display:flex;gap:20px}nav a{color:var(--muted);text-decoration:none;font-weight:700;font-size:13px}
main{max-width:1320px;margin:30px auto;padding:0 24px}h1,h2,h3{font-family:Manrope,sans-serif;color:var(--navy);margin:4px 0}h1{font-size:30px;letter-spacing:-1px}h2{font-size:18px}h3{font-size:14px}
.muted{color:var(--muted);font-size:13px;line-height:1.55}.eyebrow{font-size:10px;letter-spacing:2px;color:var(--blue);font-weight:800}
.panel,.stat,.auth{background:#fff;border:1px solid var(--line);border-radius:19px;box-shadow:0 12px 34px #173d4b0b}
.panel{padding:22px;margin-bottom:18px}.auth{max-width:430px;margin:55px auto;padding:30px}
label{display:block;font-size:12px;font-weight:700;margin:13px 0 7px}input{width:100%;border:1px solid #dce6ef;border-radius:11px;padding:12px;font:inherit;font-size:13px;background:#fbfdff}
input:focus{outline:3px solid #dbe7ff;border-color:var(--blue)}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{display:inline-flex;align-items:center;justify-content:center;border:0;border-radius:11px;background:linear-gradient(120deg,var(--blue),#765de8);color:white;padding:12px 17px;font-weight:800;font-size:13px;cursor:pointer;margin-top:14px;box-shadow:0 7px 16px #4776f52b}
.light{background:#edf2ff;color:#365bc4;box-shadow:none}.full{width:100%}
.welcome{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:22px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:20px}.stat{padding:19px;position:relative;overflow:hidden}.stat:after{content:"";position:absolute;width:82px;height:82px;border-radius:50%;right:-23px;top:-25px;background:var(--tint,#edf2ff)}
.stat span{font-size:12px;color:var(--muted);font-weight:700}.stat strong{display:block;font:800 27px Manrope;margin-top:8px;color:var(--navy);position:relative;z-index:1}.stat small{display:block;margin-top:5px;color:var(--muted);font-size:10px}
.tabs{display:flex;gap:8px;overflow-x:auto;padding:7px;background:#e9eef7;border-radius:15px;margin:0 0 22px;scrollbar-width:thin}
.tab-btn{border:0;background:transparent;color:#62748b;padding:12px 16px;border-radius:11px;font:700 12px 'DM Sans';white-space:nowrap;cursor:pointer;transition:.2s}
.tab-btn.active{background:#fff;color:#315de0;box-shadow:0 4px 14px #233f6814}
.tab-panel{display:none}.tab-panel.active{display:block;animation:rise .25s ease}@keyframes rise{from{opacity:.45;transform:translateY(5px)}to{opacity:1;transform:none}}
.columns{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}.bottom{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}
.sampling-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.sampling-point{padding:15px;border:1px solid var(--line);border-radius:15px;background:#fbfcff}.sampling-point h3{font-size:14px;color:var(--blue)}.search-line{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:end}.search-line .btn{margin-top:0;padding:11px 12px;white-space:nowrap}.sampling-point .two{gap:8px}.sampling-point input[readonly]{background:#f0f4fa;color:#52657a}
#map{height:440px;border-radius:14px;z-index:1}.flash{padding:12px 15px;background:#fff3d9;color:#805b12;border-radius:11px;font-size:13px;margin-bottom:13px;border:1px solid #f8dfaa}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:left;padding:13px 11px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:#72839a;font-size:10px;text-transform:uppercase;letter-spacing:.8px}.tablewrap{overflow:auto}
.reply{padding:16px;background:linear-gradient(135deg,#eef3ff,#f6f0ff);border:1px solid #e3e8ff;border-radius:13px;min-height:70px;font-size:13px;line-height:1.6}
.note{font-size:11px;line-height:1.6;color:#64788a;background:#edf3f8;padding:14px;border-radius:12px;margin:12px 0 25px}
.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:15px}.pill{display:inline-flex;padding:6px 10px;border-radius:99px;background:#eef2ff;color:#5268c9;font-size:10px;font-weight:800}
.metric-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}.metric{padding:17px;border:1px solid var(--line);border-radius:15px;background:linear-gradient(145deg,#fff,#f8faff)}.metric b{display:block;font:800 23px Manrope;margin-top:7px}.metric span{font-size:11px;color:var(--muted);font-weight:700}
.legend-row{display:flex;flex-wrap:wrap;gap:14px;font-size:11px;color:var(--muted);margin:12px 0}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
footer{text-align:center;color:#93a3b5;font-size:11px;padding:24px}
@media(max-width:900px){.columns,.bottom{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
@media(max-width:560px){.two,.metric-grid,.sampling-grid{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}h1{font-size:24px}main{padding:0 13px}.panel{padding:16px}.welcome{align-items:flex-start;flex-direction:column}header{padding:0 14px}}

/* Microplastic Pollution Tracking in Urban Water Bodies System vivid refresh */
:root{--ink:#102c43;--muted:#71869a;--line:#e7edf5;--bg:#f3f6fc;--navy:#102a43;--blue:#4169f5;--cyan:#08b8c8;--purple:#8056e8;--pink:#ed68a5;--green:#16ad8b;--amber:#f2a43a}
body{background:linear-gradient(135deg,#f1f5ff 0%,#f7f4ff 48%,#effbfb 100%);min-height:100vh}
header{background:rgba(13,31,59,.96);border:0;color:#fff;box-shadow:0 8px 30px #142d4a20}
.brand{color:#fff}.brand b{color:#35e0d1;font-size:20px}
header nav a{color:#dce7ff;padding:10px 13px;border-radius:10px}header nav a:hover{background:#ffffff18;color:#fff}
main{max-width:1380px;margin:28px auto 40px}
h1{font-size:clamp(27px,3vw,38px);letter-spacing:-1.3px}
.panel,.stat,.auth{border:1px solid #e6ebf5;box-shadow:0 14px 38px #233f6810}
.panel{border-radius:22px;padding:25px}
.welcome{padding:28px 30px;border-radius:24px;color:#fff;background:linear-gradient(115deg,#173b79 0%,#4268e9 53%,#8459dc 100%);box-shadow:0 18px 38px #4565c62e;position:relative;overflow:hidden}
.welcome:after{content:"";position:absolute;width:250px;height:250px;border:1px solid #ffffff35;border-radius:50%;right:8%;top:-145px;box-shadow:0 0 0 28px #ffffff0c,0 0 0 58px #ffffff09}
.welcome h1,.welcome .muted,.welcome .eyebrow{color:#fff;position:relative;z-index:1}.welcome .muted{opacity:.82}.welcome .pill{background:#ffffff24;color:#fff;border:1px solid #ffffff3b;position:relative;z-index:1}
.stats{gap:17px}.stat{border:0;border-radius:19px;padding:22px;background:#fff;box-shadow:0 13px 30px #243f6410;transition:transform .2s,box-shadow .2s}
.stat:hover{transform:translateY(-4px);box-shadow:0 18px 36px #243f641c}
.stat:after{width:105px;height:105px;right:-28px;top:-32px;opacity:.8}
.stat:nth-child(1){border-top:4px solid #5276f6}.stat:nth-child(2){border-top:4px solid #16b9c6}.stat:nth-child(3){border-top:4px solid #9569e9}.stat:nth-child(4){border-top:4px solid #f0a344}
.stat strong{font-size:31px}
.tabs{background:#172e50;padding:8px;border-radius:17px;gap:7px;box-shadow:0 12px 28px #172e501c}
.tab-btn{color:#d7e2f5;padding:13px 17px;border-radius:12px}
.tab-btn:hover{background:#ffffff16;color:#fff}
.tab-btn.active{background:linear-gradient(120deg,#fff,#eef3ff);color:#3458d6;box-shadow:0 5px 16px #081b3b35}
.tab-panel.active{animation:tabIn .3s ease}
@keyframes tabIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.eyebrow{letter-spacing:1.8px}
.panel h2{font-size:20px}
.metric{border:0;background:linear-gradient(145deg,#f0f4ff,#fff);box-shadow:inset 0 0 0 1px #e4eafb}
.metric:nth-child(2){background:linear-gradient(145deg,#eafafa,#fff)}
.metric:nth-child(3){background:linear-gradient(145deg,#f5edff,#fff)}
.btn{background:linear-gradient(110deg,#4169f5,#8656e8);border-radius:12px;padding:13px 19px;transition:transform .18s,filter .18s}
.btn:hover{transform:translateY(-2px);filter:saturate(1.2)}
.btn.light{background:linear-gradient(110deg,#e9efff,#f2eaff);color:#405bd0;border:1px solid #dfe5ff}
input{border:1px solid #dce5f1;background:#f9fbff;border-radius:12px;padding:13px 14px}
input:focus{outline:3px solid #dce5ff;border-color:#6a83f5}
#map{border:1px solid #dce6f3;box-shadow:0 8px 22px #163a5b12}
.note{border:1px solid #e0e9f6;background:linear-gradient(110deg,#edf3ff,#f3efff)}
/* Bright, editorial login and registration pages */
.auth-shell{max-width:1110px;margin:34px auto 45px;min-height:610px;display:grid;grid-template-columns:1.05fr .95fr;border-radius:30px;overflow:hidden;background:#fff;box-shadow:0 28px 80px #1a356b25;border:1px solid #ffffff}
.auth-art{position:relative;overflow:hidden;padding:54px 48px;color:#fff;background:linear-gradient(145deg,#102c56 0%,#2756b9 48%,#8058db 100%);display:flex;flex-direction:column;justify-content:space-between}
.auth-art:before,.auth-art:after{content:"";position:absolute;border-radius:50%;border:1px solid #ffffff35}
.auth-art:before{width:420px;height:420px;right:-160px;top:-110px;box-shadow:0 0 0 35px #ffffff0d,0 0 0 75px #ffffff09}
.auth-art:after{width:260px;height:260px;left:-130px;bottom:-110px;background:#16c8c322;box-shadow:0 0 0 25px #ffffff0a}
.auth-art>*{position:relative;z-index:1}
.auth-logo{font:800 16px Manrope;letter-spacing:0;line-height:1.35;max-width:100%}.auth-logo span{color:#56f0d7}
.auth-art h1{font-size:clamp(34px,4vw,48px);line-height:1.1;color:#fff;max-width:440px;margin:35px 0 14px}
.auth-art p{color:#e0e9ff;line-height:1.8;max-width:410px;font-size:14px}
.auth-feature{display:flex;align-items:center;gap:12px;padding:14px 15px;border:1px solid #ffffff35;background:#ffffff14;border-radius:15px;margin-top:12px;backdrop-filter:blur(10px);font-size:12px;font-weight:700}
.auth-feature i{font-style:normal;width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:#ffffff24;font-size:17px}
.auth-form-side{padding:52px 55px;display:flex;align-items:center;justify-content:center;background:radial-gradient(circle at 100% 0%,#eeeaff,transparent 35%),#fff}
.auth-card{width:100%;max-width:390px}.auth-card .eyebrow{color:#6d63df}.auth-card h2{font:800 30px Manrope;color:#142d4a;margin:9px 0}
.auth-card .muted{font-size:13px;margin-bottom:24px}
.auth-card label{font-size:12px;color:#334d68;margin-top:17px}
.auth-card input{height:49px;background:#f7f9ff;border:1px solid #e0e7f2}
.auth-card .btn{height:49px;font-size:14px;margin-top:23px}
.auth-switch{margin-top:23px;text-align:center;font-size:13px;color:#718398}
.auth-switch a{color:#5d58db;font-weight:800;text-decoration:none}
.auth-shield{display:inline-flex;gap:7px;align-items:center;font-size:11px;color:#8b9aab;margin-top:18px}
@media(max-width:760px){.auth-shell{grid-template-columns:1fr;margin:18px 0}.auth-art{padding:30px;min-height:330px}.auth-art h1{font-size:34px;margin-top:22px}.auth-form-side{padding:34px 24px}.welcome{padding:23px}.tabs{border-radius:14px}.tab-btn{padding:12px 13px}}
</style></head><body>
<header><a class="brand" href="/"><b>≈</b> Microplastic Pollution Tracking in Urban Water Bodies System</a>{% if session.get('user_id') %}<nav><a href="/dashboard">Dashboard</a><a href="/logout">Logout</a></nav>{% endif %}</header>
<main>
{% with msgs=get_flashed_messages() %}{% if msgs %}<div class="flash">{% for msg in msgs %}<div>{{ msg }}</div>{% endfor %}</div>{% endif %}{% endwith %}
{{ body|safe }}
</main><footer>Microplastic Pollution Tracking in Urban Water Bodies System • Academic prototype</footer>
</body></html>
"""

LOGIN_BODY = r"""<section class="auth-shell">
 <div class="auth-art">
  <div><div class="auth-logo"><span>≈</span> Microplastic Pollution Tracking in Urban Water Bodies System</div>
   <div class="eyebrow" style="color:#9cece8;margin-top:58px">SMART WATER • HEALTHIER FUTURE</div>
   <h1>Every drop tells a story.</h1>
   <p>Monitor urban water bodies, record sampling locations and explore microplastic image-analysis results in one place.</p>
  </div>
  <div>
   <div class="auth-feature"><i>⌖</i><span>GPS-based sampling location records</span></div>
   <div class="auth-feature"><i>◉</i><span>Concentration insights and sample history</span></div>
   <div class="auth-feature"><i>✧</i><span>One workspace for water monitoring</span></div>
  </div>
 </div>
 <div class="auth-form-side"><div class="auth-card">
  <div class="eyebrow">YOUR MONITORING SPACE</div><h2>Welcome back</h2>
  <p class="muted">Sign in to continue to your Microplastic Pollution Tracking in Urban Water Bodies System dashboard.</p>
  <form method="post">
   <label>Email address</label><input type="email" name="email" placeholder="you@example.com" autocomplete="email" required>
   <label>Password</label><input type="password" name="password" placeholder="Enter your password" autocomplete="current-password" required>
   <button class="btn full">Sign in to Microplastic Pollution Tracking in Urban Water Bodies System →</button>
  </form>
  <div class="auth-switch">Don't have an account? <a href="/signup">Create account</a></div>
  <div class="auth-shield">✦ Secure sign-in • Microplastic Pollution Tracking in Urban Water Bodies System academic project</div>
 </div></div>
</section>"""
SIGNUP_BODY = r"""<section class="auth-shell">
 <div class="auth-art">
  <div><div class="auth-logo"><span>≈</span> Microplastic Pollution Tracking in Urban Water Bodies System</div>
   <div class="eyebrow" style="color:#9cece8;margin-top:58px">START MONITORING WITH PURPOSE</div>
   <h1>Make water data meaningful.</h1>
   <p>Create your account to manage microscope samples, map collection points and review water-monitoring observations.</p>
  </div>
  <div>
   <div class="auth-feature"><i>＋</i><span>Save new water sample observations</span></div>
   <div class="auth-feature"><i>⌖</i><span>Keep coordinates with each sample</span></div>
   <div class="auth-feature"><i>▤</i><span>Build your own monitoring history</span></div>
  </div>
 </div>
 <div class="auth-form-side"><div class="auth-card">
  <div class="eyebrow">JOIN MICROPLASTIC POLLUTION TRACKING IN URBAN WATER BODIES SYSTEM</div><h2>Create your account</h2>
  <p class="muted">Set up your workspace in a few simple steps.</p>
  <form method="post">
   <label>Your name</label><input name="name" placeholder="Enter your name" autocomplete="name" required>
   <label>Email address</label><input type="email" name="email" placeholder="you@example.com" autocomplete="email" required>
   <label>Password <span style="font-weight:500;color:#8999aa">(minimum 6 characters)</span></label><input type="password" name="password" placeholder="Create a password" minlength="6" autocomplete="new-password" required>
   <button class="btn full">Create Microplastic Pollution Tracking in Urban Water Bodies System account →</button>
  </form>
  <div class="auth-switch">Already have an account? <a href="/login">Sign in</a></div>
  <div class="auth-shield">✦ Your sample workspace starts here</div>
 </div></div>
</section>"""

@app.route("/")
def home():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name=request.form.get("name","").strip(); email=request.form.get("email","").strip().lower(); password=request.form.get("password","")
        if not name or not email or len(password)<6:
            flash("Enter name, valid email and password with at least 6 characters.")
        else:
            try:
                with connect_db() as con:
                    cur=con.execute("INSERT INTO users(name,email,password_hash) VALUES(?,?,?)",(name,email,generate_password_hash(password)))
                    session["user_id"]=cur.lastrowid; session["user_name"]=name
                return redirect(url_for("dashboard"))
            except sqlite3.IntegrityError:
                flash("This email is already registered.")
    return render_template_string(BASE_HTML, title="Sign up", body=SIGNUP_BODY)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method=="POST":
        email=request.form.get("email","").strip().lower(); password=request.form.get("password","")
        with connect_db() as con: user=con.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
        if user and check_password_hash(user["password_hash"],password):
            session["user_id"]=user["id"]; session["user_name"]=user["name"]
            return redirect(url_for("dashboard"))
        flash("Incorrect email or password.")
    return render_template_string(BASE_HTML, title="Login", body=LOGIN_BODY)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

DASHBOARD_BODY = r"""
<div class="welcome">
 <div><div class="eyebrow">MICROPLASTIC POLLUTION TRACKING IN URBAN WATER BODIES SYSTEM • WATER INTELLIGENCE</div><h1>Environmental monitoring hub</h1>
 <p class="muted">Welcome, {{ name }}. Explore your water-sample data across dedicated workspaces.</p></div>
 <span class="pill">● SYSTEM ONLINE</span>
</div>
<div class="stats">
 <div class="stat" style="--tint:#e5edff"><span>Total samples</span><strong id="samples">0</strong><small>Saved observations</small></div>
 <div class="stat" style="--tint:#e2fbfa"><span>Particle candidates</span><strong id="particles">0</strong><small>Image-based estimates</small></div>
 <div class="stat" style="--tint:#f0eaff"><span>Locations monitored</span><strong id="locations">0</strong><small>Distinct sampling points</small></div>
 <div class="stat" style="--tint:#fff0e1"><span>Average concentration</span><strong id="average">0</strong><small>Candidate particles / L</small></div>
</div>

<div class="tabs" role="tablist" aria-label="Dashboard sections">
 <button class="tab-btn active" data-tab="samples-tab">＋ Add sample</button>
 <button class="tab-btn" data-tab="overview">✦ Overview</button>
 <button class="tab-btn" data-tab="concentration">◉ Concentration</button>
 <button class="tab-btn" data-tab="locations">⌖ Location tracking</button>
 <button class="tab-btn" data-tab="history">▤ Sample history</button>
 <button class="tab-btn" data-tab="assistant">◖ Voice assistant</button>
</div>

<section class="tab-panel" id="tab-overview">
 <div class="columns">
  <section class="panel"><div class="section-head"><div><div class="eyebrow">DATA SNAPSHOT</div><h2>Concentration trend</h2></div><span class="pill">Recent samples</span></div>
   <canvas id="overviewChart" height="145"></canvas>
  </section>
  <section class="panel"><div class="section-head"><div><div class="eyebrow">QUICK INSIGHT</div><h2>Monitoring summary</h2></div><span class="pill">Live data</span></div>
   <div class="metric-grid" style="grid-template-columns:1fr 1fr">
    <div class="metric"><span>Highest candidate/L</span><b id="highestValue">—</b><small id="highestPlace" class="muted">No sample yet</small></div>
    <div class="metric"><span>Latest sample</span><b id="latestValue">—</b><small id="latestPlace" class="muted">No sample yet</small></div>
   </div>
   <p class="muted" style="margin-top:16px">Use the tabs above to explore concentration measurements, map sampling locations, add a new sample or review past records.</p>
  </section>
 </div>
 <section class="panel"><div class="section-head"><div><div class="eyebrow">RECENT ACTIVITY</div><h2>Latest sample records</h2></div><button class="btn light" onclick="showTab('history')">View all history →</button></div>
  <div class="tablewrap"><table><thead><tr><th>Date</th><th>Location</th><th>Candidate count</th><th>Candidate/L</th><th>Classifier</th></tr></thead><tbody id="overviewRows"><tr><td colspan="5">Loading sample data…</td></tr></tbody></table></div>
 </section>
</section>

<section class="tab-panel" id="tab-concentration">
 <section class="panel"><div class="section-head"><div><div class="eyebrow">POLLUTION MEASUREMENTS</div><h2>Concentration analytics</h2><p class="muted">Estimated image-based particle candidates divided by recorded sample volume.</p></div><span class="pill">Candidates / litre</span></div>
  <div class="metric-grid">
   <div class="metric"><span>Average concentration</span><b id="concAverage">0</b></div>
   <div class="metric"><span>Maximum concentration</span><b id="concMaximum">0</b></div>
   <div class="metric"><span>Minimum concentration</span><b id="concMinimum">0</b></div>
  </div>
 </section>
 <div class="columns">
  <section class="panel"><h2>Concentration by sample</h2><p class="muted">Compare estimated candidate counts per litre.</p><canvas id="concentrationChart" height="210"></canvas></section>
  <section class="panel"><h2>Classifier results</h2><p class="muted">Distribution of model labels, when a compatible classifier is available.</p><canvas id="classChart" height="210"></canvas><div class="legend-row"><span><i class="dot" style="background:#e85e75"></i>High</span><span><i class="dot" style="background:#efa83d"></i>Moderate</span><span><i class="dot" style="background:#16a887"></i>Low</span><span><i class="dot" style="background:#aab5c4"></i>Unavailable</span></div></section>
 </div>
</section>

<section class="tab-panel" id="tab-locations">
 <section class="panel"><div class="section-head"><div><div class="eyebrow">GEOSPATIAL MONITORING</div><h2>Location tracking map</h2><p class="muted">Map markers show the GPS coordinates saved with each sample.</p></div><span class="pill">OpenStreetMap</span></div>
  <div id="map"></div>
 </section>
 <section class="panel"><h2>Sampling locations</h2><p class="muted">Each saved sample contains four separate sampling points.</p><div class="tablewrap"><table><thead><tr><th>Sample</th><th>Point</th><th>Location</th><th>Latitude</th><th>Longitude</th><th>Candidate/L</th><th>Class</th></tr></thead><tbody id="locationRows"><tr><td colspan="7">Loading locations…</td></tr></tbody></table></div></section>
</section>

<section class="tab-panel active" id="tab-samples-tab">
 <div class="columns">
  <section class="panel"><div class="eyebrow">01 / NEW OBSERVATION</div><h2>Start with a new water sample</h2><p class="muted">Upload a microscope image and enter the collection details to analyse and save your sample.</p>
   <form method="post" action="/analyze" enctype="multipart/form-data">
    <label>Microscope image</label><input type="file" name="image" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" required>
    <p class="muted">Enter four sampling points. Use each Search map button to find a place and capture its coordinates.</p>
    <div class="sampling-grid">
     {% for n in range(1,5) %}
     <div class="sampling-point">
      <h3>Sampling Location {{ n }}</h3>
      <label for="location{{ n }}">Location {{ n }} name</label>
      <div class="search-line">
       <input id="location{{ n }}" name="location{{ n }}" placeholder="Search a lake, river or area" required>
       <button type="button" class="btn light search-map-btn" data-point="{{ n }}">Search map</button>
      </div>
      <div class="two">
       <div><label for="latitude{{ n }}">Latitude</label><input id="latitude{{ n }}" name="latitude{{ n }}" type="number" step="any" placeholder="Search location first" readonly required></div>
       <div><label for="longitude{{ n }}">Longitude</label><input id="longitude{{ n }}" name="longitude{{ n }}" type="number" step="any" placeholder="Search location first" readonly required></div>
      </div>
      <p class="muted" id="status{{ n }}">No location selected yet.</p>
     </div>
     {% endfor %}
    </div>
    <label>Sample volume (litres)</label><input name="volume_liters" type="number" min="0.001" step="0.001" value="1" required>
    <button type="button" class="btn light" id="gps">⌖ Use my current GPS location for Location 1</button>
    <button class="btn full">✦ Analyse and save one sample with all 4 locations</button>
   </form><p class="muted">Search each location to fill its coordinates. GPS can be used for Location 1.</p>
  </section>
  <section class="panel"><div class="eyebrow">HOW IT WORKS</div><h2>Sample processing</h2>
   <div class="metric" style="margin-top:14px"><span>01 • Upload</span><p class="muted">Choose a clear microscope image of the filtered water sample.</p></div>
   <div class="metric" style="margin-top:10px"><span>02 • Record</span><p class="muted">Select four sampling sites and enter the water volume.</p></div>
   <div class="metric" style="margin-top:10px"><span>03 • Analyse</span><p class="muted">The prototype estimates image particle candidates and calculates candidates per litre.</p></div>
  </section>
 </div>
</section>

<section class="tab-panel" id="tab-history">
 <section class="panel"><div class="section-head"><div><div class="eyebrow">YOUR OBSERVATIONS</div><h2>Complete sample history</h2></div><span class="pill" id="historyCount">0 records</span></div>
  <div class="tablewrap"><table><thead><tr><th>Date</th><th>Location</th><th>Volume</th><th>Candidate count</th><th>Candidate/L</th><th>Classifier</th><th>Confidence</th></tr></thead><tbody id="historyRows"><tr><td colspan="7">Loading sample data…</td></tr></tbody></table></div>
 </section>
</section>

<section class="tab-panel" id="tab-assistant">
 <section class="panel"><div class="eyebrow">MICROPLASTIC POLLUTION TRACKING IN URBAN WATER BODIES SYSTEM HELPER</div><h2>Voice assistant</h2><p class="muted">Ask about your saved samples, total candidate count or the highest and lowest estimated concentration.</p>
  <div class="reply" id="reply">Your assistant is ready. Press the microphone button to ask a question.</div>
  <button class="btn" id="voiceBtn">🎙 Start voice question</button>
  <button class="btn light" id="read">Read dashboard summary aloud</button>
 </section>
</section>

<div class="note">Scientific limitation: the current OpenCV counter reports image-based particle candidates, not confirmed microplastics. It may count dust, bubbles, scratches and debris. Concentration is candidate count divided by the sample volume entered. Classifier labels depend on a compatible trained model. Laboratory validation is required for scientific claims.</div>

<script>
let records=[], charts={}, mapInstance=null;
function showTab(name){
 document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));
 document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('active',p.id==='tab-'+name));
 if(name==='locations')setTimeout(()=>{if(mapInstance)mapInstance.invalidateSize()},120);
 if(name==='concentration')setTimeout(renderConcentrationCharts,80);
}
document.querySelectorAll('.tab-btn').forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
document.querySelectorAll('.search-map-btn').forEach(b=>b.addEventListener('click',()=>searchSamplingLocation(Number(b.dataset.point))));
function color(level){return level==='High'?'#e85e75':level==='Moderate'?'#efa83d':level==='Low'?'#16a887':'#aab5c4'}
function fmt(v){return Number(v||0).toLocaleString(undefined,{maximumFractionDigits:2})}
function fillRows(){
 const recent=records.slice(0,8);
 const rowHtml=arr=>arr.map(r=>`<tr><td>${r.created_at||'—'}</td><td>${r.location}</td><td>${r.particle_candidates}</td><td>${fmt(r.concentration)}</td><td>${r.predicted_class||'Unavailable'}</td></tr>`).join('');
 document.getElementById('overviewRows').innerHTML=recent.length?rowHtml(recent):'<tr><td colspan="5">No samples saved yet.</td></tr>';
 document.getElementById('historyRows').innerHTML=records.length?records.map(r=>`<tr><td>${r.created_at||'—'}</td><td>${r.location}</td><td>${r.volume_liters??'—'}</td><td>${r.particle_candidates}</td><td>${fmt(r.concentration)}</td><td>${r.predicted_class||'Unavailable'}</td><td>${r.confidence==null?'—':fmt(r.confidence)+'%'}</td></tr>`).join(''):'<tr><td colspan="7">No samples saved yet.</td></tr>';
 document.getElementById('locationRows').innerHTML=records.length?records.map((r,i)=>{
  return [1,2,3,4].map(n=>`<tr><td>${i+1}</td><td>Location ${n}</td><td>${r['location'+n]||'—'}</td><td>${r['latitude'+n]??'—'}</td><td>${r['longitude'+n]??'—'}</td><td>${fmt(r.concentration)}</td><td>${r.predicted_class||'Unavailable'}</td></tr>`).join('');
 }).join(''):'<tr><td colspan="7">No locations saved yet.</td></tr>';
 document.getElementById('historyCount').textContent=records.length+' records';
}
let searchMarkers=[];
function drawMap(){
 if(!document.getElementById('map')||mapInstance)return;
 mapInstance=L.map('map').setView([12.9716,77.5946],11);
 L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'&copy; OpenStreetMap'}).addTo(mapInstance);
 const allPoints=[];
 records.forEach((r,i)=>{
  [1,2,3,4].forEach(n=>{
   if(r['latitude'+n]==null||r['longitude'+n]==null)return;
   const lat=Number(r['latitude'+n]),lon=Number(r['longitude'+n]);
   if(!Number.isFinite(lat)||!Number.isFinite(lon))return;
   const place=r['location'+n]||('Location '+n),c=color(r.predicted_class);
   allPoints.push([lat,lon]);
   L.circleMarker([lat,lon],{radius:8,color:c,fillColor:c,fillOpacity:.85,weight:2})
    .addTo(mapInstance).bindPopup(`<b>Sample ${i+1} • Location ${n}</b><br>${place}<br>Candidate/L: ${r.concentration}<br>Class: ${r.predicted_class||'Unavailable'}`);
  });
 });
 if(allPoints.length)mapInstance.fitBounds(L.latLngBounds(allPoints).pad(.2),{maxZoom:14});
}
async function searchSamplingLocation(point){
 const nameInput=document.getElementById('location'+point);
 const status=document.getElementById('status'+point);
 const query=nameInput.value.trim();
 if(!query){status.textContent='Please enter a place name first.';return}
 status.textContent='Searching for location…';
 try{
  const url='https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&countrycodes=in&q='+encodeURIComponent(query);
  const response=await fetch(url,{headers:{'Accept':'application/json'}});
  if(!response.ok)throw new Error('Location search service is unavailable.');
  const results=await response.json();
  if(!results.length){status.textContent='Location not found. Try a more specific name.';return}
  const place=results[0],lat=Number(place.lat),lon=Number(place.lon);
  document.getElementById('latitude'+point).value=lat.toFixed(6);
  document.getElementById('longitude'+point).value=lon.toFixed(6);
  status.textContent='Selected: '+place.display_name;
  showTab('locations');
  setTimeout(()=>{
   if(!mapInstance)drawMap();
   if(mapInstance){
    mapInstance.invalidateSize();
    mapInstance.setView([lat,lon],14);
    searchMarkers.forEach(m=>mapInstance.removeLayer(m));
    searchMarkers=[L.marker([lat,lon]).addTo(mapInstance).bindPopup(`<b>Location ${point}</b><br>${nameInput.value}`).openPopup()];
   }
  },140);
 }catch(error){status.textContent='Search failed. Check your internet connection and try again.'}
}
function renderConcentrationCharts(){
 if(!records.length)return;
 const labels=records.slice(0,15).reverse().map((r,i)=>r.location+' #'+(i+1));
 const vals=records.slice(0,15).reverse().map(r=>r.concentration);
 const configs=[
  ['concentrationChart','bar',{labels,datasets:[{label:'Candidate/L',data:vals,backgroundColor:['#5c7cfa','#20b8c5','#9a75ec','#f0a64a','#e978aa'],borderRadius:7}]},{plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}],
  ['classChart','doughnut',{labels:['High','Moderate','Low','Unavailable'],datasets:[{data:['High','Moderate','Low','Unavailable'].map(k=>records.filter(r=>(r.predicted_class||'Unavailable')===k).length),backgroundColor:['#e85e75','#efa83d','#16a887','#aab5c4'],borderWidth:0}]},{plugins:{legend:{position:'bottom'}}}]
 ];
 configs.forEach(([id,type,data,options])=>{const el=document.getElementById(id);if(!el)return;if(charts[id])charts[id].destroy();charts[id]=new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:true,...options}})});
}
function summary(){
 if(!records.length)return 'No sample records are available. Upload your first microscope image.';
 let high=[...records].sort((a,b)=>b.concentration-a.concentration)[0];
 return `There are ${records.length} samples. The highest estimated particle candidate concentration is ${high.concentration} candidates per litre at ${high.location}. These are image estimates, not confirmed microplastic measurements.`;
}
function say(t){if(!('speechSynthesis'in window))return;speechSynthesis.cancel();let u=new SpeechSynthesisUtterance(t);u.lang='en-IN';speechSynthesis.speak(u)}
document.getElementById('read').onclick=()=>say(summary());
document.getElementById('gps').onclick=()=>{if(!navigator.geolocation)return alert('GPS not supported');navigator.geolocation.getCurrentPosition(p=>{document.getElementById('latitude1').value=p.coords.latitude.toFixed(6);document.getElementById('longitude1').value=p.coords.longitude.toFixed(6);document.getElementById('status1').textContent='Current GPS location selected.';if(!document.getElementById('location1').value)document.getElementById('location1').value='Current GPS location'},()=>alert('Allow location access or search for a place manually'))};
document.getElementById('voiceBtn').onclick=()=>{
 const R=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!R){document.getElementById('reply').textContent='Voice input is not supported in this browser. Try Chrome.';return}
 let r=new R();r.lang='en-IN';document.getElementById('reply').textContent='Listening…';r.onresult=e=>{
 let q=e.results[0][0].transcript.toLowerCase(),ans=summary();
 if(q.includes('how many')||q.includes('count'))ans=`There are ${records.length} saved samples and ${records.reduce((n,x)=>n+x.particle_candidates,0)} image-based particle candidates in total.`;
 if(q.includes('cleanest')||q.includes('lowest')){let x=[...records].sort((a,b)=>a.concentration-b.concentration)[0];ans=x?`${x.location} has the lowest estimated concentration, ${x.concentration} candidates per litre.`:'No records yet.'}
 document.getElementById('reply').textContent='You asked: '+e.results[0][0].transcript+'. '+ans;say(ans)};
 r.onerror=()=>document.getElementById('reply').textContent='I could not hear clearly. Please try again.';r.start()
};
async function init(){
 const [a,b]=await Promise.all([fetch('/api/samples'),fetch('/api/summary')]);records=await a.json();const s=await b.json();
 document.getElementById('samples').textContent=s.total_samples;
 document.getElementById('particles').textContent=s.total_candidates;
 document.getElementById('locations').textContent=s.locations;
 document.getElementById('average').textContent=fmt(s.avg_concentration);
 document.getElementById('concAverage').textContent=fmt(s.avg_concentration);
 document.getElementById('concMaximum').textContent=fmt(records.length?Math.max(...records.map(r=>r.concentration)):0);
 document.getElementById('concMinimum').textContent=fmt(records.length?Math.min(...records.map(r=>r.concentration)):0);
 if(records.length){let hi=[...records].sort((a,b)=>b.concentration-a.concentration)[0],latest=records[0];document.getElementById('highestValue').textContent=fmt(hi.concentration);document.getElementById('highestPlace').textContent=hi.location;document.getElementById('latestValue').textContent=fmt(latest.concentration);document.getElementById('latestPlace').textContent=latest.location}
 fillRows();drawMap();
 if(records.length){const recent=records.slice(0,12).reverse();charts.overview=new Chart(document.getElementById('overviewChart'),{type:'line',data:{labels:recent.map((x,i)=>'Sample '+(i+1)),datasets:[{label:'Candidate/L',data:recent.map(x=>x.concentration),borderColor:'#4776f5',backgroundColor:'rgba(71,118,245,.13)',fill:true,tension:.35,pointRadius:4}]},options:{responsive:true,plugins:{legend:{display:false}},scales:{y:{beginAtZero:true}}}})}
 renderConcentrationCharts();
}
init().catch(err=>console.error('Dashboard data error:',err));
</script>
"""

@app.route("/dashboard")
@require_login
def dashboard():
    with connect_db() as con:
        samples=con.execute("SELECT * FROM samples WHERE user_id=? ORDER BY id DESC LIMIT 100",(session["user_id"],)).fetchall()
    body=render_template_string(DASHBOARD_BODY,name=session.get("user_name","User"),samples=samples)
    return render_template_string(BASE_HTML,title="Dashboard",body=body)

@app.route("/analyze",methods=["POST"])
@require_login
def analyze():
    f=request.files.get("image")
    allowed={"png","jpg","jpeg","bmp","tif","tiff"}
    if not f or not f.filename or "." not in f.filename or f.filename.rsplit(".",1)[1].lower() not in allowed:
        flash("Choose a valid image file.")
        return redirect(url_for("dashboard"))
    try:
        volume=float(request.form["volume_liters"])
        points=[]
        for n in range(1,5):
            place=request.form.get(f"location{n}","").strip()
            lat=float(request.form.get(f"latitude{n}",""))
            lon=float(request.form.get(f"longitude{n}",""))
            if not place or not (-90<=lat<=90) or not (-180<=lon<=180):
                raise ValueError
            points.append((place,lat,lon))
        if volume<=0:
            raise ValueError
    except (ValueError,KeyError):
        flash("Enter and search all four sampling locations, and check the sample volume.")
        return redirect(url_for("dashboard"))

    # Keep a combined legacy location and first-point coordinates for compatibility.
    combined_location=" | ".join(point[0] for point in points)
    filename=uuid.uuid4().hex+"_"+secure_filename(f.filename)
    path=os.path.join(UPLOAD_DIR,filename); f.save(path)
    count=count_image_candidates(path)
    label,confidence=predict_class(path)
    concentration=round(count/volume,2)
    with connect_db() as con:
        con.execute("""INSERT INTO samples(
            user_id,location,latitude,longitude,volume_liters,particle_candidates,
            concentration,predicted_class,confidence,image_name,created_at,
            location1,latitude1,longitude1,location2,latitude2,longitude2,
            location3,latitude3,longitude3,location4,latitude4,longitude4
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (session["user_id"],combined_location,points[0][1],points[0][2],volume,count,
         concentration,label,confidence,filename,datetime.now().isoformat(timespec="seconds"),
         points[0][0],points[0][1],points[0][2],
         points[1][0],points[1][1],points[1][2],
         points[2][0],points[2][1],points[2][2],
         points[3][0],points[3][1],points[3][2]))
    flash("One sample record with all four sampling locations was saved.")
    if label is None: flash("Classifier unavailable. Keep your pollution_classifier_final.keras file beside app.py and check model compatibility.")
    return redirect(url_for("dashboard"))

@app.route("/api/samples")
@require_login
def api_samples():
    with connect_db() as con:
        rows=con.execute("""SELECT location,latitude,longitude,
        location1,latitude1,longitude1,location2,latitude2,longitude2,
        location3,latitude3,longitude3,location4,latitude4,longitude4,
        volume_liters,particle_candidates,concentration,predicted_class,confidence,created_at
        FROM samples WHERE user_id=? ORDER BY id DESC""",(session["user_id"],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/api/summary")
@require_login
def api_summary():
    with connect_db() as con:
        r=con.execute("""SELECT COUNT(*) total_samples,COALESCE(SUM(particle_candidates),0) total_candidates,
        (SELECT COUNT(DISTINCT place) FROM (
            SELECT location1 AS place FROM samples WHERE user_id=? UNION
            SELECT location2 AS place FROM samples WHERE user_id=? UNION
            SELECT location3 AS place FROM samples WHERE user_id=? UNION
            SELECT location4 AS place FROM samples WHERE user_id=?
        ) WHERE place IS NOT NULL AND place<>'' ) locations,
        COALESCE(AVG(concentration),0) avg_concentration
        FROM samples WHERE user_id=?""",(session["user_id"],session["user_id"],session["user_id"],session["user_id"],session["user_id"])).fetchone()
    return jsonify(dict(r))

@app.errorhandler(413)
def too_large(_):
    flash("Image too large. Maximum size is 12 MB.")
    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    print("Microplastic Pollution Tracking in Urban Water Bodies System starting...")
    print("Model status:", model_load_message)
    app.run(host="127.0.0.1",port=5000,debug=False,use_reloader=False)
