from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import mysql.connector
import json
from werkzeug.security import generate_password_hash, check_password_hash
from config import API_KEY  # ← Tvoj API ključ iz posebnog fajla

app = Flask(__name__)

# ✅ Učitavanje modela
model = tf.keras.models.load_model("model.h5")
print("✅ Model je učitan.")

# ✅ Konekcija sa MySQL bazom
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Bugarin12345",  # ← Izmeni za deployment
    database="spedicija"
)
cursor = db.cursor(dictionary=True)
print("✅ Povezano sa MySQL bazom.")

# ✅ Provera API ključa
def valid_api_key():
    key = request.headers.get("X-API-Key")
    print("🔑 Primljen API ključ:", key)
    if not key:
        return False
    cursor.execute("SELECT * FROM api_keys WHERE api_key = %s AND is_active = TRUE", (key,))
    return cursor.fetchone() is not None

# ✅ Početna ruta
@app.route("/")
def home():
    return "Flask API je aktivan! 🎯"

# ✅ Predikcija ruta
@app.route("/predict", methods=["POST"])
def predict():
    print("🚀 POZVAN JE /predict")
    
    if not valid_api_key():
        return jsonify({"error": "Nevažeći API ključ"}), 401

    data = request.get_json()
    print("📥 Primljen JSON:", data)

    if "troškovi" not in data or "vremenski_faktori" not in data:
        return jsonify({"error": "Nedostaju troškovi ili vremenski_faktori"}), 400

    try:
        troskovi = data["troškovi"]
        vreme = data["vremenski_faktori"]

        if not isinstance(troskovi, list) or not isinstance(vreme, list):
            return jsonify({"error": "Polja 'troškovi' i 'vremenski_faktori' moraju biti liste brojeva."}), 400

        if not all(isinstance(x, (int, float)) for x in troskovi + vreme):
            return jsonify({"error": "Svi elementi moraju biti brojevi (int ili float)."}), 400

        combined = troskovi + vreme
        if len(combined) != 10:
            return jsonify({"error": f"Očekivano je tačno 10 vrednosti, primljeno {len(combined)}."}), 400

        ulaz = np.array(combined).reshape(1, -1)
        predikcija = model.predict(ulaz)

        rezultat = {
            "ukupni_trošak": float(predikcija[0][0]),
            "vreme_putovanja": float(predikcija[0][1])
        }

        # Logovanje u bazu
        ip = request.remote_addr
        key = request.headers.get("X-API-Key")
        cursor.execute("SELECT firma FROM api_keys WHERE api_key = %s", (key,))
        firma = cursor.fetchone()["firma"]

        cursor.execute(
            "INSERT INTO api_logs (api_key, firma, ulaz_json, rezultat_json, ip_adresa) VALUES (%s, %s, %s, %s, %s)",
            (key, firma, json.dumps(data), json.dumps(rezultat), ip)
        )
        db.commit()

        return jsonify(rezultat), 200

    except Exception as e:
        print("❌ Greška:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ Registracija korisnika
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")
        firma = data.get("firma")

        if not email or not password or not firma:
            return jsonify({"error": "Sva polja su obavezna."}), 400

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            return jsonify({"error": "Korisnik već postoji."}), 400

        hashed_password = generate_password_hash(password)
        cursor.execute(
            "INSERT INTO users (email, password_hash, firma) VALUES (%s, %s, %s)",
            (email, hashed_password, firma)
        )
        db.commit()
        return jsonify({"message": "Uspešno registrovan korisnik."}), 201

    except Exception as e:
        print("❌ Greška u registraciji:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ Pokretanje servera
if __name__ == "__main__":
    print("✅ Flask kreće sada...")
    app.run(debug=True, port=5050)
