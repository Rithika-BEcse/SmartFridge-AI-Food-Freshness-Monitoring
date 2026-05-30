import os
import sys
import subprocess
import importlib
import threading
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Ensure the root project directory is available for imports
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Automatically install missing packages if needed
REQUIRED_PACKAGES = {
    "pyzbar": "pyzbar",
    "streamlit_webrtc": "streamlit-webrtc",
    "plotly": "plotly",
    "pyttsx3": "pyttsx3",
}


def install_package(package_name):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def import_or_install(module_name, package_name=None):
    package_name = package_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        installed = install_package(package_name)
        if installed:
            try:
                return importlib.import_module(module_name)
            except ImportError:
                return None
        return None

# Core imports
streamlit = import_or_install("streamlit")
if streamlit is None:
    raise ImportError("Streamlit is required to run this app. Please install it using pip.")

st = streamlit

PIL = import_or_install("PIL")
if PIL is None:
    raise ImportError("Pillow is required. Please install it using pip.")
from PIL import Image

np = import_or_install("numpy")
if np is None:
    raise ImportError("numpy is required. Please install it using pip.")

cv2 = import_or_install("cv2", "opencv-python")
if cv2 is None:
    raise ImportError("opencv-python is required. Please install it using pip.")

px = None
plotly = import_or_install("plotly")
if plotly is not None:
    px = import_or_install("plotly.express")

pyzbar = import_or_install("pyzbar")
pyzbar_decode = None
if pyzbar is not None:
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode
    except Exception:
        pyzbar_decode = None

PYZBAR_AVAILABLE = pyzbar_decode is not None

streamlit_webrtc = import_or_install("streamlit_webrtc")
if streamlit_webrtc is not None:
    try:
        from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
    except Exception:
        pass

pyttsx3 = import_or_install("pyttsx3")

# Local freshness predictor loader
freshness_model = None
freshness_classes = []


def load_freshness_model():
    global freshness_model, freshness_classes
    if freshness_model is not None and freshness_classes:
        return freshness_model, freshness_classes

    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
    except ImportError:
        freshness_model = None
        freshness_classes = []
        return None, []

    model_path = os.path.join(ROOT_DIR, "src", "models", "freshness_model.h5")
    class_path = os.path.join(ROOT_DIR, "src", "models", "class_indices.json")

    if not os.path.exists(model_path) or not os.path.exists(class_path):
        freshness_model = None
        freshness_classes = []
        return None, []

    try:
        freshness_model = load_model(model_path, compile=False)
    except Exception:
        freshness_model = None

    try:
        import json
        with open(class_path, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        inv = {int(v): k for k, v in class_indices.items()}
        freshness_classes = [inv[i] for i in range(max(inv.keys()) + 1)]
    except Exception:
        freshness_classes = []

    return freshness_model, freshness_classes


def load_latest_report():
    report_path = os.path.join(ROOT_DIR, "outputs", "latest_report.json")
    if not os.path.exists(report_path):
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


UPLOAD_DIR = os.path.join(ROOT_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def save_uploaded_image(uploaded_file, prefix="capture"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}_{uploaded_file.name}"
    safe_path = os.path.join(UPLOAD_DIR, filename)
    with open(safe_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return safe_path


def save_image_from_pil(image, prefix="capture"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    safe_path = os.path.join(UPLOAD_DIR, filename)
    image.save(safe_path, format="JPEG")
    return safe_path


def get_predictor_result(image_path):
    model, classes = load_freshness_model()
    if model is None or not classes:
        return "unknown", 0.0, [], None, None

    try:
        from tensorflow.keras.preprocessing import image as keras_image
        import tensorflow as tf
    except Exception:
        return "unknown", 0.0, [], None, None

    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        probs = np.asarray(preds)
        if probs.ndim == 2 and probs.shape[0] == 1:
            probs = probs[0]

        if not np.isclose(np.sum(probs), 1.0, rtol=1e-3, atol=1e-3):
            try:
                probs = tf.nn.softmax(probs).numpy()
            except Exception:
                exps = np.exp(probs - np.max(probs))
                probs = exps / np.sum(exps)

        if probs.ndim != 1:
            probs = np.ravel(probs)

        top_k = min(3, probs.shape[0])
        top_indices = np.argsort(probs)[::-1][:top_k]

        top3 = []
        for idx in top_indices:
            label = classes[idx] if 0 <= idx < len(classes) else f"unknown_{idx}"
            top3.append({
                "label": label,
                "probability": round(float(probs[idx]) * 100, 2)
            })

        predicted_index = int(top_indices[0])
        predicted_label = classes[predicted_index] if 0 <= predicted_index < len(classes) else "unknown"
        confidence = top3[0]["probability"]
        if confidence < 45.0:
            predicted_label = "unknown"

        return predicted_label, confidence, top3, None, None
    except Exception:
        return "unknown", 0.0, [], None, None


def cv2_barcode_decode(image):
    try:
        image_np = np.array(image.convert("RGB"))
    except Exception:
        return None

    try:
        qr = cv2.QRCodeDetector()
        data, _, _ = qr.detectAndDecode(image_np)
        if data:
            return data
    except Exception:
        pass

    try:
        if hasattr(cv2, "barcode_BarcodeDetector"):
            detector = cv2.barcode_BarcodeDetector()
            ok, decoded_info, _, _ = detector.detectAndDecode(image_np)
            if ok and decoded_info:
                return ", ".join([str(d) for d in decoded_info if d])
    except Exception:
        pass

    return None


def decode_barcode(image):
    if image is None:
        return None

    if PYZBAR_AVAILABLE:
        try:
            decoded = pyzbar_decode(image)
            if decoded:
                return ", ".join([item.data.decode("utf-8", errors="ignore") for item in decoded if item.data])
        except Exception:
            pass

    return cv2_barcode_decode(image)


def speak_alert(message):
    if pyttsx3 is None:
        return False
    try:
        engine = pyttsx3.init("sapi5") if sys.platform == "win32" else pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.setProperty("volume", 0.9)

        def worker():
            try:
                engine.say(message)
                engine.runAndWait()
            except Exception:
                pass

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return True
    except Exception:
        return False


def send_email_alert(sender, password, receiver, food_name="Food Item", freshness="Unknown", expiry_days="Unknown"):
    if not sender or not password or not receiver:
        return False, "Missing email configuration."

    subject = f"Smart Fridge Alert: {food_name}"
    body = f"Food Item: {food_name}\nFreshness: {freshness}\nExpiry: {expiry_days}\n\nThis is a notification from Smart Fridge AI."

    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=20)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, receiver, message.as_string())
        server.quit()
        return True, "Email sent successfully."
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check Gmail app password."
    except Exception as exc:
        return False, str(exc)


def init_session_state():
    if "scan_history" not in st.session_state:
        st.session_state["scan_history"] = []
    if "alert_config" not in st.session_state:
        st.session_state["alert_config"] = {
            "sender": os.getenv("SMARTFRIDGE_SENDER_EMAIL", ""),
            "password": os.getenv("SMARTFRIDGE_APP_PASSWORD", ""),
            "receiver": os.getenv("SMARTFRIDGE_RECEIVER_EMAIL", ""),
        }
    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None
    if "last_capture_path" not in st.session_state:
        st.session_state["last_capture_path"] = None
    if "last_barcode" not in st.session_state:
        st.session_state["last_barcode"] = None


def record_scan_entry(mode, label, confidence=0.0):
    st.session_state["scan_history"].append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "label": label,
            "confidence": float(confidence),
        }
    )


def get_status_counts():
    fresh = sum(1 for item in st.session_state["scan_history"] if item["label"] == "fresh")
    moderate = sum(1 for item in st.session_state["scan_history"] if item["label"] == "moderate")
    spoiled = sum(1 for item in st.session_state["scan_history"] if item["label"] == "spoiled")
    unknown = sum(1 for item in st.session_state["scan_history"] if item["label"] == "unknown")
    return fresh, moderate, spoiled, unknown


def render_home():
    st.title("🧊 Smart Fridge AI")
    st.markdown(
        "Smart Fridge AI brings food freshness prediction, webcam capture, barcode scanning, email alerting, and analytics into one Streamlit dashboard."
    )

    report = load_latest_report()
    fresh, moderate, spoiled, unknown = get_status_counts()
    total_scans = len(st.session_state["scan_history"])

    total_items = report.get("total_items", total_scans)
    total_calories = report.get("total_calories", 0)
    total_protein = report.get("total_protein", 0.0)
    scan_time = report.get("scan_time", "N/A")
    scan_date = scan_time.replace("T", " ") if scan_time != "N/A" else "N/A"

    items = report.get("items", [])
    category_counts = {}
    for item in items:
        category = item.get("category", "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    top_category = max(category_counts, key=category_counts.get) if category_counts else "N/A"
    spoiled_count = report.get("spoiled_count", spoiled)
    waste_rate = f"{round((spoiled_count / total_items) * 100, 1)}%" if total_items else "0%"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Items", total_items)
    col2.metric("Fresh Items", report.get("fresh_count", fresh))
    col3.metric("Moderate Items", report.get("moderate_count", moderate))
    col4.metric("Spoiled Items", spoiled_count)

    col5, col6, col7 = st.columns([1.2, 1.2, 1.6])
    col5.metric("Total Calories", f"{total_calories}")
    col6.metric("Total Protein", f"{total_protein} g")
    col7.metric("Latest Scan", scan_date)

    st.markdown("---")
    st.subheader("Overview")
    if report:
        overview_cols = st.columns(3)
        overview_cols[0].info(f"**Scan Time:**\n{scan_date}")
        overview_cols[1].info(f"**Top Category:**\n{top_category}")
        overview_cols[2].info(f"**Waste Rate:**\n{waste_rate}")
    else:
        st.info("No latest report data found. Use AI prediction or camera capture to create scan results.")

    st.markdown("---")
    st.subheader("Recent Activity")
    if total_scans == 0:
        st.info("No activity yet. Use the menu to scan, predict, or capture images.")
    else:
        history = list(reversed(st.session_state["scan_history"]))[:10]
        st.table(history)

    st.markdown("---")
    st.subheader("How to Use")
    st.write("- Navigate to **AI Freshness Prediction** to upload food photos.")
    st.write("- Use **Camera Capture** to grab an image directly from your webcam.")
    st.write("- Use **Barcode Scanner** to scan QR codes and barcodes.")
    st.write("- Use **Alerts Setup** to configure Gmail notifications and test voice alerts.")


def render_prediction_page():
    st.title("🤖 AI Freshness Prediction")
    st.markdown("Upload a food image and get a predicted freshness class, confidence score, and top predictions.")

    uploaded_file = st.file_uploader("Upload an image file:", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Upload a photo of food to predict freshness.")
        return

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict Freshness"):
        image_path = save_uploaded_image(uploaded_file, prefix="predict")
        prediction, confidence, top3, food_type, gradcam_path = get_predictor_result(image_path)
        st.session_state["last_prediction"] = {
            "path": image_path,
            "prediction": prediction,
            "confidence": confidence,
            "top3": top3,
            "food_type": food_type,
        }

        if prediction != "unknown":
            st.success(f"Predicted: {prediction} ({confidence:.2f}% confidence)")
            st.write(f"**Detected food group:** {food_type or 'Unknown'}")
            record_scan_entry("AI Prediction", prediction, confidence)
        else:
            st.warning("Prediction is unknown or low confidence.")

        if top3:
            st.subheader("Top Predictions")
            for item in top3:
                st.write(f"- {item['label']}: {item['probability']}%")

        if gradcam_path:
            gradcam_path_full = os.path.join(ROOT_DIR, "static", gradcam_path)
            if os.path.exists(gradcam_path_full):
                st.image(gradcam_path_full, caption="Grad-CAM visualization", use_column_width=True)


def render_camera_capture_page():
    st.title("📷 Camera Capture")
    st.markdown("Use your webcam inside Streamlit to capture a food image, save it to static/uploads, and optionally analyze freshness.")

    camera_image = st.camera_input("Point the camera at food and take a picture")
    if camera_image is None:
        st.info("Allow webcam access and capture a photo.")
        return

    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured frame", use_column_width=True)

    if st.button("Save Capture"):
        image_path = save_uploaded_image(camera_image, prefix="camera")
        st.success(f"Saved to {image_path}")
        st.session_state["last_capture_path"] = image_path
        record_scan_entry("Camera Capture", "captured", 0.0)

    if st.button("Analyze Capture"):
        image_path = save_uploaded_image(camera_image, prefix="camera_analyze")
        prediction, confidence, top3, food_type, gradcam_path = get_predictor_result(image_path)
        if prediction != "unknown":
            st.success(f"Predicted: {prediction} ({confidence:.2f}% confidence)")
            st.write(f"**Detected food group:** {food_type or 'Unknown'}")
            record_scan_entry("Camera Prediction", prediction, confidence)
        else:
            st.warning("Prediction is unknown or low confidence.")

        if top3:
            st.subheader("Top Predictions")
            for item in top3:
                st.write(f"- {item['label']}: {item['probability']}%")


def render_barcode_scanner_page():
    st.title("📠 Barcode Scanner")
    st.markdown("Scan a QR code or barcode using your webcam input. The result displays directly in the app.")

    barcode_image = st.camera_input("Point the camera at a barcode or QR code")
    if barcode_image is None:
        st.info("Capture the barcode image and then decode it.")
        return

    image = Image.open(barcode_image).convert("RGB")
    st.image(image, caption="Barcode capture", use_column_width=True)

    if st.button("Decode Barcode"):
        decoded_value = decode_barcode(image)
        if decoded_value:
            st.success("Decoded successfully!")
            st.code(decoded_value)
            st.session_state["last_barcode"] = decoded_value
            record_scan_entry("Barcode Scan", decoded_value, 100.0)
            if speak_alert(f"Barcode detected: {decoded_value}"):
                st.info("Voice alert played.")
        else:
            st.warning("No barcode or QR code detected. Try another angle or better lighting.")

    if st.session_state["last_barcode"]:
        st.markdown("---")
        st.write("**Last decoded value:**")
        st.write(st.session_state["last_barcode"])


def render_dashboard_analytics_page():
    st.title("📊 Dashboard Analytics")
    st.markdown("Professional overview of freshness detection, high priority items, and food insights.")

    report = load_latest_report()
    items = report.get("items", [])
    total_items = report.get("total_items", len(items))
    fresh_count = report.get("fresh_count", sum(1 for item in items if item.get("freshness_status") == "fresh"))
    moderate_count = report.get("moderate_count", sum(1 for item in items if item.get("freshness_status") == "moderate"))
    spoiled_count = report.get("spoiled_count", sum(1 for item in items if item.get("freshness_status") == "spoiled"))
    total_calories = report.get("total_calories", sum(item.get("nutrition", {}).get("calories_kcal", 0) for item in items))
    total_protein = report.get("total_protein", sum(item.get("nutrition", {}).get("protein_g", 0.0) for item in items))
    scan_time = report.get("scan_time", "N/A").replace("T", " ") if report.get("scan_time") else "N/A"

    category_counts = {}
    for item in items:
        category = item.get("category", "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

    danger_items = [item for item in items if item.get("alert", {}).get("level") == "danger"]
    warning_items = [item for item in items if item.get("alert", {}).get("level") == "warning"]
    storage_tips = [item.get("storage_tip") for item in items if item.get("storage_tip")]

    row1, row2 = st.columns(2)
    with row1:
        st.metric("Total Items", total_items)
        st.metric("Fresh Items", fresh_count)
        st.metric("Moderate Items", moderate_count)
        st.metric("Spoiled Items", spoiled_count)
    with row2:
        st.metric("Total Calories", f"{total_calories}")
        st.metric("Total Protein", f"{total_protein} g")
        st.metric("Scan Time", scan_time)
        st.metric("Total Scans", len(st.session_state["scan_history"]))

    st.markdown("---")
    if report and items:
        high_priority = sorted(items, key=lambda i: i.get("priority_score", 0), reverse=True)[:5]
        st.subheader("High Priority Items")
        for item in high_priority:
            st.write(f"**{item.get('display_name', item.get('name','Unknown'))}** — {item.get('freshness_status','unknown').title()} ({item.get('priority_score',0)} priority)")
            st.caption(item.get("alert", {}).get("message", item.get("explanation", "")))

        nutrition_totals = {
            "calories": sum(item.get("nutrition", {}).get("calories_kcal", 0) for item in items),
            "protein": sum(item.get("nutrition", {}).get("protein_g", 0.0) for item in items),
            "carbs": sum(item.get("nutrition", {}).get("carbs_g", 0.0) for item in items),
            "fat": sum(item.get("nutrition", {}).get("fat_g", 0.0) for item in items),
            "fiber": sum(item.get("nutrition", {}).get("fiber_g", 0.0) for item in items),
        }
        nutrition_cols = st.columns(5)
        nutrition_cols[0].metric("Calories", f"{nutrition_totals['calories']}")
        nutrition_cols[1].metric("Protein", f"{nutrition_totals['protein']:.1f} g")
        nutrition_cols[2].metric("Carbs", f"{nutrition_totals['carbs']:.1f} g")
        nutrition_cols[3].metric("Fat", f"{nutrition_totals['fat']:.1f} g")
        nutrition_cols[4].metric("Fiber", f"{nutrition_totals['fiber']:.1f} g")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        if px is not None:
            status_data = {
                "status": ["fresh", "moderate", "spoiled", "unknown"],
                "count": [fresh_count, moderate_count, spoiled_count, total_items - fresh_count - moderate_count - spoiled_count],
            }
            fig_pie = px.pie(status_data, names="status", values="count", title="Freshness Distribution")
            col_a.plotly_chart(fig_pie, use_container_width=True)

            category_data = {
                "category": list(category_counts.keys()),
                "count": list(category_counts.values()),
            }
            fig_categories = px.bar(category_data, x="category", y="count", title="Category Breakdown", labels={"category": "Category", "count": "Items"})
            fig_categories.update_layout(xaxis_tickangle=-45)
            col_b.plotly_chart(fig_categories, use_container_width=True)

            scores = [item.get("priority_score", 0) for item in items]
            names = [item.get("display_name", item.get("name", "Unknown")) for item in items]
            fig_bar = px.bar(x=names, y=scores, title="Item Priority Score", labels={"x": "Item", "y": "Priority"})
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            col_a.warning("Plotly is not available. Charts disabled.")
            col_b.warning("Plotly is not available. Charts disabled.")

        st.markdown("---")
        st.subheader("Top Food Item Insights")
        table_rows = []
        for item in items[:10]:
            table_rows.append(
                {
                    "Food": item.get("display_name", item.get("name", "Unknown")),
                    "Status": item.get("freshness_status", "unknown").title(),
                    "Score": item.get("freshness_score", 0),
                    "Calories": item.get("nutrition", {}).get("calories_kcal", 0),
                    "Protein": item.get("nutrition", {}).get("protein_g", 0.0),
                    "Days Left": item.get("days_left", "N/A"),
                }
            )
        st.table(table_rows)

        st.markdown("---")
        st.subheader("Alert Summary")
        st.write(f"**Danger alerts:** {len(danger_items)} | **Warning alerts:** {len(warning_items)}")
        if danger_items:
            st.error("Danger items require immediate attention. Review the top priority list.")
        elif warning_items:
            st.warning("Some items have warning alerts. Consider using or refrigerating them soon.")
        else:
            st.success("No danger or warning alerts in the latest report.")

        if storage_tips:
            st.markdown("---")
            st.subheader("Storage Tips")
            for tip in storage_tips[:3]:
                st.info(tip)

        st.markdown("---")
        st.subheader("Insights")
        if spoiled_count > 0:
            st.error(f"{spoiled_count} spoiled item(s) detected. Act quickly to reduce waste.")
        if fresh_count >= moderate_count:
            st.success("Most items are in good condition. Keep monitoring freshness regularly.")
        else:
            st.info("Several items are aging. Check the moderate items for early consumption.")
    else:
        st.info("No report data available. Run a scan or use AI prediction to populate analytics.")

    st.markdown("---")
    st.subheader("Recent Scan History")
    st.table(list(reversed(st.session_state["scan_history"]))[:10])


def render_alerts_setup_page():
    st.title("🔔 Alerts Setup")
    st.markdown("Configure email notifications and voice alert support.")

    config = st.session_state["alert_config"]
    with st.form(key="alert_form"):
        sender = st.text_input("Sender Gmail Address", value=config.get("sender", ""))
        password = st.text_input("Gmail App Password", value=config.get("password", ""), type="password")
        receiver = st.text_input("Receiver Email Address", value=config.get("receiver", ""))
        save_settings = st.form_submit_button("Save Settings")

    if save_settings:
        st.session_state["alert_config"] = {
            "sender": sender,
            "password": password,
            "receiver": receiver,
        }
        os.environ["SMARTFRIDGE_SENDER_EMAIL"] = sender
        os.environ["SMARTFRIDGE_APP_PASSWORD"] = password
        os.environ["SMARTFRIDGE_RECEIVER_EMAIL"] = receiver
        st.success("Alert settings saved.")

    st.markdown("---")
    if st.button("Send Test Email"):
        sender = st.session_state["alert_config"].get("sender", "")
        password = st.session_state["alert_config"].get("password", "")
        receiver = st.session_state["alert_config"].get("receiver", "")
        success, message = send_email_alert(sender, password, receiver, food_name="Test Item", freshness="Test", expiry_days="Now")
        if success:
            st.success(message)
        else:
            st.error(message)

    if st.button("Play Test Voice Alert"):
        result = speak_alert("This is a Smart Fridge AI test voice alert.")
        if result:
            st.success("Voice alert should be playing.")
        else:
            st.error("Voice alert is unavailable. Check pyttsx3 installation and audio output.")

    st.markdown("---")
    st.write("**Note:** Use a Gmail account and App Password. Do not use your regular Gmail password.")
    if pyttsx3 is None:
        st.warning("pyttsx3 is not installed. Voice alerts are disabled.")
    if not PYZBAR_AVAILABLE:
        st.warning("pyzbar is unavailable or failed to load. Barcode decoding will use OpenCV QR fallback when possible.")
    if px is None:
        st.warning("Plotly is not installed. Dashboard charts are disabled.")


def main():
    init_session_state()
    st.set_page_config(page_title="Smart Fridge AI", page_icon="🧊", layout="wide")

    with st.sidebar:
        st.title("Smart Fridge AI")
        page = st.selectbox(
            "Menu",
            [
                "Home",
                "AI Freshness Prediction",
                "Camera Capture",
                "Barcode Scanner",
                "Dashboard Analytics",
                "Alerts Setup",
            ],
        )
        st.markdown("---")
        st.write("Use this app to predict food freshness, capture images, scan barcodes, and configure alerts.")

    if page == "Home":
        render_home()
    elif page == "AI Freshness Prediction":
        render_prediction_page()
    elif page == "Camera Capture":
        render_camera_capture_page()
    elif page == "Barcode Scanner":
        render_barcode_scanner_page()
    elif page == "Dashboard Analytics":
        render_dashboard_analytics_page()
    elif page == "Alerts Setup":
        render_alerts_setup_page()


if __name__ == "__main__":
    main()
