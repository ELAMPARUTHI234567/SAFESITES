# SafeVision AI — Smart PPE Detection Platform

SafeVision AI is a fully functional, glassmorphic visual surveillance platform that integrates web camera streaming and manual image or video uploads to identify safety equipment compliance (helmets, vests, goggles, masks, and boots) using the YOLOv8 model.

---

## ⚡ Key Features

- **Webcam Surveillance Sandbox**: Renders real-time camera views (with automated fallback simulated feeds if no hardware camera is present), drawing compliant (green) or non-compliant (red) bounding frames.
- **Image compliance scan**: Allows testing photos to assess PPE adherence and display confidence statistics.
- **CCTV video inspection runs**: Evaluates MP4 loops frame-by-frame and renders annotated output.
- **Automated Sound Alerts**: Synthesizes audible alarms in modern browsers (using Web Audio API) if safety exceptions are flagged.
- **Unified analytics dashboard**: Populates lines, bar chart, and donut graphs dynamically from SQLite databases.
- **Robust Model Fallback Engine**: Attempts to load custom PPE-trained models from `models/ppe.pt`. If absent, defaults to pre-trained Person detection model `yolov8n.pt` rather than failing, displaying a "Custom PPE Model Not Found" watermark overlay.

---

## 📂 Folder Structure

```text
SafeVision-AI/
├── app.py
├── requirements.txt
├── README.md
├── database.db (created on first run)
├── models/
│   └── yolov8n.pt (downloaded automatically or placed manually)
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   ├── live.html
│   ├── analytics.html
│   ├── about.html
│   └── contact.html
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   ├── app.js
    │   └── dashboard.js
    ├── uploads/ (created automatically)
    └── processed/ (created automatically)
```

---

## 🚀 Setup & Launch

Follow these steps to run the application locally on **Windows**:

### 1. Establish Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Environment
```powershell
venv\Scripts\activate
```

### 3. Install Package Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Execute Backend App Server
```powershell
python app.py
```

### 5. Access the Platform
Once launched, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🛡️ Technical Overview

1. **Flask Engine**: Manages HTTP request endpoints, renders Jinja page templates, saves media streams, and interacts with SQLite records.
2. **OpenCV API**: Triggers hardware webcam devices and handles frame-writing pipelines for file evaluation feeds.
3. **Ultralytics YOLOv8**: Processes frames to isolate person elements, extracting confidence arrays and drawing overlay boxes.
4. **SQLite database**: Saves historical safety compliance metrics over structured audits.
5. **Chart.js Canvas**: Pulls logs from endpoint collections to populate interactive data graphs.
6. **Web Audio Alarm**: Plays dual-tone sawtooth oscillation beeps during safety infractions.
