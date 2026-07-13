# 🦺 SafeVision AI – AI Powered PPE Detection for Construction Sites

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?style=for-the-badge&logo=flask)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-AI-red?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite)
![Hackathon](https://img.shields.io/badge/Hackathon-Project-orange?style=for-the-badge)

### 🚧 AI-Powered PPE Detection for Safer Construction Sites

Real-Time Construction Worker Safety Monitoring using Computer Vision & Deep Learning

</div>

---

# 📌 Overview

**SafeVision AI** is an AI-powered Personal Protective Equipment (PPE) Detection System designed for construction sites.

The system monitors workers in real time using AI, detects whether mandatory safety equipment is being worn, and instantly alerts supervisors whenever a safety violation occurs.

The application supports:

- 📹 CCTV Cameras
- 🎥 Live Webcam
- 🖼 Image Upload
- 📂 Video Upload

using **YOLOv8**, **OpenCV**, and **Flask**.

---

# 🎯 Problem Statement

Construction sites are one of the most accident-prone workplaces.

Common reasons include:

- Workers not wearing helmets
- Missing safety vests
- Missing gloves
- Missing boots
- Missing face masks
- Lack of continuous supervision

Manual monitoring is expensive, inefficient, and prone to human error.

---

# 💡 Solution

SafeVision AI continuously monitors workers using Computer Vision.

The AI automatically:

✅ Detects workers

✅ Detects PPE

✅ Identifies violations

✅ Displays live alerts

✅ Stores detection history

✅ Generates analytics

---

# 🚀 Features

## Live Detection

- Live Webcam
- CCTV Stream
- Real-time Detection
- FPS Counter
- Confidence Score

---

## PPE Detection

- 🪖 Helmet Detection
- 🦺 Safety Vest Detection
- 🥽 Safety Goggles Detection
- 🧤 Gloves Detection
- 👢 Safety Boots Detection
- 😷 Face Mask Detection

---

## Dashboard

- Worker Count
- Helmet Compliance
- Vest Compliance
- Safety Score
- Active Violations
- Detection Accuracy
- Compliance Percentage

---

## AI Features

- YOLOv8 Detection
- OpenCV Processing
- Motion Detection
- ByteTrack Support
- Real-Time Alerts
- Analytics Dashboard
- SQLite Database
- Image Processing
- Video Processing

---

# 🛠 Technology Stack

## Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- Chart.js

## Backend

- Python
- Flask
- OpenCV
- Ultralytics YOLOv8

## Database

- SQLite

## Tools

- Git
- GitHub
- VS Code

---

# 🏗 System Architecture

```text
        CCTV Camera / Webcam
                 │
                 ▼
          Video Stream
                 │
                 ▼
       OpenCV Frame Capture
                 │
                 ▼
      YOLOv8 Object Detection
                 │
                 ▼
      PPE Compliance Analysis
                 │
                 ▼
      Safety Violation Engine
                 │
                 ▼
          SQLite Database
                 │
                 ▼
        Analytics Dashboard
                 │
                 ▼
      Supervisor Alert System
```

---

# 📂 Project Structure

```text
SafeVision-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── database.db
│
├── models/
│      └── ppe.pt
│
├── templates/
│      ├── index.html
│      ├── live.html
│      ├── dashboard.html
│      ├── analytics.html
│      ├── about.html
│      └── contact.html
│
├── static/
│      ├── css/
│      ├── js/
│      ├── uploads/
│      ├── processed/
│      └── images/
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/SafeVision-AI.git
```

```bash
cd SafeVision-AI
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

# 📊 Dashboard

The dashboard provides

- Workers Detected
- Helmet Compliance
- Vest Compliance
- Gloves Detection
- Safety Score
- Daily Violations
- Detection Accuracy

---

# 📈 Analytics

- Daily Reports
- Monthly Reports
- Helmet Compliance
- Vest Compliance
- Worker Statistics
- AI Accuracy
- Safety Trends

---

# 📸 Screenshots

## Home Page

> Add Screenshot Here

---

## Live Detection

> Add Screenshot Here

---

## Dashboard

> Add Screenshot Here

---

## Analytics

> Add Screenshot Here

---

## Alerts

> Add Screenshot Here

---

# 📡 API Endpoints

| Method | Endpoint | Description |
|----------|-------------------|----------------|
| GET | / | Home |
| GET | /live | Live Detection |
| GET | /dashboard | Dashboard |
| GET | /analytics | Analytics |
| POST | /api/upload_image | Upload Image |
| POST | /api/upload_video | Upload Video |
| GET | /video_feed | Live Webcam |
| GET | /api/stats | Statistics |
| GET | /api/history | Detection History |

---

# 🔮 Future Improvements

- Email Alerts
- SMS Notifications
- Cloud Deployment
- Multi-Camera Support
- Face Recognition
- Employee Attendance
- AI Report Generator
- Android App
- Admin Login
- Supervisor Login

---

# 👨‍💻 Team

| Name | Role |
|-------|------|
| Your Name | Full Stack Developer |
| Team Member | AI Engineer |
| Team Member | Frontend Developer |

---

# 📬 Contact

Email

```
your@email.com
```

GitHub

```
https://github.com/yourusername
```

LinkedIn

```
https://linkedin.com/in/yourprofile
```

---

# 📜 License

This project is developed for educational purposes, research, and hackathon demonstrations.

---

# ⭐ Support

If you like this project, please ⭐ Star the repository on GitHub.

---

<div align="center">

## 🚀 SafeVision AI

### Making Construction Sites Safer with Artificial Intelligence

Made with ❤️ using Python, Flask, OpenCV & YOLOv8

</div>

# SafeVision — Smart PPE Detection Platform

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
SafeVision/
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

