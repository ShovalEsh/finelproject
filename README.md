# Phishing Alert App

# Requirements

Before running the project, install:

- Python 3.11
- Visual Studio Code
- Android Studio
- Android SDK
- Git

# Backend Setup (Visual Studio Code)

### 1. Clone the repository

```bash
git clone https://github.com/ShovalEsh/finelproject.git
```

### 2. Open the project

Open the **TEXT_ALERT** folder in Visual Studio Code.

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

Windows

```bash
venv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```

### 5. Run the backend

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8100
```

The server will be available at:

```
http://localhost:8100/docs
```

---

# Android Application (Android Studio)

1. Open **PhishingAlertApp** in Android Studio.
2. Wait for Gradle Sync to finish.
3. Make sure the backend server is running.
4. In `ApiService.kt`, verify that:

```kotlin
private const val BASE_URL = "http://10.0.2.2:8100/"
```

5. Run the application on an emulator or Android device.

---

# Using the System

1. Start the backend.
2. Open the Android application.
3. Enter or paste a message.
4. Press **Analyze Message**.
5. View the risk score, explanation and recommendations.

---

# Project Structure

```
TEXT_ALERT/
    Backend (FastAPI + AI Model)

PhishingAlertApp/
    Android Application
```

---

# Troubleshooting

If the backend does not start:

- Make sure the virtual environment is activated.
- Run:

```bash
pip install -r requirements.txt
```

If the Android application cannot connect:

- Verify that the backend is running.
- Verify that `BASE_URL` points to:

```
http://10.0.2.2:8100/
```

when using the Android emulator.
