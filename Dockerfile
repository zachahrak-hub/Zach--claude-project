# Playwright base image — Chromium already installed, no re-download on deploys
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

# Install Python deps first — cached by Docker unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code last — only this layer rebuilds on code changes
COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
