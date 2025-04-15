FROM python:3.9-slim

# 1) Install system packages needed to build Python/C extensions
RUN apt-get update && apt-get install -y gcc

# 2) Create a working directory and copy your files
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# 3) Install Python dependencies
RUN pip install --no-cache-dir flask flask-cors gunicorn -r requirements.txt

# 4) Copy the rest of your project
COPY . /app

EXPOSE 8080
CMD ["gunicorn", "--bind", ":8080", "app:app"]
