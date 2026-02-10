# 1. Base Image: Official lightweight Python
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for caching speed)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project into the container
COPY . .

# 6. Expose the port (Good practice, though Render ignores this)
EXPOSE 8000

# 7. Start the server
# We use "0.0.0.0" so the outside world can reach it
# We point to "zone_3_inference.app.main:app" based on  file structure
CMD ["uvicorn", "zone_3_inference.app.main:app", "--host", "0.0.0.0", "--port", "8000"]