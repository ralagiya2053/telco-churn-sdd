FROM python:3.12-slim

# Set the working directory to /var/task as required by AWS Lambda
WORKDIR /var/task

# Copy requirements first to leverage Docker cache
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy the inference handler and the saved model artifacts
COPY src/handler.py .
COPY model/ ./model/

# Set the Lambda handler entrypoint
CMD ["handler.lambda_handler"]
