https://lizpynuyrfbnwnp52nblyy3syi0lyvpb.lambda-url.eu-central-1.on.aws/

HTTP POST Request BODY: {"input":[6.1, 2.8, 4.7, 1.2]}



# Documentation

### 1. Usage

### 2. Data

### 3. Model

### 4. Deployment

We chose a serverless option to deploy the model. We used an AWS S3 bucket in the Europe Central 1 Region (Frankfurt data center) to store the best performing model. To use the model for prediction, we deployed a Docker image as an AWS Lambda function. The Docker image is built in the AWS Lambda Python 3.11 runtime alongside the packages specified in the requirements file. The image contains a prediction function which takes input data from the HTTP POST Request Body. The prediction function makes the neccesary data transforms, loads the model from the S3 bucket, makes a prediction and returns the predicted value.

The deployment code is in src/api. Inside this directory we initialized an AWS CDK project in Typescript. The crucial part of the src/api/ CDK project are directories src/api/image and src/api/lib. The src/api/image contains the Dockerfile and the predict function Python file. The src/api/lib contains a Typescript file which defines the Lambda Function Stack configuration (function source code, memory size, timeout, function URL, CORS, authentication).

With this setup, a client just pings the Lambda function endpoint with a POST request which contains the input data in the body parameter. Since one Lambda function call costs $0.0000002, this setup is a cheap option which eliminates the need for maintaining a server.