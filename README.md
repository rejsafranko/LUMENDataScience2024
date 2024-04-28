
# Documentation

### 1. Usage
**How to Run the Multi-step and One-step Prediction APIs**

**Multi-step and One-step API**
**Set Up Your Environment:**
- **Navigate** to the **one-step-multi-step-api folder**.
- **Execute** the command **python app.py** in your command prompt.

**Upload and Visualize Data:**
- **Access** the web interface at **http://127.0.0.1:1234**.
- **Upload** your **test.parquet** file to view the results.

**Results:**
- The interface will display **daily, weekly, and monthly predictions** for both **one-step** and **multi-step** processes across the entire test dataset.

**Specific One-step Daily Model**
**Set Up Your Environment:**
- **Navigate** to the **daily-one-step-api-v1/flask_app folder**.
- **Execute** the command **python app.py** in your command prompt.

**Upload and Visualize Data:**
- **Access** the web interface at **http://127.0.0.1:5000**.
- **Upload** your **test.parquet** file. Note that visualization may take some time due to the model specifics.

**Model Specifics:**
- This model integrates a **classification scheme** for predicting **reservation cancellations** and a **regression model** for estimating the number of **occupied rooms**.

**Access via HTTP GET**
- **Multi-step API Endpoint:** **https://cv7b3eubc3ddeeqklbxndce2n40rimfk.lambda-url.eu-central-1.on.aws/**
- **One-step API Endpoint:** **https://bu33nynn6hgg37pigtankuyypq0asggf.lambda-url.eu-central-1.on.aws/**

### 2. Data

### 3. Model

### 4. Deployment

We chose a serverless option to deploy the model. We used an AWS S3 bucket in the Europe Central 1 Region (Frankfurt data center) to store the best performing model. To use the model for prediction, we deployed a Docker image as an AWS Lambda function. The Docker image is built in the AWS Lambda Python 3.11 runtime alongside the packages specified in the requirements file. The image contains prediction functions (one-step, multi-step) which are pinged with an HTTP GET Request. The prediction function makes the neccesary data transforms, loads the models and data from the S3 bucket, makes predictions for all the models and returns the prediction metrics as a response. The function also fills a MySQL Database hosted on AWS RDS with the prediction results. Those results are used to generate dashboards in Looker Studio which update every 15 minutes or on-demand.

The deployment code is in ```src/api/```. Inside this directory we initialized an AWS CDK project in Typescript. The crucial part of the ```src/api/``` CDK project are directories ```src/api/image/``` and ```src/api/lib/```. The ```src/api/image/``` contains the Dockerfile and the predict function Python files. The ```src/api/lib/``` contains a Typescript file which defines the Lambda Function Stack configuration (function source code, memory size, timeout, function URL, CORS, authentication).

With this setup, a client just pings the Lambda function endpoint with a GET request which contains the input data in the body parameter. Since one Lambda function call costs $0.0000002, this setup is a cheap option which eliminates the need for maintaining a server.


Commands used for creating an AWS CDK Project:

```npm install -g aws-cdk```

```cdk init app --language typescript```



Commands used to deploy the Lambda function:

```cdk bootstrap --region [REGION]```

```cdk deploy```