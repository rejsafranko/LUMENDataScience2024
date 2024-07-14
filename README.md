# LUMEN Data Science 2024 2nd Place Solution
## Hotel Occupancy Prediction with Time-Series Modeling

Competition article:

https://www.alfatec.ai/academy/resource-library/lumen-data-science-competition-finals-the-winners-are-vizionari-3.0

Original repo:

https://gitlab.com/overfittingoverlords/lumen-hotel-occupancy-prediction

This is a mirrored repository of our project on GitLab that won 2nd place in the LUMEN Data Science 2024 competition. The task was to predict the occupancy of hotel rooms in a future point of time. Our AWS Lambda deployment is currently offline but we provided a Flask server API that can be run in a local environment. The repository was handed over for grading with an unknown test set but we will create and upload a new data split soon for playing with the project.

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

**Access via HTTP GET Request to AWS Lambda**
- **Multi-step API Endpoint:** **https://cv7b3eubc3ddeeqklbxndce2n40rimfk.lambda-url.eu-central-1.on.aws/**
- **One-step API Endpoint:** **https://bu33nynn6hgg37pigtankuyypq0asggf.lambda-url.eu-central-1.on.aws/**

### 2. Project Structure

The project is organized into several directories, each with a specific type of content:

- `/.git` - This directory contains Git version control system files, which are used for tracking changes in the project files over time.

- `/notebooks` - Jupyter notebooks with data cleaning, exploratory data analysis, model development, etc. can be found here.

- `/one-step-multi-step-api` - This directory houses the combined API for both one-step and multi-step prediction models.

- `/daily-one-step-api-v1` - This folder holds the the API for the daily one-step prediction model.

- `/data` - Contains datasets used by the models, including training, validation, and test sets.

- `/documentation` - Stores project documentation.

- `/models` - Includes trained model files. 


### 3. AWS Deployment

We chose a serverless option to deploy the model. We used an AWS S3 bucket in the Europe Central 1 Region (Frankfurt data center) to store the best performing model. To use the model for prediction, we deployed a Docker image as an AWS Lambda function. The Docker image is built in the AWS Lambda Python 3.11 runtime alongside the packages specified in the requirements file. The image contains prediction functions (one-step, multi-step) which are pinged with an HTTP GET Request. The prediction function makes the neccesary data transforms, loads the models and data from the S3 bucket, makes predictions for all the models and returns the prediction metrics as a response. The function also fills a MySQL Database hosted on AWS RDS with the prediction results. Those results are used to generate dashboards in Looker Studio which update every 15 minutes or on-demand.

The deployment code is in ```src/api/```. Inside this directory we initialized an AWS CDK project in Typescript. The crucial part of the ```src/api/``` CDK project are directories ```src/api/image/``` and ```src/api/lib/```. The ```src/api/image/``` contains the Dockerfile and the predict function Python files. The ```src/api/lib/``` contains a Typescript file which defines the Lambda Function Stack configuration (function source code, memory size, timeout, function URL, CORS, authentication).

With this setup, a client just pings the Lambda function endpoint with a GET request which contains the input data in the body parameter. Since one Lambda function call costs $0.0000002, this setup is a cheap option which eliminates the need for maintaining a server.


Commands used for creating an AWS CDK Project:

```npm install -g aws-cdk```

```cdk init app --language typescript```



Commands used to deploy the Lambda function:

```cdk bootstrap --region [REGION]```

```cdk deploy```
