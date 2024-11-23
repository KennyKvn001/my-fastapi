# Salary Prediction API

This API is designed to predict salaries based on various factors such as gender, degree, job title, and other features. It utilizes a trained machine learning model to provide accurate predictions.

## Endpoints

The API has a single endpoint for predicting salaries. The endpoint is `/predict` and it accepts a POST request with the following JSON body:

```json
{
  "gender": "MALE" or "FEMALE",
  "degree": "BACHELORS", "MASTERS", or "PHD",
  "job_title": One of the job titles listed in the `JobTitle` enum,
  "other_features": Additional features as required by the model
}
```

## Response

The API returns a JSON response with the predicted salary. The response format is as follows:

```json
{
  "predicted_salary": <predicted_salary_value>
}
```

## How to Use

1. Ensure you have the API running on a server or locally using a tool like `uvicorn`.
2. Use a tool like `curl` or a HTTP client library in your preferred programming language to send a POST request to the `/predict` endpoint.
3. In the request body, provide the required information in the format specified above.
4. The API will respond with the predicted salary based on the provided information.

Example using `curl`:
```bash
curl -X POST \
  http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender": "MALE", "degree": "BACHELORS", "job_title": "SOFTWARE_ENGINEER", "other_features": {"feature1": "value1", "feature2": "value2"}}'
```

Replace `http://localhost:8000` with the actual URL where the API is running.

## Model Details

The model used for prediction is a trained machine learning model that has been serialized and stored in the `salary_prediction_model.pkl` file. The model is loaded into the API at startup, and it uses the following components:

* `model`: The trained machine learning model.
* `label_encoders`: Encoders for categorical fields.
* `scalers`: Scalers for numerical fields.
* `feature_order`: The order of features expected by the model.

The model is designed to work with the following categorical fields:

* `Gender`: MALE or FEMALE.
* `Degree`: BACHELORS, MASTERS, or PHD.
* `JobTitle`: One of the job titles listed in the `JobTitle` enum.

## Error Handling

The API is designed to handle errors gracefully. If an error occurs during prediction, the API will return a JSON response with an error message. The response format for errors is as follows:

```json
{
  "error": "<error_message>"
}
```

## Development

To develop or modify the API, ensure you have the following dependencies installed:

* `fastapi`
* `uvicorn`
* `pydantic`
* `numpy`
* `pickle`

The API code is structured as follows:

* `main.py`: The main application file that defines the API endpoints and logic.
* `salary_prediction_model.pkl`: The serialized machine learning model file.

To run the API locally, use the following command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
