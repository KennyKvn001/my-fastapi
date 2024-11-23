from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn
from typing import List
import pickle
import numpy as np

# Load model components first (your existing loading code)
try:
    with open("salary_prediction_model.pkl", "rb") as file:
        model_components = pickle.load(file)

    model = model_components["model"]
    label_encoders = model_components["label_encoders"]
    scalers = model_components["scalers"]
    feature_order = model_components["feature_order"]
except FileNotFoundError:
    raise Exception("Model file 'salary_prediction_model.pkl' not found")


# Create enums for categorical fields based on trained data
class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"


class Degree(str, Enum):
    BACHELORS = "Bachelor's"
    MASTERS = "Master's"
    PHD = "PhD"


class JobTitle(str, Enum):
    SOFTWARE_ENGINEER = "Software Engineer"
    DATA_SCIENTIST = "Data Scientist"
    PRODUCT_MANAGER = "Product Manager"
    SYSTEM_ANALYST = "System Analyst"
    DEVOPS_ENGINEER = "DevOps Engineer"
    DATA_ANALYST = "Data Analyst"
    MACHINE_LEARNING_ENGINEER = "Machine Learning Engineer"
    WEB_DEVELOPER = "Web Developer"
    DATABASE_ADMINISTRATOR = "Database Administrator"
    NETWORK_ENGINEER = "Network Engineer"
    QA_ENGINEER = "Quality Assurance Engineer"
    ACCOUNT_MANAGER = "Account Manager"
    ACCOUNTANT = "Accountant"
    ADMINISTRATIVE_ASSISTANT = "Administrative Assistant"
    BUSINESS_ANALYST = "Business Analyst"
    BUSINESS_DEVELOPMENT_MANAGER = "Business Development Manager"
    BUSINESS_INTELLIGENCE_ANALYST = "Business Intelligence Analyst"
    CEO = "CEO"
    CHIEF_DATA_OFFICER = "Chief Data Officer"
    CHIEF_TECHNOLOGY_OFFICER = "Chief Technology Officer"
    CONTENT_MARKETING_MANAGER = "Content Marketing Manager"
    COPYWRITER = "Copywriter"
    CREATIVE_DIRECTOR = "Creative Director"
    CUSTOMER_SERVICE_MANAGER = "Customer Service Manager"
    CUSTOMER_SERVICE_REP = "Customer Service Rep"
    CUSTOMER_SERVICE_REPRESENTATIVE = "Customer Service Representative"
    CUSTOMER_SUCCESS_MANAGER = "Customer Success Manager"
    CUSTOMER_SUCCESS_REP = "Customer Success Rep"
    DATA_ENTRY_CLERK = "Data Entry Clerk"
    DIGITAL_CONTENT_PRODUCER = "Digital Content Producer"
    DIGITAL_MARKETING_MANAGER = "Digital Marketing Manager"
    DIRECTOR = "Director"
    DIRECTOR_OF_BUSINESS_DEVELOPMENT = "Director of Business Development"
    DIRECTOR_OF_ENGINEERING = "Director of Engineering"
    DIRECTOR_OF_FINANCE = "Director of Finance"
    DIRECTOR_OF_HR = "Director of HR"
    DIRECTOR_OF_HUMAN_CAPITAL = "Director of Human Capital"
    DIRECTOR_OF_HUMAN_RESOURCES = "Director of Human Resources"
    DIRECTOR_OF_MARKETING = "Director of Marketing"
    DIRECTOR_OF_OPERATIONS = "Director of Operations"
    DIRECTOR_OF_PRODUCT_MANAGEMENT = "Director of Product Management"
    DIRECTOR_OF_SALES = "Director of Sales"
    DIRECTOR_OF_SALES_AND_MARKETING = "Director of Sales and Marketing"
    EVENT_COORDINATOR = "Event Coordinator"
    FINANCIAL_ADVISOR = "Financial Advisor"
    FINANCIAL_ANALYST = "Financial Analyst"
    FINANCIAL_MANAGER = "Financial Manager"
    GRAPHIC_DESIGNER = "Graphic Designer"
    HR_GENERALIST = "HR Generalist"
    HR_MANAGER = "HR Manager"
    HELP_DESK_ANALYST = "Help Desk Analyst"
    HUMAN_RESOURCES_DIRECTOR = "Human Resources Director"
    IT_MANAGER = "IT Manager"
    IT_SUPPORT = "IT Support"
    IT_SUPPORT_SPECIALIST = "IT Support Specialist"
    JUNIOR_ACCOUNT_MANAGER = "Junior Account Manager"
    JUNIOR_ACCOUNTANT = "Junior Accountant"
    JUNIOR_ADVERTISING_COORDINATOR = "Junior Advertising Coordinator"
    JUNIOR_BUSINESS_ANALYST = "Junior Business Analyst"
    JUNIOR_BUSINESS_DEVELOPMENT_ASSOCIATE = "Junior Business Development Associate"
    JUNIOR_BUSINESS_OPERATIONS_ANALYST = "Junior Business Operations Analyst"
    JUNIOR_COPYWRITER = "Junior Copywriter"
    JUNIOR_CUSTOMER_SUPPORT_SPECIALIST = "Junior Customer Support Specialist"
    JUNIOR_DATA_ANALYST = "Junior Data Analyst"
    JUNIOR_DATA_SCIENTIST = "Junior Data Scientist"
    JUNIOR_DESIGNER = "Junior Designer"
    JUNIOR_DEVELOPER = "Junior Developer"
    JUNIOR_FINANCIAL_ADVISOR = "Junior Financial Advisor"
    JUNIOR_FINANCIAL_ANALYST = "Junior Financial Analyst"
    JUNIOR_HR_COORDINATOR = "Junior HR Coordinator"
    JUNIOR_HR_GENERALIST = "Junior HR Generalist"
    JUNIOR_MARKETING_ANALYST = "Junior Marketing Analyst"
    JUNIOR_MARKETING_COORDINATOR = "Junior Marketing Coordinator"
    JUNIOR_MARKETING_MANAGER = "Junior Marketing Manager"
    JUNIOR_MARKETING_SPECIALIST = "Junior Marketing Specialist"
    JUNIOR_OPERATIONS_ANALYST = "Junior Operations Analyst"
    JUNIOR_OPERATIONS_COORDINATOR = "Junior Operations Coordinator"
    JUNIOR_OPERATIONS_MANAGER = "Junior Operations Manager"
    JUNIOR_PRODUCT_MANAGER = "Junior Product Manager"
    JUNIOR_PROJECT_MANAGER = "Junior Project Manager"
    JUNIOR_RECRUITER = "Junior Recruiter"
    JUNIOR_RESEARCH_SCIENTIST = "Junior Research Scientist"
    JUNIOR_SALES_REPRESENTATIVE = "Junior Sales Representative"
    JUNIOR_SOCIAL_MEDIA_MANAGER = "Junior Social Media Manager"
    JUNIOR_SOCIAL_MEDIA_SPECIALIST = "Junior Social Media Specialist"
    JUNIOR_SOFTWARE_DEVELOPER = "Junior Software Developer"
    JUNIOR_SOFTWARE_ENGINEER = "Junior Software Engineer"
    JUNIOR_UX_DESIGNER = "Junior UX Designer"
    JUNIOR_WEB_DESIGNER = "Junior Web Designer"
    JUNIOR_WEB_DEVELOPER = "Junior Web Developer"
    MARKETING_ANALYST = "Marketing Analyst"
    MARKETING_COORDINATOR = "Marketing Coordinator"
    MARKETING_MANAGER = "Marketing Manager"
    MARKETING_SPECIALIST = "Marketing Specialist"
    OFFICE_MANAGER = "Office Manager"
    OPERATIONS_ANALYST = "Operations Analyst"
    OPERATIONS_DIRECTOR = "Operations Director"
    OPERATIONS_MANAGER = "Operations Manager"
    PRINCIPAL_ENGINEER = "Principal Engineer"
    PRINCIPAL_SCIENTIST = "Principal Scientist"
    PRODUCT_DESIGNER = "Product Designer"
    PRODUCT_MARKETING_MANAGER = "Product Marketing Manager"
    PROJECT_ENGINEER = "Project Engineer"
    PROJECT_MANAGER = "Project Manager"
    PUBLIC_RELATIONS_MANAGER = "Public Relations Manager"
    RECRUITER = "Recruiter"
    RESEARCH_DIRECTOR = "Research Director"
    RESEARCH_SCIENTIST = "Research Scientist"
    SALES_ASSOCIATE = "Sales Associate"
    SALES_DIRECTOR = "Sales Director"
    SALES_EXECUTIVE = "Sales Executive"
    SALES_MANAGER = "Sales Manager"
    SALES_OPERATIONS_MANAGER = "Sales Operations Manager"
    SALES_REPRESENTATIVE = "Sales Representative"
    SENIOR_ACCOUNT_EXECUTIVE = "Senior Account Executive"
    SENIOR_ACCOUNT_MANAGER = "Senior Account Manager"
    SENIOR_ACCOUNTANT = "Senior Accountant"
    SENIOR_BUSINESS_ANALYST = "Senior Business Analyst"
    SENIOR_BUSINESS_DEVELOPMENT_MANAGER = "Senior Business Development Manager"
    SENIOR_CONSULTANT = "Senior Consultant"
    SENIOR_DATA_ANALYST = "Senior Data Analyst"
    SENIOR_DATA_ENGINEER = "Senior Data Engineer"
    SENIOR_DATA_SCIENTIST = "Senior Data Scientist"
    SENIOR_ENGINEER = "Senior Engineer"
    SENIOR_FINANCIAL_ADVISOR = "Senior Financial Advisor"
    SENIOR_FINANCIAL_ANALYST = "Senior Financial Analyst"
    SENIOR_FINANCIAL_MANAGER = "Senior Financial Manager"
    SENIOR_GRAPHIC_DESIGNER = "Senior Graphic Designer"
    SENIOR_HR_GENERALIST = "Senior HR Generalist"
    SENIOR_HR_MANAGER = "Senior HR Manager"
    SENIOR_HR_SPECIALIST = "Senior HR Specialist"
    SENIOR_HUMAN_RESOURCES_COORDINATOR = "Senior Human Resources Coordinator"
    SENIOR_HUMAN_RESOURCES_MANAGER = "Senior Human Resources Manager"
    SENIOR_HUMAN_RESOURCES_SPECIALIST = "Senior Human Resources Specialist"
    SENIOR_IT_CONSULTANT = "Senior IT Consultant"
    SENIOR_IT_PROJECT_MANAGER = "Senior IT Project Manager"
    SENIOR_IT_SUPPORT_SPECIALIST = "Senior IT Support Specialist"
    SENIOR_MANAGER = "Senior Manager"
    SENIOR_MARKETING_ANALYST = "Senior Marketing Analyst"
    SENIOR_MARKETING_COORDINATOR = "Senior Marketing Coordinator"
    SENIOR_MARKETING_DIRECTOR = "Senior Marketing Director"
    SENIOR_MARKETING_MANAGER = "Senior Marketing Manager"
    SENIOR_MARKETING_SPECIALIST = "Senior Marketing Specialist"
    SENIOR_OPERATIONS_ANALYST = "Senior Operations Analyst"
    SENIOR_OPERATIONS_COORDINATOR = "Senior Operations Coordinator"
    SENIOR_OPERATIONS_MANAGER = "Senior Operations Manager"
    SENIOR_PRODUCT_DESIGNER = "Senior Product Designer"
    SENIOR_PRODUCT_DEVELOPMENT_MANAGER = "Senior Product Development Manager"
    SENIOR_PRODUCT_MANAGER = "Senior Product Manager"
    SENIOR_PRODUCT_MARKETING_MANAGER = "Senior Product Marketing Manager"
    SENIOR_PROJECT_COORDINATOR = "Senior Project Coordinator"
    SENIOR_PROJECT_MANAGER = "Senior Project Manager"
    SENIOR_QUALITY_ASSURANCE_ANALYST = "Senior Quality Assurance Analyst"
    SENIOR_RESEARCH_SCIENTIST = "Senior Research Scientist"
    SENIOR_RESEARCHER = "Senior Researcher"
    SENIOR_SALES_MANAGER = "Senior Sales Manager"
    SENIOR_SALES_REPRESENTATIVE = "Senior Sales Representative"
    SENIOR_SCIENTIST = "Senior Scientist"
    SENIOR_SOFTWARE_ARCHITECT = "Senior Software Architect"
    SENIOR_SOFTWARE_DEVELOPER = "Senior Software Developer"
    SENIOR_SOFTWARE_ENGINEER = "Senior Software Engineer"
    SENIOR_TRAINING_SPECIALIST = "Senior Training Specialist"
    SENIOR_UX_DESIGNER = "Senior UX Designer"
    SOCIAL_MEDIA_MANAGER = "Social Media Manager"
    SOCIAL_MEDIA_SPECIALIST = "Social Media Specialist"
    SOFTWARE_DEVELOPER = "Software Developer"
    SOFTWARE_MANAGER = "Software Manager"
    SOFTWARE_PROJECT_MANAGER = "Software Project Manager"
    STRATEGY_CONSULTANT = "Strategy Consultant"
    SUPPLY_CHAIN_ANALYST = "Supply Chain Analyst"
    SUPPLY_CHAIN_MANAGER = "Supply Chain Manager"
    TECHNICAL_RECRUITER = "Technical Recruiter"
    TECHNICAL_SUPPORT_SPECIALIST = "Technical Support Specialist"
    TECHNICAL_WRITER = "Technical Writer"
    TRAINING_SPECIALIST = "Training Specialist"
    UX_DESIGNER = "UX Designer"
    UX_RESEARCHER = "UX Researcher"
    VP_OF_FINANCE = "VP of Finance"
    VP_OF_OPERATIONS = "VP of Operations"


class InputData(BaseModel):
    age: int = Field(..., description="Age of the employee", ge=18, le=100, example=30)
    gender: Gender = Field(..., description="Gender of the employee", example="Male")
    degree: Degree = Field(..., description="Education level", example="Bachelor's")
    job_title: JobTitle = Field(
        ..., description="Job title", example="Software Engineer"
    )
    experience: float = Field(
        ..., description="Years of experience", ge=0, le=50, example=5.0
    )


class PredictionResponse(BaseModel):
    salary_prediction: float = Field(..., description="Predicted salary in USD")
    currency: str = Field(..., description="Currency of the prediction")


class CategoryResponse(BaseModel):
    genders: List[str] = Field(..., description="List of valid gender categories")
    degrees: List[str] = Field(..., description="List of valid degree categories")
    job_titles: List[str] = Field(..., description="List of valid job titles")


class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Current health status of the API")
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded successfully"
    )
    preprocessors_loaded: dict = Field(
        ..., description="Status of preprocessor components"
    )


app = FastAPI(
    title="Salary Prediction API",
    description="""
    This API predicts salaries based on various employee features.
    
    ## Features
    * Predict salary based on employee characteristics
    * Get valid categories for categorical inputs
    * Check API health status
    
    ## Notes
    * All salary predictions are in USD
    * Age must be between 18 and 100
    * Experience must be between 0 and 50 years
    """,
    version="1.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Your Name",
        "url": "http://example.com/contact/",
        "email": "your.email@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict salary based on employee features",
    response_description="Returns the predicted salary in USD",
)
async def predict(input_data: InputData):
    """
    Predicts salary based on employee characteristics.

    ## Parameters
    * **age**: Employee's age (18-100)
    * **gender**: Employee's gender
    * **degree**: Education level
    * **job_title**: Current job title
    * **experience**: Years of experience (0-50)

    ## Returns
    * **salary_prediction**: Predicted salary in USD
    * **currency**: Currency code (USD)

    ## Errors
    * **400**: Invalid input data
    * **500**: Server error during prediction
    """
    try:
        input_features = {}

        input_features["age_scale"] = scalers["age"].transform([[input_data.age]])[0][0]
        input_features["experience_scale"] = scalers["experience"].transform(
            [[input_data.experience]]
        )[0][0]

        try:
            input_features["gender_num"] = label_encoders["gender"].transform(
                [input_data.gender]
            )[0]
            input_features["degree_num"] = label_encoders["degree"].transform(
                [input_data.degree]
            )[0]
            input_features["job-title_num"] = label_encoders["job-title"].transform(
                [input_data.job_title]
            )[0]
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail="Invalid category provided. Please ensure all categories are correct.",
            )

        features = np.array([[input_features[feature] for feature in feature_order]])
        prediction = model.predict(features)[0]

        return {"salary_prediction": round(prediction, 2), "currency": "USD"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get(
    "/categories",
    response_model=CategoryResponse,
    summary="Get valid categories",
    response_description="Returns lists of valid categories for categorical inputs",
)
async def get_categories():
    """
    Returns all valid categories for categorical features.

    ## Returns
    * **genders**: List of valid gender categories
    * **degrees**: List of valid education levels
    * **job_titles**: List of valid job titles
    """
    return {
        "genders": label_encoders["gender"].classes_.tolist(),
        "degrees": label_encoders["degree"].classes_.tolist(),
        "job_titles": label_encoders["job-title"].classes_.tolist(),
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Check API health",
    response_description="Returns the current health status of the API",
)
async def health_check():
    """
    Checks the health status of the API and its components.

    ## Returns
    * **status**: Current health status
    * **model_loaded**: Whether the model is loaded
    * **preprocessors_loaded**: Status of preprocessing components
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessors_loaded": {
            "encoders": list(label_encoders.keys()),
            "scalers": list(scalers.keys()),
        },
    }


# Customize OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": str(exc)},
    )


import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

if __name__ == "__main__":
    # Running the FastAPI application using uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        workers=4,
        log_level="debug",
    )
