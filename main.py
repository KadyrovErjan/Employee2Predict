from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib

hr_app = FastAPI()

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

roles = ['Human Resources', 'Laboratory Technician', 'Manager', 'Manufacturing Director', 'Research Director', 'Research Scientist', 'Sales Executive', 'Sales Representative']

class EmployeeSchema(BaseModel):
    OverTime: str
    MonthlyIncome: float
    DistanceFromHome: float
    JobRole: str
    JobSatisfaction: float
    Age: int
    EnvironmentSatisfaction: float
    YearsAtCompany: float
    WorkLifeBalance: float

@hr_app.post('/predict/')
async def predict(person: EmployeeSchema):
    print(person)
    person_dict = person.dict()
    print(person_dict)

    new_over_time = person_dict.pop('OverTime')
    over_time1_0 = [1 if new_over_time == 'Yes' else 0]

    new_job = person_dict.pop('JobRole')
    job1_0 = [1 if new_job == i else 0 for i in roles]

    features = list(person_dict.values()) + over_time1_0 + job1_0
    print(features)

    scaled = scaler.transform([features])
    print(model.predict(scaled))
    print(model.predict(scaled)[0])
    pred = model.predict(scaled)[0]
    print(model.predict_proba(scaled))
    print(model.predict_proba(scaled)[0][1])

    prob = model.predict_proba(scaled)[0][1]

    return {"approved": pred, "probability": round(prob, 2)}


if __name__ == '__main__':
    uvicorn.run(hr_app, host="127.0.0.1", port=8001)



