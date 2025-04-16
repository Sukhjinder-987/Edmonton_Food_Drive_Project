# **API Documentation for `predict_api.py`**

## **Overview**
This API is built using Flask to provide sentiment predictions based on input data. It loads a pre-trained model and processes input data to return predictions on the sentiment of comments.

---

## **Base URL**

http://localhost:5001
---

## **Endpoints**

### 1. **Health Check**
This endpoint verifies that the API is running.

#### **Endpoint**:

GET /health


**Sample Request**
```bash
curl -X GET http://localhost:5001/health

#### **Sample Request**:
```bash

{
  "status": "API is running"
}

POST /predict

curl -X POST http://127.0.0.1:5001/predict -H "Content-Type: application/json" -d '{
  "Ward/Branch": "Silver Berry Ward",
  "Completed More Than One Route": "Yes",
  "Comment Sentiments": "Positive"
}'

{
  "Ward/Branch": "Silver Berry Ward",
  "Completed More Than One Route": "Yes",
  "Comment Sentiments": "Positive"
}

{
  "prediction": "Positive"
}

{
  "error": "ValueError: Input contains NaN"
}

'Silver Berry Ward': 0,
'Crawford Plains Ward': 1,
'Lee Ridge Ward': 2,
'Griesbach Ward': 3,
'Londonderry Ward': 4,
'Ellerslie Ward': 5,
'Blackmud Creek Ward': 6,
'Clareview Ward': 7,
'Rutherford Ward': 8,
'Southgate Ward': 9,
'Forest Heights Ward': 10,
'Rabbit Hill Ward': 11,
'Greenfield Ward': 12,
'Terwillegar Park Ward': 13,
'Namao Ward': 14,
'Woodbend Ward': 15,
'Connors Hill Ward': 16,
'Stony Plain Ward': 17,
'Strathcona Married Student Ward': 18,
'Rio Vista Ward': 19,
'Beaumont Ward': 20,
'Wild Rose Ward': 21,
'Drayton Valley Ward': 22,
'Wainwright Branch': 23,
'Lago Lindo Branch': 24,
'Pioneer Ward': 25

"Completed More Than One Route"
    "Yes" → 1
    "No" → 0

"Comment Sentiments"
    "Positive" → 1
    "Negative" → 0
    "Neutral" → 2

0 → Negative
1 → Positive
2 → Neutral

pip install flask scikit-learn pandas
python predict_api.py








