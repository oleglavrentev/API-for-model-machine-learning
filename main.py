import uvicorn
from fastapi import FastAPI
from model import Assist
import words


app = FastAPI()
model = Assist()

@app.get('/')
def dataset():
    return words.data_set

@app.get('/recognize')
def recognize():
    data = "расскажи про общежития"
    predict = model.recognize(data)
    return {data: predict}

if __name__=='__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)


#uvicorn main:app --reload