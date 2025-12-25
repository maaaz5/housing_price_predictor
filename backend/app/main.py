from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

app = FastAPI(
  title="Housing Price Predictor API",
  description="API for predicting house prices using a custom linear regression model created by me",
  version="1.0"
)


app.add_middleware(
  CORSMiddleware,
  allow_origins=['https:localhost:3000'],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

#api routes
app.include_router(router, prefix="api", tags=['api'])


#root endp
@app.get("/")
def root_func():
  return {
    "message":"Housing Price Predictor API is runinngngnngngngngnngngn",
    "status":'we running'
  }


@app.get("/health")
def root_func():
  return {
    "status":'we doing okay bud!'
  }