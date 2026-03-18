from fastapi import FastAPI, Request
from pydantic import BaseModel
from agent.langgraph_agent import process_order

app = FastAPI()

class OrderRequest(BaseModel):
    raw_input: str
    customer_id: str | None = None

@app.post("/order")
def order_endpoint(order: OrderRequest):
    result = process_order(order.raw_input, order.customer_id)
    return result

@app.get("/")
def root():
    return {"status": "ok", "message": "Order Agent API is running"}
