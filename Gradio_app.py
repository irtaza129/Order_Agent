"""
gradio_app.py — Chat UI for the KFC Order Agent
Requires api_service.py to be running on port 8000.

Run:
    # terminal 1
    uvicorn api_service:app --host 127.0.0.1 --port 8000 --reload
    # terminal 2
    python gradio_app.py
"""

import uuid
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8001"


def chat(message: str, history: list, session_id: str) -> str:
    try:
        response = requests.post(
            f"{API_URL}/order",
            json={
                "text":        message,
                "session_id":  session_id,
                "customer_id": "gradio_user",
            },
            timeout=60,
        )
        print(f"[Gradio] status={response.status_code} body={response.text[:300]}")
        data = response.json()
        return data.get("voice_reply", f"No voice_reply in response: {data}")
    except requests.exceptions.ConnectionError:
        return f"Cannot connect to {API_URL} — make sure api_service.py is running."
    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(title="KFC AI Order Agent") as demo:
    gr.Markdown("# 🍗 KFC AI Order Agent")
    gr.Markdown("Place orders, view the menu, or manage items via chat.")

    session_id = gr.State(lambda: str(uuid.uuid4()))

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[session_id],
    )

if __name__ == "__main__":
    demo.launch()