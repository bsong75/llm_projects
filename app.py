import gradio as gr
import requests

LLAMA_URL = "http://llama-container:11434/v1/chat/completions"

def chat_with_llama(message, history):
    messages = []
    for human_msg, ai_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})
    
    payload = {"messages": messages, 
               "max_tokens": 200, #512
               "temperature": 0.7, 
               "model": "llama2"}
    response = requests.post(LLAMA_URL, json=payload)
    result = response.json()
    bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    
    history.append([message, bot_response])
    return history, ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Type message...")
    
    msg.submit(chat_with_llama, [msg, chatbot], [chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860)