import gradio as gr
import requests
import PyPDF2
import io

LLAMA_URL = "http://llama-container:11434/v1/chat/completions"

# Global variable to store PDF content
pdf_content = ""

def extract_pdf_text(pdf_file):
    global pdf_content
    if pdf_file is None:
        return "No PDF uploaded"
    
    try:
        with open(pdf_file, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        pdf_content = text
        return f"PDF loaded successfully! ({len(text)} characters)"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def chat_with_pdf(message, history):
    global pdf_content
    
    # Include PDF content in the system message
    system_message = f"You are a helpful assistant. Answer questions based on this document:\n\n{pdf_content[:4000]}..."
    
    messages = [{"role": "system", "content": system_message}]
    
    for human_msg, ai_msg in history:
        messages.append({"role": "user", "content": human_msg})
        messages.append({"role": "assistant", "content": ai_msg})
    messages.append({"role": "user", "content": message})
    
    payload = {"messages": messages, 
               "max_tokens": 512, 
               "temperature": 0.7, 
               "model": "llama3.2:3b"}
    response = requests.post(LLAMA_URL, json=payload)
    result = response.json()
    bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    
    history.append([message, bot_response])
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# PDF Chat Assistant")
    
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_status = gr.Textbox(label="Status", interactive=False)
    
    pdf_upload.upload(extract_pdf_text, pdf_upload, pdf_status)
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask questions about your PDF...")
    
    msg.submit(chat_with_pdf, [msg, chatbot], [chatbot, msg])

demo.launch(server_name="0.0.0.0", server_port=7860)





# import gradio as gr
# import requests
# import json

# LLAMA_URL = "http://llama-container:11434/v1/chat/completions"


# def chat_with_llama(message, history):
#     messages = []
#     for human_msg, ai_msg in history:
#         messages.append({"role": "user", "content": human_msg})
#         messages.append({"role": "assistant", "content": ai_msg})
#     messages.append({"role": "user", "content": message})
    
#     payload = {"messages": messages, 
#                "max_tokens": 512, #512
#                "temperature": 0.1, 
#                "model": "llama3.2:3b"}
#     response = requests.post(LLAMA_URL, json=payload)
#     result = response.json()
#     bot_response = result.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    
#     history.append([message, bot_response])
#     return history, ""

# with gr.Blocks(title="Kusco", theme="soft", css="footer {display: none ! important}") as demo:
#     chatbot = gr.Chatbot(label="Chat History")
#     gr.Markdown("# Kusco at your service!")
#     msg = gr.Textbox(label="How many I serve you?", 
#                      placeholder="Type your question here.")
    
#     msg.submit(chat_with_llama, [msg, chatbot], [chatbot, msg])

# demo.launch(server_name="0.0.0.0", 
#             server_port=7860,
#             show_api=False)