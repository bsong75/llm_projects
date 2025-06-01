# Types of Chatbots
    1. Natural Chatbot (Gradio)
    2. Chat with PDF (Gradio)
    3. Chat with PDF (Streamlit/ Vector DB)

## Create containers for Llama and Frontend creation with Streamlit, Gradio 
docker-compose up -d
docker-compose up -d --build --force-recreate

## Pull the model, using Ollama
docker exec -it llama-container ollama pull llama3.2
docker exec -it llama-container ollama pull llama3.2:1b
docker exec -it llama-container ollama pull nomic-embed-text
docker exec -it llama-container ollama rm llama3.2:3b