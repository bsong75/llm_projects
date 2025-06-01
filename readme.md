## Create containers for Llama and Frontend creation with Gradio
docker-compose up -d

## Pull the model, using Ollama
docker exec -it llama-container ollama run llama2

docker-compose up -d --build --force-recreate