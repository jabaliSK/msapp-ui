# Use the official vLLM image
FROM vllm/vllm-openai:latest

# Install curl for the health checks in the script
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy the entrypoint script into the container
COPY entrypoint.sh /opt/entrypoint.sh

# Make the script executable
RUN chmod +x /opt/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/opt/entrypoint.sh"]
