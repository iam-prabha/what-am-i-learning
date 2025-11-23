# Docker

Docker learning resources and notes for containerization.

## 📚 Contents

### Core Concepts

#### What is Docker?

- Docker is a platform that helps us build, run and share software applications inside containers.
- Docker uses containers to create isolated environments where programs run.
- It provides a consistent environment across different machines and operating systems.

#### What is a Container?

- Containers use the same hardware as our operating system (CPU, RAM, etc.).
- However, they are separated entities that operate in a sandbox environment.
- All software requirements are pre-installed inside the container.
- Perfect for team collaboration where each member has a different OS.
- Containers are lightweight and start quickly compared to virtual machines.

#### What are Images?

- Docker images are similar to GitHub repositories.
- But instead of software - they store environments where software is installed.
- Docker images are read-only format.
- We build new containers from images, but we cannot change the existing ones.
- We can think of them as a static set of instructions. We use them to create containers.
- Containers are running instances of images.
- Unlike images, we can change and interact with them.

## 🚀 Common Docker Commands

### Image Management
```bash
# Pull an image from Docker Hub
docker pull image_name:tag

# List all images
docker images

# Remove an image
docker rmi image_name

# Build an image from Dockerfile
docker build -t image_name:tag .
```

### Container Management
```bash
# Run a container
docker run image_name

# Run a container in detached mode
docker run -d image_name

# Run with port mapping
docker run -p host_port:container_port image_name

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop container_id

# Remove a container
docker rm container_id

# Execute command in running container
docker exec -it container_id command
```

### Docker Compose
```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs
```

## 📝 Dockerfile Basics

A Dockerfile is a text file that contains instructions for building a Docker image:

```dockerfile
# Use a base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run command
CMD ["python", "app.py"]
```

## 💡 Use Cases

- **Development Environment** - Consistent setup across team members
- **Application Deployment** - Package applications with dependencies
- **CI/CD Pipelines** - Automated testing and deployment
- **Microservices** - Isolated service containers
- **Data Science** - Reproducible research environments

## 🔗 Related Content

- Python Projects: `../python/`
- Data Science Projects: `../projects/`
- Linux Commands: `../linux/`

## 📚 Additional Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
