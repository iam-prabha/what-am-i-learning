# Docker

<!--toc:start-->

- [Docker](#docker)
  - [What is Docker?](#what-is-docker)
  - [What is a Container?](#what-is-a-container)
  - [What are Images?](#what-are-images)

<!--toc:end-->

## What is Docker?

- Docker is a platform that helps us build, run and share software applications inside containers.
- Docker uses containers to create isolated environments where programs run.

## What is a Container?

- Containers use the same hardware as our operating system (CPU, RAM, etc.).
- However, they are separated entities that operate in a sandbox environment.
- Therefore all software requirements are pre-installed inside the container.
- Imagine we are working on a coding project with our team (each with a different OS).

## What are Images?

- Docker images are similar to GitHub repositories.
- But instead of software - they store environments where software is installed.
- Docker images are read-only format.
- We build new containers from images, but we cannot change the existing ones.
- We can think of them as a static set of instructions. We use them to create containers.
- Containers are running instances of images.
- Unlike images, we can change and interact with them.
