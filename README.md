# Retrieval-Augmented Image Search

A FastAPI-based REST API for visual product search using DINOv2 embeddings and cross-encoder ranking.

## Overview

This project implements a scalable visual search system that allows users to:
- Search for products using image uploads
- Refine results using text queries
- Manage product images and their vector embeddings
- Support multi-tenant architecture

## Tech Stack

- **Framework**: FastAPI
- **Image Embedding**: Facebook DINOv2 Base
- **Text Reranking**: MiniLM Cross-Encoder
- **Vector Database**: Qdrant
- **Database**: MySQL
- **Authentication**: API Key-based

## Features

- Visual similarity search using DINOv2 embeddings
- Text-based reranking of results using cross-encoder
- Multi-tenant support with configuration management
- Async image processing and batch operations
- RESTful API with proper error handling
- Health check endpoints

## API Endpoints

### Search
- `POST /search_by_image/` - Search products using an image upload

### Product Management
- `POST /add_product/` - Add a new product with images
- `DELETE /delete_products/` - Delete a product and its embeddings
- `POST /add_new_images/` - Add new images to existing product
- `DELETE /delete_image/` - Delete specific image from a product

### System
- `GET /` - Health check endpoint

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
