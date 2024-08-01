# AI-Powered-Recommendation-Engine
An AI-powered recommendation engine for e-commerce platforms

# AI-Powered Recommendation Engine

## Project Structure
- app.py: Flask application script
- data.csv: Dataset file
- Dockerfile: Docker configuration for building the Flask app
- docker-compose.yml: Docker Compose configuration for setting up multi-container application
- nginx.conf: Nginx configuration file for reverse proxy
- requirements.txt: Python dependencies
- my-prod-recom.ipynb: Jupyter notebook with the implementation

## Setup Instructions
### Prerequisites
- Docker installed
- Docker Compose installed

### Running Locally
1. Clone the repository:
       git clone https://github.com/your-username/AI-Powered-Recommendation-Engine.git
    cd AI-Powered-Recommendation-Engine
    
2. Build and run the Docker containers:
       docker-compose up --build
    
3. Access the application at http://localhost:5000.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
