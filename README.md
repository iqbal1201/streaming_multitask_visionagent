# Streaming Vision Agent

This guide provides instructions for setting up and running a streaming vision agent using the Agent Development Kit (ADK).

## 1. Setup Environment & Install ADK

It is highly recommended to use a Python virtual environment for this project.

### Create and Activate Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv
```

### Activate the virtual environment

```bash
# Create a virtual environment
source .venv/bin/activate
```

### Install Libraries

```bash
# install libraries
pip install -r requirements.txt
```

## 2. Project Structure
Create the following folder and file structure:
```
directory_folder/               # Project folder
├── app/                     # Web app folder
│   └── .env                 # Gemini API key file
└── vision_agent/     # Agent logic folder
    ├── __init__.py          # Python package init
    └── agent.py
    └── prompts.py            # Agent definition
```

## 3. Set Up the Platform
Choose one of the following platforms :
- Google AI Studio (recommended for local development)
- Google Cloud Vertex AI (recommended one)

Generate your API Key.

Create a .env file inside the app/ folder with the following content:
```bash
# Create .env file
GOOGLE_GENAI_USE_VERTEXAI="TRUE"
GOOGLE_CLOUD_PROJECT="xxxxxx"
GOOGLE_CLOUD_LOCATION="us-central1"
GOOGLE_API_KEY=xxxxxxxxxxxxxx
RAG_CORPUS='projects/xxxxxxxxx/locations/us-central1/ragCorpora/2305843009213693952'
```


## 4. Run Agent
Now you're ready to try the agent.

Navigate to the App Directory
First, change your current directory to the app/ folder.

```bash
# change directories
cd app
```

Set SSL_CERT_FILE (Required for Voice & Video)
This step is necessary for enabling voice and video features. (Running in WSL/Linux is recommended)

```bash
# change directories
export SSL_CERT_FILE=$(python -m certifi)
```

Launch the Dev UI
Finally, run the adk web command to start the development user interface.

```bash
# launch web
adk web
```





# Deploying vision-agent to Google Cloud Run

This guide explains how to build, push, and deploy the `vision-agent` service to **Google Cloud Run** using **Artifact Registry**.

---


## 1. Make sure you have .env file in app folder
This agent will use Vertex AI service so that we need to make sure GOOGLE_GENAI_USE_VERTEXAI = True

```bash
GOOGLE_GENAI_USE_VERTEXAI="TRUE"
GOOGLE_CLOUD_PROJECT="xxxxxxx"
GOOGLE_CLOUD_LOCATION="us-central1"

```

## 2. Change the directory

```bash
cd aisee-adk/app

```

## 3. Push Image to Artifact Registry

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/xxxxxx/aisee-tridorian/vision-agent .

```


## 4. Build Cloud Run

```bash
gcloud run deploy vision-agent \
  --image us-central1-docker.pkg.dev/xxxxxxx/aisee-tridorian/vision-agent \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated 

```

