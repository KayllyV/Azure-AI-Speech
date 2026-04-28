**OVERVIEW**
This project is a full end to end voice memo analyzer built using Azure AI services. It takes an audio input (either uploaded or recorded), transcribes it into text, analyzes the meaning, and then converts a summary back into speech. The pipeline is Audio then Speech-to-Text then Language Analysis then Text-to-Speech and lastly JSON + Audio OutputThis project was built using Flask (Python) and deployed to Azure App Service 


ARCHITECTURE 
The system is a 3 stage pipeline. 

Stage 1: Speech to Text takes audio input and converts it into a transcript 
Stage 2: Language Analysis Uses Azure Language to extract, key phrases, sentiment and entites 
Stage 3: Text to Speech Converts a generated summary back into audio Each stage depends on the previous one so the order matters. 

I also added: 
-Telemetry (Application Insights) 
-Custom metrics (latency + confidence) 
-Distributed tracing (pipeline spans) 
-/telemetry-summary endpoint (in memory logging)



SETUP INSTRUCTIONS 
Step 1: Clone the repo git clone <your-repo-link> cd CSC391-AISpeech_Project 
Step 2: Create virtual enviornment python3 -m venv venv source venv/bin/activate To have a seperate workspace from your Python packages to prevent weird errors 
Step 3: Install dependencies pip install -r requirements.txt S
tep 4: Create .env Copy from .env.example and fill in values: 
AZURE_SPEECH_KEY=... 
AZURE_SPEECH_REGION=... 
AZURE_LANGUAGE_KEY=... 
AZURE_LANGUAGE_ENDPOINT=... 
APPLICATIONINSIGHTS_CONNECTION_STRING=... 
Step 5: Run Locally python3 app.py Then open: https://127.0.0.1:5000 -- 

__
CLI COMMANDS 
Login az login Create Resource Group 
    az group create \ 
        --name csc391-speech-rg \ 
        --location westus2 (I originally tried eastus but it was blocked by my subscription, so I switched to westus2 and it worked) 

Create Speech Resource 
    az cognitiveservices account create \ 
        --name csc391-speech \ 
        --resource-group csc391-speech-rg \ 
        --kind SpeechServices \ 
        --sku F0 \
        --location westus2 \ 
        --yes 
        
Create Language Resource 
    az cognitiveservices account create \ 
        --name csc391-language \ 
        --resource-group csc391-speech-rg \ 
        --kind TextAnalytics \ 
        --sku F0 \ 
        --location westus2 \ 
        --yes 

Create Log Analytics Workspace 
    az monitor log-analytics workspace create \ 
        --resource-group csc391-speech-rg \ 
        --workspace-name csc391-logs 
        
Create Application Insights 
    az monitor app-insights component create \ 
        --app csc391-insights \ 
        --location westus2 \ 
        --resource-group csc391-speech-rg \ 
        --workspace csc391-logs 
        
Get Keys 
    az cognitiveservices account keys list \ 
        --name csc391-speech \ 
        --resource-group csc391-speech-rg 


















