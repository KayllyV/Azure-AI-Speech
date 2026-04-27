import os
from azure.monitor.opentelemetry import configure_azure_monitor

def init_telemetry():
    connection_string = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    
    if not connection_string:
        print("No Application Insights connection string found — skipping telemetry")
        return
    
    try:
        configure_azure_monitor(connection_string=connection_string)
        print("Application Insights initialized.")
    except Exception as e:
        print("Telemetry failed:", e)