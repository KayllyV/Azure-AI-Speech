#!/bin/bash
apt-get update && apt-get install -y ffmpeg
gunicorn --bind=0.0.0.0 --timeout 600 app:app
