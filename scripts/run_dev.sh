#!/usr/bin/env bash
set -e
echo "Starting CopMap prototype (uvicorn)"
uvicorn app.main:app --reload --port 8000
