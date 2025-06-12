#!/usr/bin/env python3
"""
Simple web server that serves the HTML UI and proxies requests to Ray Serve
"""

import os
import requests
from flask import Flask, request, jsonify, send_from_directory

# from flask_cors import CORS

app = Flask(__name__)
# CORS(app)

# Ray Serve endpoint inside cluster
RAY_SERVE_URL = os.getenv("RAY_SERVE_URL", "http://ray-cluster-kuberay-head-svc:8000/")


@app.route("/")
def serve_ui():
    """Serve the HTML UI from file"""
    try:
        with open("C8S1-index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "HTML file not found", 404


@app.route("/predict", methods=["POST"])
def predict_proxy():
    """Proxy prediction requests to Ray Serve"""
    try:
        # Get the JSON payload from the request
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        # Forward the request to Ray Serve
        response = requests.post(RAY_SERVE_URL, json=data, timeout=30, headers={"Content-Type": "application/json"})

        # Return the Ray Serve response
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return (
                jsonify({"error": f"Ray Serve returned status {response.status_code}: {response.text}"}),
                response.status_code,
            )

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to Ray Serve: {str(e)}"}), 503
    except Exception as e:
        return jsonify({"error": f"Proxy error: {str(e)}"}), 500


@app.route("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test connection to Ray Serve
        response = requests.get(f"{RAY_SERVE_URL.rstrip('/')}/health", timeout=5)
        ray_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        ray_status = "unreachable"

    return jsonify({"status": "healthy", "ray_serve_status": ray_status, "ray_serve_url": RAY_SERVE_URL})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    print(f"Starting web server on port {port}")
    print(f"Ray Serve URL: {RAY_SERVE_URL}")

    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "False").lower() == "true")
