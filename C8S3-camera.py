#!/usr/bin/env python3
"""
Camera-based finger counting using the MLOps workshop model (continuous mode)
Usage: python camera_finger_counter.py [--endpoint URL] [--interval SECONDS]
"""

import argparse
import base64
import io
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


def preprocess_for_hand_detection(image):
    """Convert colored image to binary-like format using only PIL (same as Ray Serve)"""
    # Convert to grayscale
    gray = image.convert("L")

    # Enhance contrast to make hand stand out
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(3.0)  # Increase contrast

    # Apply slight blur to reduce noise
    blurred = enhanced.filter(ImageFilter.GaussianBlur(radius=1))

    blurred = ImageOps.invert(blurred)
    return blurred.convert("RGB")

    # Simple thresholding - convert to binary
    threshold = 128  # Lower = more permissive for darker areas
    binary = blurred.point(lambda x: 255 if x < threshold else 0, mode="L")

    # Convert back to RGB (3 channels) for model compatibility
    rgb_binary = Image.merge("RGB", (binary, binary, binary))

    return rgb_binary


def setup_camera():
    """Initialize camera once"""
    print("📷 Opening camera...")

    # Try different camera indices (0, 1, 2) in case default doesn't work
    cap = None
    for camera_id in range(3):
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            print(f"✅ Camera {camera_id} opened successfully")
            break
        cap.release()

    if not cap or not cap.isOpened():
        raise RuntimeError("❌ Could not open camera. Check if camera is connected and not used by another app.")

    return cap


def capture_and_display(cap):
    """Capture frame and display both original and processed versions"""
    ret, frame = cap.read()
    if not ret:
        return None

    # Convert BGR to RGB for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Apply preprocessing
    processed_pil = preprocess_for_hand_detection(pil_image)

    # Convert processed image back to OpenCV format for display
    processed_array = np.array(processed_pil)
    processed_bgr = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)

    # Display both images
    # cv2.imshow("Original Camera Feed", frame)
    cv2.imshow("Processed for Model (White Hand on Black)", processed_bgr)

    # Check for ESC key to quit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        return None

    return pil_image


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def predict_fingers(image_b64, endpoint):
    """Send image to API and get prediction"""
    print(f"🔮 Sending image to {endpoint}...")

    payload = {"image": image_b64}

    try:
        response = requests.post(endpoint, json=payload, timeout=30, headers={"Content-Type": "application/json"})

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API returned status {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to connect to API: {str(e)}")


def print_result(result):
    """Print the prediction result in compact format"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    finger_emojis = ["✊", "☝️", "✌️", "🤟", "🖖", "🖐️"]
    finger_emoji = finger_emojis[result.get("prediction", 0)] if result.get("prediction", 0) < 6 else "❓"

    print(
        f"🤖 Result: {finger_emoji} {result.get('prediction', 'Unknown')} fingers | "
        f"Confidence: {result.get('confidence', 0) * 100:.1f}% | "
        f"Model: {result.get('model_type', 'Unknown')}"
    )

    if "all_probabilities" in result:
        # Show top 3 probabilities in compact format
        probs_with_idx = [(i, prob) for i, prob in enumerate(result["all_probabilities"])]
        probs_with_idx.sort(key=lambda x: x[1], reverse=True)

        top3_str = " | ".join([f"{finger_emojis[i]}{i}:{prob * 100:.0f}%" for i, prob in probs_with_idx[:3]])
        print(f"📊 Top predictions: {top3_str}")


def main():
    parser = argparse.ArgumentParser(description="Camera-based finger counting (continuous mode)")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080/predict",
        help="API endpoint URL (default: http://localhost:8080/predict)",
    )
    parser.add_argument(
        "--interval", type=float, default=5.0, help="Sleep interval between captures in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    print("🖐️ Camera Finger Counter (Continuous Mode)")
    print(f"API Endpoint: {args.endpoint}")
    print(f"Capture interval: {args.interval}s")
    print("Press Ctrl+C to stop")
    print("-" * 40)

    cap = None
    try:
        # Initialize camera once
        cap = setup_camera()

        capture_count = 0
        while True:
            capture_count += 1
            # Capture image
            image = capture_and_display(cap)
            if image is None:
                print("❌ Failed to capture image")
                time.sleep(args.interval)
                continue

            if capture_count % 100:
                continue  # slow down analysis

            print(f"\n📸 Capture #{capture_count}")

            # Convert to base64
            image_b64 = image_to_base64(image)

            # Get prediction
            try:
                result = predict_fingers(image_b64, args.endpoint)
                print_result(result)
            except Exception as e:
                print(f"❌ Prediction failed: {e}")

            # Wait before next capture
            print(f"⏱️ Waiting {args.interval}s...")
            # time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Tips:")
        print("- Make sure the web service is running: kubectl port-forward svc/finger-counting-ui-svc 8080:8080")
        print("- Check if camera is available and not used by another application")
    finally:
        if cap:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
