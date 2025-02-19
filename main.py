from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import base64
import queue
import os
import uvicorn

app = FastAPI()

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load nail images
nail_images = [
    cv2.imread(os.path.join(BASE_DIR, "static", "1.png"), cv2.IMREAD_UNCHANGED),
    cv2.imread(os.path.join(BASE_DIR, "static", "7.png"), cv2.IMREAD_UNCHANGED),
    cv2.imread(os.path.join(BASE_DIR, "static", "3.png"), cv2.IMREAD_UNCHANGED),
]

# Function to overlay an image
def overlay_image(bg_img, fg_img, x, y):
    fg_h, fg_w, fg_c = fg_img.shape
    bg_h, bg_w, bg_c = bg_img.shape

    if x + fg_h > bg_h or y + fg_w > bg_w or x < 0 or y < 0:
        return bg_img

    fg_b, fg_g, fg_r, fg_a = cv2.split(fg_img)
    fg_rgb = cv2.merge((fg_b, fg_g, fg_r))
    alpha = fg_a / 255.0

    roi = bg_img[x:x + fg_h, y:y + fg_w]
    blended = (1 - alpha)[:, :, None] * roi + alpha[:, :, None] * fg_rgb
    bg_img[x:x + fg_h, y:y + fg_w] = blended.astype(np.uint8)
    return bg_img

frame_queue = queue.Queue(maxsize=1)  # Giữ tối đa 1 khung hình

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_nail_index = 0

    while True:
        # Nhận dữ liệu từ frontend
        data = await websocket.receive_text()
        img_data = base64.b64decode(data)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Giảm độ phân giải để tăng tốc
        img_resized = cv2.resize(img, (320, 240))
        imgRGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                h, w, c = img_resized.shape
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [4, 8, 12, 16, 20]:  # Fingertip IDs
                        nail_resized = cv2.resize(nail_images[current_nail_index], (40, 40))
                        img_resized = overlay_image(img_resized, nail_resized, cy - 20, cx - 20)

        # Encode kết quả và gửi lại cho frontend
        _, buffer = cv2.imencode('.jpg', img_resized)
        encoded_img = base64.b64encode(buffer).decode('utf-8')
        await websocket.send_text(encoded_img)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)

# import cv2
# import mediapipe as mp
# import time
# import numpy as np

# cap = cv2.VideoCapture(0)

# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils

# # Load the nail image (Ensure it has transparency)
# nail_img = cv2.imread("1.png", cv2.IMREAD_UNCHANGED)

# def overlay_image(bg_img, fg_img, x, y):
#     """Efficiently overlay fg_img on bg_img at position (x, y)."""
#     fg_h, fg_w, fg_c = fg_img.shape
#     bg_h, bg_w, bg_c = bg_img.shape

#     # Check bounds
#     if x + fg_h > bg_h or y + fg_w > bg_w or x < 0 or y < 0:
#         return bg_img

#     # Extract channels
#     fg_b, fg_g, fg_r, fg_a = cv2.split(fg_img)
#     fg_rgb = cv2.merge((fg_b, fg_g, fg_r))
#     alpha = fg_a / 255.0

#     # Get ROI
#     roi = bg_img[x:x + fg_h, y:y + fg_w]
#     blended = (1 - alpha)[:, :, None] * roi + alpha[:, :, None] * fg_rgb
#     bg_img[x:x + fg_h, y:y + fg_w] = blended.astype(np.uint8)
#     return bg_img

# pTime = 0
# cTime = 0

# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)  # Flip for mirrored view
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             # Calculate distances to adjust nail size dynamically
#             h, w, c = img.shape
#             cx_thumb, cy_thumb = int(handLms.landmark[4].x * w), int(handLms.landmark[4].y * h)
#             cx_index, cy_index = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)
#             distance = int(((cx_thumb - cx_index) ** 2 + (cy_thumb - cy_index) ** 2) ** 0.5)

#             # Limit nail size
#             nail_size = max(20, min(distance, 40))  # Minimum 20px, maximum 40px

#             for id, lm in enumerate(handLms.landmark):
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 if id in [4, 8, 12, 16, 20]:  # Fingertip IDs
#                     nail_resized = cv2.resize(nail_img, (nail_size, nail_size))  # Scale based on limited size
#                     img = overlay_image(img, nail_resized, cy - nail_size // 2, cx - nail_size // 2)

#             # Optional: Draw hand landmarks for debugging
#             # mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

#     # Calculate FPS
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime

#     # Add FPS text
#     cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

#     # Display the result
#     cv2.imshow("Virtual Nail Art", img)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
