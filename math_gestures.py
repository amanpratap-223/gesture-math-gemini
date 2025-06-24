#Steps
#Run the Live Webcam Feed
#Draw the Hand Landmarks
#Create a Canvas
#Draw on the Canvas
#Send the Drawing to the AI Model
#Create a Streamlit Application
#---------------------------------
#Import All the Requried Libraries
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import handTrackingModule as ht
import streamlit as st

st.set_page_config(page_title="Math with Gestures using AI", layout = "wide")
#st.title("Virtual Calculator")
# Custom CSS to style the UI for elegance
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 10px;
    }
    h1 {
        margin-bottom: 0px;
    }
    .header {
        text-align: center;
        margin-top: -50px;  /* Moved title up */
        padding-bottom: 20px;
    }
    .video-feed {
        margin-top: 0px;
    }
    </style>
""", unsafe_allow_html=True)
# Title of the Application moved higher with padding for elegance
st.markdown("<h1 class='header'>Virtual Calculator</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([3,2])

with col1:
    run = st.checkbox('Run', value = True)
    FRAME_WINDOW = st.image([], use_column_width=True)
with col2:
    st.header("Response from AI")
    output_text_area = st.subheader("")


genai.configure(api_key="AIzaSyCvzCrtZa29E8HVF0To4qqYwqWZ5NpFZkQ")
model = genai.GenerativeModel('gemini-1.5-flash')

#Create a Video Capture Object
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 800)

detector = ht.handDetector()
canvas = None
prev_pos = None

def getHandInfo(frame):
    frame = detector.findHands(frame)
    hands, frame = detector.findPosition(frame, draw=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0,1,0,0,0]:
        current_pos = lmList[8][1:3]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1,0,0,0,0]:
        canvas = np.zeros_like(frame)
    return current_pos, canvas

def sendtoAI(model, canvas, fingers):
    if fingers == [0,1,1,1,1]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem and provide a detailed answer", pil_image])
        return response.text
response = ""
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if ret:
        info = getHandInfo(frame)
        if canvas is None:
            canvas = np.zeros_like(frame)
        if info:
            fingers, lmList = info
            #print(fingers)
            #print(lmList)
            prev_pos, canvas = draw(info, prev_pos, canvas)
            response = sendtoAI(model, canvas, fingers)
            if response:
                print("The response from AI Model", response)
        frame_combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        FRAME_WINDOW.image(frame_combined, channels = "BGR")
        if response:
            output_text_area.text(response)
        #cv2.imshow("Live Webcam", frame_combined)
        #cv2.imshow("Canvas", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

