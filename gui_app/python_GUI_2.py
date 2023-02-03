import time
import cv2
import PySimpleGUI as sg
import time
import winsound

# Load the CNN model
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np

# Function for time
def time_as_int():
    return int(round(time.time() * 100))

# Prepare the image
def prepare(imagePrepare):
    test_image = img_to_array(imagePrepare).astype(np.float32)
    test_image /= 255
    test_image = np.expand_dims(test_image, axis=0)
    return test_image

# Camera Settings
camera_Width  = 420 # 480 # 640 # 1024 # 1280
camera_Heigth = 340 # 320 # 480 # 780  # 960
frameSize = (camera_Width, camera_Heigth)
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
final_model = load_model('weights5.hdf5', compile=True)
time.sleep(2.0)

# init Windows Manager
sg.theme("dark green 7")
# sg.theme("DarkBlue")

# font
font_title = ("Arial", 18, "bold")
font_result = ("Arial", 12, "bold")

# def webcam col
colwebcam1_layout = [
    [sg.Text("Camera View", size=(40,1), justification="center")],
    [sg.Image(filename="", key="cam1")]
]

colwebcam2_layout = [
    [sg.Text("Camera View GrayScale", size=(40,1), justification="center")],
    [sg.Image(filename="", key="cam1gray")]
]

textLocation = [sg.Text("Driver Drowsiness Detection Model Testing", size=(40,1),  font=font_title, text_color="green", justification="center")]

resultText1 = [
    [sg.Text("Processed Data:", size=(20,1), font=font_result, text_color="orange")],
    [sg.HSeparator()],
    [sg.Text("Fps: ", size=(20,1), font=font_result)],
    [sg.Text("Eyes Closed Per Minute: ", size=(20,1), font=font_result)]
]

resultGraph1 = [
    [sg.Text("LIVE GRAPH HERE", size=(20,1), font=font_title, text_color="orange")]
]

webcamLayout = [
    [sg.Frame(layout=colwebcam1_layout, title='')],
    [sg.Frame(layout=colwebcam2_layout, title='')]          
]

startButton = [
    [sg.Button('Start', size=(7,1), font='Helvetica 14', key='-START-'),
     sg.Button('Stop', size=(7,1), font='Any 14', key='-STOP-'),
     sg.Button('Exit', size=(7,1), font='Helvetica 14')]
]

timeStamp = [
    [sg.Text("TIME STAMP")],
    [sg.HSeparator()],
    [sg.Frame(layout=[[sg.Text("", size=(40,2), font=("Arial", 40, "bold"), text_color="green", key="text", justification="center")]], title='')]
]

eyeCount = [
    [sg.Text("EYE COUNT", size=(20,1))],
    [sg.HSeparator()],
    [sg.Text("", font=("Arial", 120, "bold"), text_color="green", justification="center", key="count")]
]

alarmRaise = [
    [sg.Text("", size=(25,3), font=("Arial", 25, "bold"), text_color="white", justification="center", key="alarm")]
]

eyeCountLayout = [
    [sg.Frame(layout=startButton, title='', size=(280,50), element_justification="c")],
    [sg.Frame(layout=timeStamp, title='', size=(280,200), element_justification="c")],
    [sg.Frame(layout=eyeCount, title='', size=(280,300), element_justification="c")],
    [sg.Frame(layout=alarmRaise, title='', size=(280,50), element_justification="c")]
]

resultsLayout = [
    [sg.Text("Here the results shall be displayed", size=(40,8), justification="top")],
    [sg.Frame(layout=resultText1, title='')],
    [sg.Frame(layout=resultGraph1, title='')]
]

layout = [
    [
        textLocation
    ],
    [
        sg.Column(webcamLayout),
        sg.VSeparator(),        
        sg.Column(eyeCountLayout)
        # sg.VSeperator(),
        # sg.Column(resultsLayout)
    ]
]


window = sg.Window("Driver Drowsiness Detection", layout,
                    no_titlebar=False, alpha_channel=1, grab_anywhere=False, 
                    return_keyboard_events=True, location=(100, 100))
current_time, currentframe, paused, recording, closed_time, closed_eyecount,open_eyecount, raise_alarm = 0, 1, False, False, 0, 0,0, False
# start_time = time_as_int()

while True:
    event, values = window.read(timeout=20)
    pred = 0
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
        
    if event == '-START-':
        recording = True
        start_time = time_as_int()
        
    if event == '-STOP-':
        recording = False

    if event == 'Exit' or event == sg.WIN_CLOSED:
        break

    # get camera frame
    if recording:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, frameSize)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        current_time = time_as_int() - start_time
        
                
        # Reset the value here
        if current_time >= 6000:
            start_time = time_as_int()
            if closed_time < 7:
                raise_alarm = True
                # winsound.PlaySound("sound/beep.wav",winsound.SND_FILENAME|winsound.SND_ASYNC)
            closed_time = 0
            current_time = 0
        
        # Reset the alarm after 5 seconds
        if current_time >= 500 and raise_alarm == True:
            raise_alarm = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y+w, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                roi = roi_color[ey+4:ey + ew - 3, ex+4:ex + eh - 3]
                image_resized = cv2.resize(roi, (150,150))
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)
                # cv2.imwrite('imageSave/frame' + str(currentframe) + '.png', image_resized)
                result = final_model.predict(prepare(image_resized))
                pred = (result[0][0]> 0.5).astype("int32")
                currentframe += 1
        
        if pred == 0:
            fps = "Closed"
            closed_eyecount += 1
            if closed_eyecount == 5:
                closed_time += 1
                open_eyecount = 0
        else:
            fps = "Open"
            open_eyecount += 1
            print(open_eyecount)
            if open_eyecount >= 8:
                closed_eyecount = 0
        
        # cv2.putText(frame, str(fps), (10, camera_Heigth - 10), font, 2, (0, 255, 0), 5, cv2.LINE_AA)
        # cv2.imshow('frame', frame)
        # # update webcam1
        imgbytes = cv2.imencode(".png", frame)[1].tobytes()
        window["cam1"].update(data=imgbytes)
    
        # # transform frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # update webcam2
        imgbytes = cv2.imencode(".png", gray)[1].tobytes()
        window["cam1gray"].update(data=imgbytes)
    
        # --------- Display timer in window --------
        window["text"].update('{:02d}:{:02d}.{:02d}'.format((current_time // 100) // 60,
                                                            (current_time // 100) % 60,
                                                             current_time % 100))
        
        # -------- Display counter in window ------
        window["count"].update(closed_time)
        
        # print(current_time, raise_alarm)
        # -------- Display Alarm in window --------
        if raise_alarm==True:
            window["alarm"].update("ALARM", background_color="red")
        else:
            window["alarm"].update("", background_color="#0C231E")

video_capture.release()
cv2.destroyAllWindows()