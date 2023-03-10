import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
from matplotlib.pyplot import title
from flask import render_template, request, Response, redirect, url_for
from werkzeug.utils import redirect
from app import app, APP_ROOT
import os
import json
import requests
from datetime import datetime
import cv2
from threading import Thread
from app.predictions import spiral
from uuid import uuid4
import glob
from PIL import Image 

config = {
    # "apiKey": "AIzaSyD9jO7T-GKI3zR7E623p0xqKzjy5a1xvVs",
    # "authDomain": "ezpd-d826d.firebaseapp.com",
    # "projectId": "ezpd-d826d",
    # "storageBucket": "ezpd-d826d.appspot.com",
    # "messagingSenderId": "81687716578",
    # "appId": "1:81687716578:web:b035a0e321e58da7baf4bb"


    "apiKey": "AIzaSyCEOCMOEC7xQ87nBbTsD7ZXU_3WB07cieA",
    "authDomain": "lymelyte-656c9.firebaseapp.com",
    "projectId": "lymelyte-656c9",
    "storageBucket": "lymelyte-656c9.appspot.com",
    "messagingSenderId": "268650470710",
    "appId": "1:268650470710:web:4793c70ccb708971e0e797",
}


cred = credentials.Certificate(os.path.join(
    APP_ROOT, "lymelyte-656c9-firebase-adminsdk-4r6i3-2e226d921a.json"))
app_fb = firebase_admin.initialize_app(cred)

db = firestore.client()
bucket = storage.bucket(config["storageBucket"])
person = {"is_logged_in": False, "name": "", "email": "", "uid": ""}

video = cv2.VideoCapture(0)
global rec, out
rec = False
modelInput = ''


@app.route('/')
def home():
    return render_template('index.html', title='Home')


@app.route('/record')
def record():
    if not person["is_logged_in"]:
        return redirect('/login')
    return render_template('record.html', title='Add Image')


# @app.route('/recordspeech')
# def recordspeech():
#     if not person["is_logged_in"]:
#         return redirect('/login')
#     return render_template('recordspeech.html', title='Add Speech Recording')


# @app.route('/recordpose')
# def recordpose():
#     if not person["is_logged_in"]:
#         return redirect('/login')
#     return render_template('recordpose.html', title='Add Pose Video')


@app.route('/dashboard')
def dashboard():
    class Item:
      def __init__(self, vals):
        self.__dict__ = vals
    
    if not person["is_logged_in"]:
        return redirect("/login")
    
    try:
        tests = db.collection('users').document(
            person["uid"]).get().to_dict()['tests']
    except:
        return render_template('calendar.html', title='Dashboard', name=person['name'].split(' ')[0], message='Record data first before viewing the data dashboard!')    
        
    if not tests: return render_template('calendar.html', title='Dashboard', name=person['name'].split(' ')[0], message='Record data first before viewing the data dashboard!')
    # test = tests[-1] if not request.args.get('id') else [x for x in tests if x['id'] == request.args.get('id')][0]
    # dct = {}
    # dct['date'] = test['date'].strftime("%b %d, %Y (%H:%M)")
    # # dct['speech_diag'] = ["No", "Yes"][test['diagnosis']['speech']]
    # # print(test['diagnosis']['spiral'])
    # dct['spiral_diag_positive'] = round(test['diagnosis']['spiral'], 2)
    # dct['spiral_diag_negative'] = round(100 - test['diagnosis']['spiral'], 2)
    # dct['diagnosis'] = 'Positive'
    # if dct['spiral_diag_negative'] > 50:
    #     dct['diagnosis'] = 'Negative'
    # # dct['gait_diag'] = ["No", "Yes"][test['diagnosis']['gait']]
    # dct['spiral'] = person['uid']+"_spiral_"+test['id']+'.png'
    # # dct['amp'] = person['uid']+"_amp_"+test['id']+'.png'
    # # dct['freq'] = person['uid']+"_freq_"+test['id']+'.png'
    # # dct['wave'] = person['uid']+'_wave_'+test['id']+'.png'

    # for k in ['spiral']:
    #     blob = bucket.blob(dct[k])
    #     blob.make_public()
    #     dct[k] = blob.public_url
    
    # options = [{'url': url_for('dashboard', id=t['id']), 'date': t['date'].strftime("%b %d, %Y (%H:%M)")} for t in tests]
    # options = [Item(o) for o in options][::-1]

    return render_template('calendar.html', title='Dashboard', name=person['name'].split(' ')[0])


@app.route('/login')
def login():
    if person["is_logged_in"] == False:
        return render_template('login.html', title='Log In')
    return redirect("/dashboard")


@app.route('/signup')
def signup():
    if person["is_logged_in"] == False:
        return render_template('signup.html', title="Sign Up")
    return redirect("/dashboard")


@app.route('/signin', methods=["GET", "POST"])
def sign_in():
    if request.method == "POST":  # Only if data has been posted
        result = request.form  # Get the data
        email = result["email"]
        password = result["pass"]
        try:
            # Try signing in the user with the given information
            payload = json.dumps({
                "email": email,
                "password": password,
                "returnSecureToken": True
            })

            r = requests.post("https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword",
                              params={"key": config["apiKey"]},
                              data=payload)

            user = r.json()            # Insert the user data in the global person
            if "error" not in user:
                user = auth.get_user_by_email(email)
                global person
                person["is_logged_in"] = True
                person["email"] = user.email
                person["uid"] = user.uid
                # Get the name of the user
                doc = db.collection('users').document(person["uid"]).get()
                person["name"] = doc.to_dict()["name"]
                # Redirect to welcome page
                print(person)
                return redirect('/dashboard')
            else:
                return redirect('/login')
        except:
            # If there is any error, redirect back to login
            return redirect('/login')
    else:
        if person["is_logged_in"] == True:
            return redirect('/calendarAfter')
        else:
            return redirect('/login')
        
@app.route("/form")
def form():
    if person["is_logged_in"] == True:
        return render_template('form.html', title='Form')
    else:
        return redirect('/dashboard')


@app.route("/register", methods=["POST", "GET"])
def register():
    if request.method == "POST":  # Only listen to POST
        result = request.form  # Get the data submitted
        print(result)
        email = result["email"]
        password = result["pass"]
        name = result["name"]
        try:
            # Try creating the user account using the provided data
            user = auth.create_user(email=email, password=password)
            # Add data to global person
            global person
            person["is_logged_in"] = True
            person["email"] = email
            person["uid"] = user.uid
            person["name"] = name
            # Append data to the firebase realtime database
            data = {"name": name, "email": email, 'tests': []}
            db.collection('users').document(person["uid"]).set(data)
            # Go to welcome page
            print(person)
            return redirect('/form')
        except:
            # If there is any error, redirect to register
            return redirect("/signup")

    else:
        if person["is_logged_in"] == True:
            return redirect("/form")
        else:
            return redirect("/signup")


@app.route("/uploadspiral", methods=["GET", "POST"])
def uploadspiral():
    target = os.path.join(APP_ROOT, 'temp')
    for f in os.listdir(target):
        os.remove(os.path.join(target, f))
    if request.method == 'POST':
        # 'img' is the id passed in input file form field
        file = request.files['img']
        global modelInput 
        if file.filename != '':
            # saving file in temp folder
            file.save(os.path.join(target, 'spiral.' +
                                   file.filename.split('.')[-1]))
            modelInput = str(os.path.join(target, 'spiral.' +
                                   file.filename.split('.')[-1]))
            file = Image.open(modelInput)
            file.save(modelInput[:-4] + '.png')
        else:
            _, frame = video.read()
            try:
                cv2.imwrite(os.path.join(target, 'spiral.png'), frame)
            except:
                cv2.imwrite(os.path.join(target, 'spiral.png'), frame)
        print("Spiral Upload Completed")  # printing on terminal
        return redirect('/wait')


# @app.route("/uploadspeech", methods=["GET", "POST"])
# def uploadspeech():
#     target = os.path.join(APP_ROOT, 'temp')
#     if request.method == 'POST':
#         print(request.files)
#         # 'img' is the id passed in input file form field
#         file = request.files['file']
#         # saving file in temp folder
#         file.save(os.path.join(target, 'speech.wav'))
#         print("Speech Upload Completed")  # printing on terminal
#         return redirect('/recordpose')


# @app.route("/uploadvideo", methods=["GET", "POST"])
# def uploadvideo():
#     target = os.path.join(APP_ROOT, 'temp')
#     if request.method == 'POST':
#         print(request.files)
#         # 'img' is the id passed in input file form field
#         file = request.files['file']
#         # saving file in temp folder
#         file.save(os.path.join(target, 'pose.mp4'))
#         print("Video Upload Completed")  # printing on terminal
#         return redirect('/predictions')
@app.route('/signout')
def signout():
    if person["is_logged_in"] == True:
        person["is_logged_in"] = False
        print("User signed out...")
    return redirect('/')

@app.route('/food')
def food():
    return render_template('food.html', title='Food')

@app.route('/food1')
def food1():
    return render_template('food1.html', title='Food')

@app.route('/wait')
def wait():
    return render_template('prediction.html', title='Predicting...')

@app.route('/calendarAfter')
def calendarAfter():
    return render_template('calendarAfter.html', title='Calendar')

@app.route('/predictions')
def predictions():
    # temp_path = os.path.join(APP_ROOT, 'temp')
    # # spiral_pred = spiral(modelInput)
    # spiral_pred = spiral()
    # # speech_pred = speech()
    # # gait_pred = 0 # gait()
    # u = uuid4()
    # print('-----> got preds')
    # endType = spiral_pred[1]
    # blob_spiral = bucket.blob(f'{person["uid"]}_spiral_{u}.png') #endType[-3:]
    # blob_spiral.upload_from_filename(os.path.join(temp_path, endType[-10:])) #endType[-10:]

    # blob_amp = bucket.blob(f'{person["uid"]}_amp_{u}.png')
    # blob_amp.upload_from_filename(os.path.join(temp_path, 'amplitude.png'))

    # blob_freq = bucket.blob(f'{person["uid"]}_freq_{u}.png')
    # blob_freq.upload_from_filename(os.path.join(temp_path, 'frequency.png'))

    # blob_wave = bucket.blob(f'{person["uid"]}_wave_{u}.png')
    # blob_wave.upload_from_filename(os.path.join(temp_path, 'wave.png'))

    # blob_speech = bucket.blob(f'{person["uid"]}_speech_{u}.wav')
    # blob_speech.upload_from_filename(
    #     glob.glob(os.path.normpath(temp_path)+"/*.wav")[0])
    
    # blob_gait = bucket.blob(f'{person["uid"]}_gait_{u}.avi')
    # blob_gait.upload_from_filename(os.path.join(temp_path, 'gait.avi'))

    # print('-----> uploaded stuff to storage')

    # doc = db.collection('users').document(person["uid"]).get().to_dict()
    # updated = doc['tests']
    # updated.append({'id': str(u), 'diagnosis': {'spiral': float(spiral_pred[0])}, 'date': datetime.now()}) #[0, 1][spiral_pred == 1]
    # db.collection('users').document(person["uid"]).update({"tests": updated})

    # print('-----> updated firestore')
    # print(float(spiral_pred[0]))

    return redirect('/calendarAfter')
# @app.route("/faceresults/",methods=["GET","POST"])
# def prediction():
#     #imported process.py
#     x=predict_img() #imported from process file
#     out = ''
#     if x == -1: out = 'Error: no face/eyes detected. Please conduct the test again.'
#     elif x == 0: out = 'No facial paralysis detected.'
#     else: out = 'Facial paralysis detected.'
#     return render_template('faceresults.html',results=out)


def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = video.read()
        cv2.imwrite('t.jpg', frame)
        output = (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
        os.remove('t.jpg')
        yield output


@app.route('/video_rec', methods=["POST", "GET"])
def video_rec():
    global rec, out
    rec = not rec
    if(rec):
        now = datetime.datetime.now()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(
            APP_ROOT, 'temp/vid_{}.avi'.format(str(now).replace(":", ''))), fourcc, 20.0, (640, 480))
        # Start new thread for recording the video
        thread = Thread(target=record, args=[out, ])
        thread.start()
    elif(rec == False):
        out.release()
#


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
