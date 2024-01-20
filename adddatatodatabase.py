import os
import pickle
import numpy as np
import face_recognition
import firebase_admin
import cv2
from firebase_admin import storage
from firebase_admin import db

cred=firebase_admin.credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':"https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/"
})
ref=db.reference('Students')
data={
    "16313":
        {
            "name":"amruth",
            "major":"computer science",
            "starting_year":2019,
            "total_attendance":6,
            "standing":"G",
            "year":4,
            "last_attendance_time":"2024-01-19 00:54:34"

        },
    "16314":
        {
            "name":"elon",
            "major":"Rocket scientist",
            "starting_year": 2015,
            "total_attendance": 6,
            "standing": "B",
            "year": 3,
            "last_attendance_time": "2024-01-19 00:54:34"
        },
    "16315":
        {
                "name":"ninja",
                   "major":"Aerospace",
                    "starting_year": 2014,
                    "total_attendance": 6,
                    "standing": "G",
                    "year": 4,
                    "last_attendance_time": "2023-01-19 00:54:34"
        }
    }

for key, value in data.items():
    ref.child(key).set(value)