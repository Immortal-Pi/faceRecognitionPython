import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
"""
using firebase
serviceAccountKey is the private key downloaded from firebase for connecting x

"""
cred = credentials.Certificate("serviceAccountKey.json")
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
                    "last_attendance_time": "2024-01-19 00:54:34"
        }
    }

for key,value in data.items():
    ref.child(key).set(value)
