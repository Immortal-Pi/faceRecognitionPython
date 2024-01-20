import firebase_admin
from firebase_admin import db
cred = firebase_admin.credentials.Certificate('serviceAccountKey.json')

firebase_admin.initialize_app(cred,
                              {'databaseURL':"https://facerecognitionrealtimepython-default-rtdb.firebaseio.com/"}
                              )

referenceStudent=db.reference('StudentsTest2')

data={
    "16313":{
        "name":"amruth",
        "age" : 26,
        "gender":"M",
        "height":5.9,
        "weight":72
    },
    "16314":{
        "name":"elon",
        "age" : 25,
        "gender":"M",
        "height":6.0,
        "weight":90
    },

    "16315":{
        "name":"ninja",
        "age" : 30,
        "gender":"M",
        "height":5.8,
        "weight":100
    }
}


for k,v in data.items():
    referenceStudent.child(k).set(v)