import json
from datetime import date, datetime
import firebase_admin
from firebase_admin import credentials, firestore
from os import path


cred = credentials.Certificate("goal-switching-0-firebase-adminsdk-x34tq-a1ca95a6b8.json") # here you need to put a specific key from the database (and you should only initialise the database once to get the #data off)

default_app = firebase_admin.initialize_app(cred)

collection_name = 'tasks'
document_name = 'goal-switching-trials-v4' # here the task name goes, the one you specified in your init db file
sub_collection = 'subjects'
client = firestore.client()


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))


if __name__ == "__main__":
    for subject in client.collection(collection_name).document(document_name).collection(sub_collection).stream():
        subject_data = subject.to_dict()
        subID_database = subject.id
        subID_prolific = subject_data['subjectID']
        filename = u'switching-v1/%s_%s.json' % (subID_prolific, subID_database)
        if path.exists(filename) == False:
            with open(filename, 'w') as fp:
                json.dump(subject_data, fp, default=json_serial, indent=4)