import sys, site, platform
print("PYTHON:", sys.executable)
print("SITE:", site.getsitepackages() or [site.getusersitepackages()])
print("VER:", platform.python_version())


import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1 import FieldFilter, DELETE_FIELD
import pandas as pd
import os


cred = credentials.Certificate("the-never-ending-story-a7b5f-firebase-adminsdk-ax73p-9014b38264.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

BATCH_SIZE = 400
batch = db.batch()
story_data_ref = db.collection("story_data_berlin")
stories_ref = db.collection("stories")

def save_csv():
    home_dir = os.path.expanduser("~")
    csvdownload_path = os.path.join(home_dir, "Documents", "Berlin_data", "finished_stories.csv")
    df = pd.DataFrame(make_dataframe())
    if 'timestamp' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["timestamp"])
    df.to_csv(csvdownload_path, index=False)
    
def make_dataframe():
    count_workshop1_interactions = 0
    count_stories = 0
    data = []
    full_story_data = []
    conversation_id = ""
    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    for doc in docs:
        if conversation_id == doc.get("conversation_id"):
            count_workshop1_interactions += 1
        conversation_id = doc.get("conversation_id")
        doc_dictionary = doc.to_dict()
        timestamp = doc_dictionary.get("timestamp")
        user = doc_dictionary.get("user")
        ai = doc_dictionary.get("ai")
        combined_prompt = doc_dictionary.get("combined_prompt")
        client_id = doc_dictionary.get("client_id")
        workshop_id = doc_dictionary.get("workshop_id")
        language = doc_dictionary.get("language")
        data.append({
            "timestamp": timestamp,
            "user": user,
            "ai": ai,
            "combined_prompt": combined_prompt,
            "client_id": client_id,
            "workshop_id": workshop_id,
            "language": language,
            "conversation_id": conversation_id
        })
        if count_workshop1_interactions >= 4:
            count_workshop1_interactions = 0
            count_stories += 1
            full_story_data.append(data)
        else:
            conversation_id = doc.get("conversation_id")
    print(f"Number of stories:", count_stories)
    return data

if __name__ == "__main__":
    save_csv()