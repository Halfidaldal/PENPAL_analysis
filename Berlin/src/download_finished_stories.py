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
story_data_ref = db.collection("story_data_TEXT")
stories_ref = db.collection("stories_TEXT")

def save_csv():
    home_dir = os.path.expanduser("~")
    csvdownload_path = os.path.join(home_dir, "Documents", "PENPAL_analysis", "finished_stories.csv")
    df = pd.DataFrame(make_dataframe())
    if 'timestamp' in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["timestamp"])
    df.to_csv(csvdownload_path, index=False)
    
def make_dataframe():
    # Build per-conversation chunks and return only rows that belong to full stories
    count = {}
    out_rows = []          # flattened list of dicts for all full-story rows
    full_story_data = []   # list of story groups (each is a list of dicts of length 10)
    current_conv = None
    conv_docs = []

    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    for doc in docs:
        doc_conv = doc.get("conversation_id")

        # when conversation changes, process the accumulated conv_docs
        if current_conv is not None and doc_conv != current_conv:
            # count stories in the accumulated docs
            n = len(conv_docs)
            num_full = n // 10
            if num_full:
                # split into full 10-item stories
                for i in range(num_full):
                    story_slice = conv_docs[i*10:(i+1)*10]
                    full_story_data.append(story_slice)
                    out_rows.extend(story_slice)
                count[current_conv] = count.get(current_conv, 0) + num_full
            # reset for new conversation
            conv_docs = []
            current_conv = doc_conv
        else:
            # first doc of a new conversation
            if current_conv is None:
                current_conv = doc_conv

        # build row dict for current doc and append to conv_docs
        doc_dictionary = doc.to_dict()
        row = {
            "timestamp": doc_dictionary.get("timestamp"),
            "user": doc_dictionary.get("user"),
            "ai": doc_dictionary.get("ai"),
            "combined_prompt": doc_dictionary.get("combined_prompt"),
            "client_id": doc_dictionary.get("client_id"),
            "workshop_id": doc_dictionary.get("workshop_id"),
            "language": doc_dictionary.get("language"),
            "conversation_id": doc_conv,
        }
        conv_docs.append(row)

    # process the final conversation after the loop
    if current_conv is not None and conv_docs:
        n = len(conv_docs)
        num_full = n // 10
        if num_full:
            for i in range(num_full):
                story_slice = conv_docs[i*10:(i+1)*10]
                full_story_data.append(story_slice)
                out_rows.extend(story_slice)
            count[current_conv] = count.get(current_conv, 0) + num_full

    total_stories = sum(count.values()) if count else 0
    print(f"Number of stories: {total_stories}")
    # return flattened rows for CSV export (only rows that are part of full 10-item stories)
    return out_rows

if __name__ == "__main__":
    save_csv()