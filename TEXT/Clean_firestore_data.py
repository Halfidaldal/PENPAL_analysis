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


def delete_stories():
    counts = {}
    docs = story_data_ref.order_by("conversation_id").order_by("timestamp").stream()
    conversation_id = None
    for doc in docs:
        # only consider deletion when we have a previous conversation_id
        if conversation_id is not None and conversation_id != doc.get("conversation_id") and counts.get(conversation_id, 0) < 10:
            print("Deleting story with conversation_id:", conversation_id)
            story_query = stories_ref.where(
                FieldFilter("conversation_id", "==", conversation_id)
            )
            story_docs = story_query.stream()
            for story_doc in story_docs:
                print(" - Deleting story document ID:", story_doc.id)
                stories_ref.document(story_doc.id).delete()
            
        conversation_id = doc.get("conversation_id")
        if conversation_id is None:
            continue
        counts[conversation_id] = counts.get(conversation_id, 0) + 1

    # final check for the last conversation after streaming all documents
    if conversation_id is not None and counts.get(conversation_id, 0) < 10:
        print("Deleting story with conversation_id:", conversation_id)
        story_query = stories_ref.where(
            FieldFilter("conversation_id", "==", conversation_id)
        )
        story_docs = story_query.stream()
        for story_doc in story_docs:
            print(" - Deleting story document ID:", story_doc.id)
            stories_ref.document(story_doc.id).delete()

    story_count = sum(cnt // 10 for cnt in counts.values())

    print(f"Number of stories: {story_count}")
    return story_count

if __name__ == "__main__":
    delete_stories()
    