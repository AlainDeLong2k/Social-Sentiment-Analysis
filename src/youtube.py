import os
import googleapiclient.discovery
import googleapiclient.errors
from dotenv import load_dotenv

# from dotenv import load_dotenv
import streamlit as st

load_dotenv()
api_key = os.getenv("API_KEY")
# api_key = st.secrets["API_KEY"]


def get_comments(youtube, **kwargs):
    comments = []
    results = youtube.commentThreads().list(**kwargs).execute()

    while results:
        for item in results["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        # check if there are more comments
        if "nextPageToken" in results:
            kwargs["pageToken"] = results["nextPageToken"]
            results = youtube.commentThreads().list(**kwargs).execute()
        else:
            break

    return comments


def main(video_id, api_key):
    # Disable OAuthlib's HTTPs verification when running locally.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

    video_title = "N/A"  # Provide a default title

    try:
        # Get video details using the videos().list endpoint
        print(f"DEBUG (youtube.py): Fetching video details for ID: {video_id}")
        video_response = (
            youtube.videos()
            .list(
                part="snippet",  # 'snippet' contains title, description, channel etc.
                id=video_id,  # The ID of the video user want info for
            )
            .execute()
        )

        # Extract the title from the response
        # It's usually nested like this, good to check if 'items' exists
        if video_response.get("items"):
            video_title = video_response["items"][0]["snippet"]["title"]
            print(f"DEBUG (youtube.py): Found title: '{video_title}'")  # Just a check
        else:
            print(f"WARN (youtube.py): No video items found for ID: {video_id}")
            video_title = "Video Not Found or Private"  # More informative default

    except Exception as e:
        print(
            f"ERROR (youtube.py): Failed to fetch video title for ID {video_id}. Error: {e}"
        )
        video_title = "Error Fetching Title"  # Error specific default

    comments = get_comments(
        youtube, part="snippet", videoId=video_id, textFormat="plainText"
    )
    # return comments
    # Return a dictionary containing both title and comments
    return {"title": video_title, "comments": comments}


def get_video_comments(video_id):
    return main(video_id, api_key)
