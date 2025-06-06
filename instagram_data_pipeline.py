import os
import datetime
from typing import Dict, Any, Optional, List
from pymongo.database import Database
from pymongo.collection import Collection
from dotenv import load_dotenv
import google.generativeai as genai

# For scraping, use instaloader (posts/comments), but reels/stories are limited.
import instaloader
import tempfile
import requests

# --- CONFIG LOADING EXAMPLE ---

def load_config() -> Dict[str, Any]:
    """
    Loads configuration from a .env file.
    Returns a dictionary with config values.
    """
    load_dotenv()
    config = {
        "MONGODB_URI": os.getenv("MONGODB_URI"),
        "MONGODB_DBNAME": os.getenv("MONGODB_DBNAME"),
        "INSTAGRAM_ACCOUNTS_COLLECTION": os.getenv("INSTAGRAM_ACCOUNTS_COLLECTION", "instagram_accounts"),
        "INSTAGRAM_POSTS_COLLECTION": os.getenv("INSTAGRAM_POSTS_COLLECTION", "instagram_posts"),
        "INSTAGRAM_STORIES_COLLECTION": os.getenv("INSTAGRAM_STORIES_COLLECTION", "instagram_stories"),
        "INSTAGRAM_ANALYSIS_COLLECTION": os.getenv("INSTAGRAM_ANALYSIS_COLLECTION", "instagram_analysis"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "GEMINI_MODEL": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),  
        "INSTAGRAM_USERNAME_ANALYZE":  os.getenv("INSTAGRAM_USERNAME_ANALYZE", "instagram_username"),
    }
    return config

# --- FUNCTION 1: EXTRACT AND SAVE INSTAGRAM DATA ---

# --- CONSTANTES DE PARÁMETROS ---
MAX_POSTS_ANALYSIS = 9  # Máximo de posts a analizar si no hay suficientes en 15 días
MAX_DAYS_ANALYSIS = 15  # Días hacia atrás para filtrar posts recientes
MAX_IMAGES_ACCOUNT_PROMPT = 5  # Máximo de imágenes en el prompt de análisis de cuenta
MAX_IMAGES_POST_PROMPT = 3     # Máximo de imágenes en el prompt de análisis de post
MAX_COMMENTS_ANALYSIS = 50     # Máximo de comentarios a analizar en el prompt
ACCOUNT_SUGGESTIONS_COUNT = 3  # Cantidad de sugerencias para el análisis de cuenta
POST_SUGGESTIONS_COUNT = 3     # Cantidad de sugerencias para el análisis de post

def get_and_save_instagram_data(
    username: str,
    db_client: Database,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extracts Instagram account data, posts from the last MAX_DAYS_ANALYSIS days or last MAX_POSTS_ANALYSIS posts,
    and stories (if possible), saves them to NoSQL (MongoDB), and returns a summary.
    """
    summary = {"account": None, "posts": [], "stories": []}
    extraction_date = datetime.datetime.utcnow()
    L = instaloader.Instaloader(download_pictures=False, download_videos=False, download_video_thumbnails=False, download_comments=True, save_metadata=False, compress_json=False)
    try:
        profile = instaloader.Profile.from_username(L.context, username)
        # --- Account Info ---
        account_doc = {
            "_id": username,
            "username": username,
            "full_name": profile.full_name,
            "biography": profile.biography,
            "profile_pic_url": profile.profile_pic_url,
            "followers": profile.followers,
            "followees": profile.followees,
            "is_verified": profile.is_verified,
            "is_private": profile.is_private,
            "external_url": profile.external_url,
            "account_extraction_date": extraction_date,
        }
        db_client[config["INSTAGRAM_ACCOUNTS_COLLECTION"]].update_one(
            {"_id": username}, {"$set": account_doc}, upsert=True
        )
        summary["account"] = account_doc

        # --- Posts (últimos MAX_DAYS_ANALYSIS días o últimos MAX_POSTS_ANALYSIS posts) ---
        posts_collection: Collection = db_client[config["INSTAGRAM_POSTS_COLLECTION"]]
        days_ago = extraction_date - datetime.timedelta(days=MAX_DAYS_ANALYSIS)
        posts_added = 0
        for post in profile.get_posts():
            if post.date_utc < days_ago and posts_added >= MAX_POSTS_ANALYSIS:
                break  # Stop if both conditions are met
            images = []
            if post.typename == "GraphSidecar":
                try:
                    for res in post.get_sidecar_nodes():
                        images.append(res.display_url)
                except Exception:
                    pass
            else:
                images.append(post.url)
            post_doc = {
                "_id": post.mediaid,
                "account_username": username,
                "url": f"https://www.instagram.com/p/{post.shortcode}/",
                "type": "video" if post.is_video else "image",
                "caption": post.caption,
                "date": post.date_utc,
                "likes": post.likes,
                "comments_count": post.comments,
                "comments": [],
                "images": images[:MAX_IMAGES_POST_PROMPT],
                "post_extraction_date": extraction_date,
            }
            # Comments
            try:
                for comment in post.get_comments():
                    if len(post_doc["comments"]) >= MAX_COMMENTS_ANALYSIS:
                        break
                    post_doc["comments"].append({
                        "id": comment.id,
                        "text": comment.text,
                        "author": comment.owner.username,
                        "date": comment.created_at_utc,
                    })
            except Exception:
                pass  # Comments may not always be accessible
            posts_collection.update_one(
                {"_id": post.mediaid}, {"$set": post_doc}, upsert=True
            )
            summary["posts"].append(post_doc)
            posts_added += 1

        # --- Stories (very limited, only if authenticated and only current stories) ---
        stories_collection: Collection = db_client[config["INSTAGRAM_STORIES_COLLECTION"]]
        try:
            for story in L.get_stories(userids=[profile.userid]):
                for item in story.get_items():
                    story_doc = {
                        "_id": item.mediaid,
                        "account_username": username,
                        "media_type": "video" if item.is_video else "image",
                        "media_url": item.url,
                        "publication_date": item.date_utc,
                        "expiration_date": item.date_utc + datetime.timedelta(hours=24),
                        "view_count": None,  # Not accessible via scraping
                        "interactions": [],
                        "story_extraction_date": extraction_date,
                    }
                    stories_collection.update_one(
                        {"_id": item.mediaid}, {"$set": story_doc}, upsert=True
                    )
                    summary["stories"].append(story_doc)
        except Exception:
            pass  # Stories may not be accessible

    except Exception as e:
        print(f"Error extracting data for {username}: {e}")

    return summary

def download_images(image_urls: list) -> list:
    """
    Downloads images from the provided URLs and returns a list of local file paths.
    """
    local_paths = []
    for url in image_urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            suffix = url.split('.')[-1].split('?')[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
                tmp_file.write(response.content)
                local_paths.append(tmp_file.name)
        except Exception as e:
            print(f"Error downloading image {url}: {e}")
    return local_paths

# --- FUNCTION 2: ANALYZE, SUGGEST WITH GEMINI, AND SAVE ---

def analyze_suggest_and_save_instagram(
    analyzed_username: str,
    gemini_api_key: str,
    db_client: Database,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reads Instagram data for analyzed_username, downloads images, analyzes with Gemini API (including images),
    saves suggestions to NoSQL, and returns the analysis report.
    Only analyzes posts from the last MAX_DAYS_ANALYSIS days or last MAX_POSTS_ANALYSIS posts and includes images in the prompts.
    """
    analysis_date = datetime.datetime.utcnow()
    analysis_report = {
        "username": analyzed_username,
        "analysis_date": analysis_date,
        "account_level": None,
        "content_level": [],
        "comment_level": None,
    }

    # --- Load Data ---
    accounts_col = db_client[config["INSTAGRAM_ACCOUNTS_COLLECTION"]]
    posts_col = db_client[config["INSTAGRAM_POSTS_COLLECTION"]]
    stories_col = db_client[config["INSTAGRAM_STORIES_COLLECTION"]]

    account = accounts_col.find_one({"_id": analyzed_username})
    all_posts = list(posts_col.find({"account_username": analyzed_username}).sort("date", -1))
    days_ago = analysis_date - datetime.timedelta(days=MAX_DAYS_ANALYSIS)
    # Get posts from last MAX_DAYS_ANALYSIS days or last MAX_POSTS_ANALYSIS posts, whichever is more
    posts = []
    for post in all_posts:
        if post.get("date") and post["date"] >= days_ago:
            posts.append(post)
        elif len(posts) < MAX_POSTS_ANALYSIS:
            posts.append(post)
        else:
            break
    stories = list(stories_col.find({"account_username": analyzed_username}))

    # --- Setup Gemini ---
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel(config.get("GEMINI_MODEL", "gemini-2.0-flash"))

    # --- Account-Level Analysis (download and include images from recent posts) ---
    images_for_prompt = []
    for post in posts:
        images_for_prompt.extend(post.get("images", []))
    images_for_prompt = images_for_prompt[:MAX_IMAGES_ACCOUNT_PROMPT]
    local_account_images = download_images(images_for_prompt)

    account_prompt = (
        f"Instagram account analysis:\n"
        f"Bio: {account.get('biography')}\n"
        f"Followers: {account.get('followers')}\n"
        f"Posts (last {MAX_DAYS_ANALYSIS} days or last {MAX_POSTS_ANALYSIS}): {len(posts)}\n"
        f"Stories: {len(stories)}\n"
        f"Evaluate the consistency and effectiveness of the bio and the overall use of Posts and Stories. "
        f"Suggest {ACCOUNT_SUGGESTIONS_COUNT} key improvements."
    )
    try:
        account_resp = model.generate_content(
            [account_prompt] + local_account_images
        )
        analysis_report["account_level"] = account_resp.text
    except Exception as e:
        analysis_report["account_level"] = f"Error: {e}"

    # --- Content-Level Analysis (Posts only, download and include images) ---
    for post in posts:
        post_images = post.get("images", [])[:MAX_IMAGES_POST_PROMPT]
        local_post_images = download_images(post_images)
        post_prompt = (
            f"Instagram post analysis:\n"
            f"Caption: {post.get('caption')}\n"
            f"Likes: {post.get('likes')}\n"
            f"Comments: {post.get('comments_count')}\n"
            f"How effective is this post? Suggest {POST_SUGGESTIONS_COUNT} ways to improve engagement for future posts with similar themes."
        )
        try:
            post_resp = model.generate_content(
                [post_prompt] + local_post_images
            )
            analysis_report["content_level"].append({
                "type": "post",
                "id": post["_id"],
                "suggestion": post_resp.text
            })
        except Exception as e:
            analysis_report["content_level"].append({
                "type": "post",
                "id": post["_id"],
                "suggestion": f"Error: {e}"
            })

    # --- Comment-Level Analysis (from recent posts only) ---
    all_comments = []
    for post in posts:
        all_comments.extend([c["text"] for c in post.get("comments", [])])
    if all_comments:
        comment_prompt = (
            f"Analyze the overall sentiment of these post comments:\n"
            f"{all_comments[:MAX_COMMENTS_ANALYSIS]}\n"
            f"Are there recurring themes or audience questions that the account could address better?"
        )
        try:
            comment_resp = model.generate_content(comment_prompt)
            analysis_report["comment_level"] = comment_resp.text
        except Exception as e:
            analysis_report["comment_level"] = f"Error: {e}"

    # --- Save Analysis ---
    analysis_col = db_client[config["INSTAGRAM_ANALYSIS_COLLECTION"]]
    analysis_col.update_one(
        {"username": analyzed_username},
        {"$set": analysis_report},
        upsert=True
    )

    return analysis_report

# --- EXAMPLE .env FILE ---
# MONGODB_URI=mongodb://localhost:27017
# MONGODB_DBNAME=instagram_db
# INSTAGRAM_ACCOUNTS_COLLECTION=instagram_accounts
# INSTAGRAM_POSTS_COLLECTION=instagram_posts
# INSTAGRAM_REELS_COLLECTION=instagram_reels
# INSTAGRAM_STORIES_COLLECTION=instagram_stories
# INSTAGRAM_ANALYSIS_COLLECTION=instagram_analysis
# GEMINI_API_KEY=your_gemini_api_key_here

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    from pymongo import MongoClient

    # Load config
    config = load_config()

    # Connect to MongoDB
    mongo_client = MongoClient(config["MONGODB_URI"])
    db = mongo_client[config["MONGODB_DBNAME"]]

    # Extract and save Instagram data
    summary = get_and_save_instagram_data(
        config["INSTAGRAM_USERNAME_ANALYZE"],
         db, config)
    print("Extraction summary:", summary)

    # Analyze and save suggestions
    analysis = analyze_suggest_and_save_instagram(
        analyzed_username="instagram_username",
        gemini_api_key=config["GEMINI_API_KEY"],
        db_client=db,
        config=config
    )
    print("Analysis report:", analysis)