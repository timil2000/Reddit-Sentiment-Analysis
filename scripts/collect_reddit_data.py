import os
import praw
import pandas as pd

# Set up Reddit API connection
reddit = praw.Reddit(
    client_id="your_client_id",
    client_secret="your_client_secret",
    user_agent="your_user_agent",
)


# Function to collect posts
def fetch_reddit_posts(subreddit_name, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []

    for post in subreddit.hot(limit=limit):
        posts.append({
            "post_id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "upvotes": post.score,
            "comments_count": post.num_comments,
            "created_utc": post.created_utc,
            "url": post.url,
        })

    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame(posts)
    df.to_csv(f"data/{subreddit_name}_posts.csv", index=False)
    print(f"Saved {len(df)} posts from r/{subreddit_name}")


# Run script
if __name__ == "__main__":
    fetch_reddit_posts("technology", limit=50)