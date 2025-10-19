import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from datetime import datetime

# =========================================
# 🚀 Initialize YouTube API
# =========================================
try:
    api_key = st.secrets["YOUTUBE_API_KEY"]
    youtube = build("youtube", "v3", developerKey=api_key)
except Exception as e:
    st.warning("⚠️ YouTube API key not found or invalid. You can still use the app with local data.")
    youtube = None

# =========================================
# 🎵 Example Local Data (Fallback)
# =========================================
data = [
    {"title": "Deep Focus Beats", "type": "Music", "mood": "Focus", "link": "https://www.youtube.com/watch?v=jfKfPfyJRdk"},
    {"title": "Calming Ocean Waves", "type": "Music", "mood": "Relaxed", "link": "https://www.youtube.com/watch?v=DWcJFNfaw9c"},
    {"title": "5-Minute Guided Meditation", "type": "Meditation", "mood": "Peaceful", "link": "https://www.youtube.com/watch?v=inpok4MKVLM"},
    {"title": "Motivational Speech", "type": "Video", "mood": "Motivated", "link": "https://www.youtube.com/watch?v=mgmVOuLgFB0"},
]
df_content = pd.DataFrame(data)

# =========================================
# 💡 YouTube Search Helper
# =========================================
def youtube_search(query, max_results=3):
    if youtube is None:
        return []

    try:
        search_response = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=max_results,
            type="video"
        ).execute()

        videos = []
        for item in search_response["items"]:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append({"title": title, "link": video_url})

        return videos
    except Exception as e:
        st.error(f"Error fetching YouTube data: {e}")
        return []

# =========================================
# 🧠 Recommendation Logic
# =========================================
def recommend_content(goal, df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["title"])
    goal_vec = vectorizer.transform([goal])
    cosine_sim = cosine_similarity(goal_vec, tfidf_matrix)
    idx = cosine_sim.argsort()[0][::-1]
    top_results = df.iloc[idx[:3]]
    return top_results

# =========================================
# 🧍 User Profile Initialization
# =========================================
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "satisfaction" not in st.session_state:
    st.session_state.satisfaction = []

# =========================================
# 🎨 Streamlit Interface
# =========================================
st.set_page_config(page_title="Mindful Content Companion", layout="wide")

st.sidebar.header("🧘 User Profile")
st.sidebar.write(f"**Total Sessions:** {len(st.session_state.sessions)}")
if len(st.session_state.satisfaction) > 0:
    avg_satisfaction = round(sum(st.session_state.satisfaction) / len(st.session_state.satisfaction) * 100, 2)
    st.sidebar.write(f"**Avg Satisfaction:** {avg_satisfaction}%")
else:
    st.sidebar.write("Avg Satisfaction: 0%")

st.sidebar.write("---")
st.sidebar.write("🌿 *MindfulApp helps track emotions and recommend personalized relaxing content.*")

# =========================================
# ✨ User Inputs
# =========================================
st.title("🌻 Mindful Content Companion (MCC)")
st.subheader("Track your mood, explore mindful content, and reflect on your emotional growth.")

# --- Updated Mood Input ---
st.markdown("### 💭 How are you feeling right now?")

default_moods = ["Stressed", "Anxious", "Tired", "Motivated", "Distracted", "Relaxed", "Other"]
selected_mood = st.selectbox("Select a mood or choose 'Other' to type your own:", default_moods)

if selected_mood == "Other":
    mood = st.text_input("Type your current mood in one word:", placeholder="e.g., Overwhelmed, Excited, Lonely, Curious")
else:
    mood = selected_mood

# --- Other inputs ---
goal = st.text_input("What do you want to achieve right now?", placeholder="e.g. Need something calming for focus")
desired_mood = st.selectbox("How do you want to feel after this?", ["Relaxed", "Motivated", "Peaceful", "Focused"])

if st.button("🌱 Generate Recommendations"):
    if goal.strip() == "":
        st.warning("Please describe your goal first.")
    else:
        with st.spinner("Finding the best mindful content for you..."):
            local_recs = recommend_content(goal, df_content)
            yt_results = youtube_search(goal) if youtube else []

        st.session_state.sessions.append({
            "mood": mood,
            "goal": goal,
            "target": desired_mood,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.success(f"🎯 Recommended for '{goal}' → Target mood: {desired_mood}")

        st.markdown("### 🌼 Suggested Content")
        for _, row in local_recs.iterrows():
            st.markdown(f"**{row['title']}**  \n_Type: {row['type']}_  \n[Watch here]({row['link']})")

        if len(yt_results) > 0:
            st.markdown("### 🎥 YouTube Suggestions")
            for vid in yt_results:
                st.markdown(f"**{vid['title']}**  \n[Open on YouTube]({vid['link']})")

        # Satisfaction feedback
        st.markdown("---")
        st.subheader("💬 Did this help you?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes, I'm satisfied"):
                st.session_state.satisfaction.append(1)
                st.success("Thank you! Glad it helped 🌿")
        with col2:
            if st.button("👎 Not really"):
                st.session_state.satisfaction.append(0)
                st.info("Got it — we'll improve next time 💪")

# =========================================
# 🕒 Past Sessions
# =========================================
if len(st.session_state.sessions) > 0:
    st.markdown("---")
    st.markdown("### 🧾 Your Past Sessions")
    history_df = pd.DataFrame(st.session_state.sessions)
    st.dataframe(history_df)



