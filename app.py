import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from datetime import datetime

# ====================================================
# 1️⃣  DATA  &  LOGIC
# ====================================================

def load_content_database():
    data = {
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'title': [
            "10-Minute Sun Salutation Yoga",
            "The Science of Gratitude (TED Talk)",
            "Upbeat Lo-Fi Focus Music Playlist",
            "5-Minute Guided Progressive Muscle Relaxation",
            "Nature Sounds: Forest Walk & River Flow",
            "Article: Short-Term Goal Setting for Productivity",
            "Comedy Clip: Stand-up Routine for a Quick Laugh",
            "Slow, Deep Diaphragmatic Breathing Guide",
            "Animated Short: The Ant and the Grasshopper",
            "Visual: Calming Geometric Patterns",
            "Deep Focus Beats – Study Music",
            "10-Minute Guided Meditation for Stress Relief",
            "Funny Animal Compilation – Relax & Laugh"
        ],
        'link': [
            "https://www.youtube.com/watch?v=oBu-pQG6sTY",
            "https://www.youtube.com/watch?v=JMd1CcGPHAg",
            "https://www.youtube.com/watch?v=jfKfPfyJRdk",
            "https://www.youtube.com/watch?v=ihO02wUzgkc",
            "https://www.youtube.com/watch?v=OdIJ2x3nxzQ",
            "https://www.verywellmind.com/how-to-set-goals-2791797",
            "https://www.youtube.com/watch?v=Jb8Rju6lqjw",
            "https://www.youtube.com/watch?v=nkK1bT8ls0M",
            "https://www.youtube.com/watch?v=cGz_0q8OGP0",
            "https://www.youtube.com/watch?v=1HkT6DhTqdc",
            "https://www.youtube.com/watch?v=WPni755-Krg",
            "https://www.youtube.com/watch?v=ZToicYcHIOU",
            "https://www.youtube.com/watch?v=YZbUO2LZl3c"
        ],
        'type': [
            "Video (Active)", "Video (Uplift)", "Audio (Focus)", "Technique",
            "Audio (Calm)", "Article", "Video (Uplift)", "Technique",
            "Video (Uplift)", "Visual (Calm)", "Audio (Focus)",
            "Technique (Calm)", "Video (Uplift)"
        ],
        'tags': [
            "Energizing Active Uplifting Focus Stress",
            "Uplifting Inspiring Motivational Sadness Perspective",
            "Focus Concentration ADHD Work Productivity Calm",
            "Calm Relaxation Stress Anxiety Sleep",
            "Calm Focus Relaxation Stress Nature",
            "Focus Productivity Motivation Goals",
            "Uplifting Joy Sadness Energy",
            "Calm Breathing Focus Anxiety Stress",
            "Uplifting Motivational Sadness Perspective",
            "Calm Focus Distraction ADHD",
            "Focus Study Calm Productivity Concentration",
            "Calm Stress Relief Meditation Mindfulness Sleep",
            "Laughter Joy Relaxation Stress Relief Fun"
        ]
    }
    df = pd.DataFrame(data)
    df.index = df['id']
    return df

CONTRARIAN_MAP = {
    'Sad / Low': 'Uplifting Joy Energy Motivational',
    'Anxious / Stressed': 'Calm Relaxation Breathing Focus',
    'Distracted / ADHD': 'Focus Concentration Calm Productivity',
    'Tired / Insomnia': 'Relaxation Sleep Restful Calm',
    'Angry / Frustrated': 'Calm Perspective Breathing'
}

@st.cache_data
def get_vectorizer(_df):
    vec = TfidfVectorizer(stop_words='english', token_pattern=r'\b\w+\b')
    vec.fit(_df['tags'])
    return vec

@st.cache_data
def get_content_matrix(df, _vec):
    return _vec.transform(df['tags'])

def recommend_content(target_tags, df):
    vec = get_vectorizer(df)
    mat = get_content_matrix(df, vec)
    target = vec.transform([target_tags])
    sims = cosine_similarity(target, mat).flatten()
    scores = pd.Series(sims, index=df.index).sort_values(ascending=False)
    top = df.loc[scores.index[:3]].copy()
    top['confidence'] = [scores[i] for i in top.index]
    return top

# ====================================================
# 2️⃣  USER DATA MANAGEMENT
# ====================================================

USER_DATA_FILE = "user_data.csv"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        return pd.read_csv(USER_DATA_FILE)
    else:
        return pd.DataFrame(columns=["timestamp","before_mood","after_mood","satisfied"])

def save_user_entry(before, after, satisfied):
    df = load_user_data()
    df.loc[len(df)] = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), before, after, satisfied]
    df.to_csv(USER_DATA_FILE, index=False)

# ====================================================
# 3️⃣  STREAMLIT APP
# ====================================================

def main():
    st.set_page_config(layout="wide", page_title="Mindful Content Companion (MCC)")

    # Sidebar – user profile
    st.sidebar.header("🪪 User Profile")
    user_data = load_user_data()
    total = len(user_data)
    satisfied_rate = user_data['satisfied'].mean()*100 if total>0 else 0

    st.sidebar.write(f"Total Sessions: **{total}**")
    st.sidebar.write(f"Avg Satisfaction: **{satisfied_rate:.1f}%**")

    if total > 1:
        st.sidebar.bar_chart(user_data['satisfied'])

    df_content = load_content_database()

    st.markdown("<h1 style='text-align:center;color:#4C00B0'>✨ Mindful Content Companion (MCC)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Track your moods, get mindful content, and see your emotional progress.</p>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1,2])

    with col_left:
        st.subheader("1. How are you feeling now?")
        mood = st.selectbox("Select mood:", list(CONTRARIAN_MAP.keys()), index=None, placeholder="Choose...")

        st.subheader("2. What do you want to achieve?")
        intent = st.text_area("Describe your goal:", placeholder="e.g., I want calm background music")

        after_mood = st.selectbox("3. How do you want to feel after this session?", 
                                  ["Calm","Focused","Happy","Motivated","Relaxed","Energetic"], index=None)

        if st.button("🌿 Generate Recommendations"):
            if not mood:
                st.warning("Please select a mood first.")
            else:
                st.session_state['result'] = recommend_content(CONTRARIAN_MAP[mood]+" "+intent, df_content)
                st.session_state['mood'] = mood
                st.session_state['after'] = after_mood

    with col_right:
        st.subheader("Your Personalized Recommendations")

        if 'result' in st.session_state:
            recs = st.session_state['result']
            st.success(f"Target: From **{st.session_state['mood']}** → **{st.session_state['after']}**")

            for _, r in recs.iterrows():
                st.markdown(f"**{r['title']}**")
                st.info(f"Type: {r['type']} | Confidence: {r['confidence']:.2f}")
                st.markdown(f"[Open Content 🎬]({r['link']})", unsafe_allow_html=True)
                st.markdown("---")

            st.write("### Did this help you?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("👍 Yes, I'm satisfied"):
                    save_user_entry(st.session_state['mood'], st.session_state['after'], 1)
                    st.success("Thanks! Your feedback helps improve future suggestions.")
            with c2:
                if st.button("👎 Not really"):
                    save_user_entry(st.session_state['mood'], st.session_state['after'], 0)
                    st.info("Got it — we’ll adjust future recommendations.")
        else:
            st.info("Select your mood and goal, then click *Generate Recommendations*.")

# ====================================================
# RUN
# ====================================================
if __name__ == "__main__":
    main()
