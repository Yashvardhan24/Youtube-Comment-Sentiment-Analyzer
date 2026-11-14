import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="YouTube Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("YouTube Comment Sentiment Analyzer")
st.write("Analyze audience sentiment using deep learning.")

# User inputs
video_url = st.text_input("Enter YouTube Video URL")

max_comments = st.slider(
    "Maximum comments to analyze",
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)

if st.button("Analyze Comments", type="primary"):

    if not video_url.strip():
        st.error("Please enter a YouTube URL.")

    else:
        with st.spinner("Fetching comments and analyzing sentiment..."):

            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={
                        "video_url": video_url,
                        "max_comments": max_comments
                    },
                    timeout=300
                )

                response.raise_for_status()
                data = response.json()

                st.success("Analysis completed!")

                # Video information
                st.header(data["video_title"])
                st.write("Channel:", data["channel"])

                st.write(
                    f"Fetched: {data['total_fetched']} | "
                    f"Analyzed: {data['total_analyzed']}"
                )

                # Sentiment metrics
                counts = data["sentiment_counts"]

                st.subheader("Sentiment Overview")

                c1, c2, c3 = st.columns(3)

                c1.metric("Positive", counts["Positive"])
                c2.metric("Negative", counts["Negative"])
                c3.metric("Neutral", counts["Neutral"])

                # Charts
                chart_data = pd.DataFrame({
                    "Sentiment": list(counts.keys()),
                    "Comments": list(counts.values())
                })

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.pie(
                        chart_data,
                        names="Sentiment",
                        values="Comments",
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        chart_data,
                        x="Sentiment",
                        y="Comments",
                        color="Sentiment",
                        title="Sentiment Counts"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Individual comments
                st.subheader("Comment Analysis")

                comments_df = pd.DataFrame(data["comments"])

                if not comments_df.empty:
                    st.dataframe(
                        comments_df[
                            ["text", "sentiment", "confidence", "likes"]
                        ],
                        use_container_width=True
                    )

                    csv = comments_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "sentiment_analysis.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No usable comments were available.")

            except requests.exceptions.RequestException as e:
                st.error(f"Backend request failed: {e}")

            except Exception as e:
                st.error(f"Analysis failed: {e}")