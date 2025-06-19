import streamlit as st
import pandas as pd
import re  # For robust YouTube video ID extraction
import plotly.express as px

# Import custom modules from the src directory
from src.predict import predict_sentiments
from src.youtube import get_video_comments


def extract_video_id(url_or_id: str):
    """
    Tries to get the YouTube video ID from different common URL types.
    Also handles if the input is just the ID itself.
    A bit of regex to find the ID part in common URLs.
    """
    if not url_or_id:
        return None

    # Patterns for various YouTube URL formats
    # Order matters: more specific patterns should come first if overlap exists
    patterns = [
        r"watch\?v=([a-zA-Z0-9_-]{11})",  # Standard watch URL
        r"youtu\.be/([a-zA-Z0-9_-]{11})",  # Shortened URL
        r"embed/([a-zA-Z0-9_-]{11})",  # Embed URL
        r"shorts/([a-zA-Z0-9_-]{11})",  # Shorts URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)  # The first capturing group is the ID

    # If no pattern matches, check if the input itself is a valid 11-char ID
    # Basic check: 11 chars, no spaces, not starting with http (already handled by regex above implicitly)
    if len(url_or_id) == 11 and not (
        "/" in url_or_id or "?" in url_or_id or "=" in url_or_id or "." in url_or_id
    ):
        return url_or_id  # Assume it's a direct ID

    return None  # Return None if no ID found


def analyze_youtube_video(video_url_or_id: str):
    """
    Main function for the YouTube analysis part.
    It gets comments, then predicts their sentiments.
    Then it summarizes the results.
    """
    video_id = extract_video_id(video_url_or_id)
    if not video_id:
        # Give a more helpful error message to the user
        st.error(
            "Oops! That doesn't look like a valid YouTube URL or Video ID. Please check and try again. Example: Z9kGRMglw-I or youtu.be/3?v=Z9kGRMglw-I"
        )
        return None  # Stop if no valid ID

    summary_data = {}  # Initialize
    # comments_with_sentiments = []  # Initialize

    try:
        with st.spinner(f"Fetching comments & title for video ID: {video_id}..."):
            video_data = get_video_comments(video_id)
            comments_text_list = video_data.get("comments", [])
            video_title = video_data.get("title", "Video Title Not Found")
            print(
                f"DEBUG (streamlit_app.py): Received title from youtube.py: '{video_title}'"
            )

        # Check if actually got any comments
        if not comments_text_list:
            st.warning(
                "Hmm, no comments found for this video. Are comments enabled? Or is it a very new video?"
            )
            # Provide a default empty summary structure
            summary_data = {
                "num_comments_fetched": 0,
                "num_comments_analyzed": 0,
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "positive_percentage": 0,
                "neutral_percentage": 0,
                "negative_percentage": 0,
                "num_valid_predictions": 0,
            }
            return {"summary": summary_data, "comments_data": []}

        st.info(
            f"Great! Found {len(comments_text_list)} comments. Now thinking about their feelings (sentiments)..."
        )
        # Another spinner for the prediction part, as this can be slow on CPU
        with st.spinner("Analyzing sentiments with the model... Please wait."):
            # This calls predict_sentiments from predict.py
            # Expected to return: ["positive", "negative", "neutral", ...]
            prediction_results = predict_sentiments(comments_text_list)

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        error_count = 0

        for result in prediction_results:
            label = result.get("label")
            if label == "positive":
                positive_count += 1
            elif label == "negative":
                negative_count += 1
            elif label == "neutral":
                neutral_count += 1
            else:
                error_count += 1

        num_valid_predictions = positive_count + negative_count + neutral_count
        total_comments_processed = len(prediction_results)
        if error_count > 0:
            st.warning(
                f"Could not predict sentiment properly for {error_count} comments."
            )

        summary_data = {
            "video_title": video_title,
            "num_comments_fetched": len(comments_text_list),
            "num_comments_analyzed": total_comments_processed,
            "num_valid_predictions": num_valid_predictions,
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "positive_percentage": (
                (positive_count / num_valid_predictions) * 100
                if num_valid_predictions > 0
                else 0
            ),
            "neutral_percentage": (
                (neutral_count / num_valid_predictions) * 100
                if num_valid_predictions > 0
                else 0
            ),
            "negative_percentage": (
                (negative_count / num_valid_predictions) * 100
                if num_valid_predictions > 0
                else 0
            ),
        }

        comments_data_for_df = []
        for i in range(len(comments_text_list)):
            comment_text = comments_text_list[i]
            result = prediction_results[i]
            label = result.get("label", "Error")
            scores = result.get("scores", {})
            confidence = max(scores.values()) if scores else 0.0

            comments_data_for_df.append(
                {
                    "Comment Text": comment_text,
                    "Predicted Sentiment": label,
                    "Confidence": confidence,
                    # "All Scores": scores
                }
            )

        return {"summary": summary_data, "comments_data": comments_data_for_df}

    except Exception as e:
        # Show a general error if anything unexpected happens
        st.error(f"Uh oh! An error popped up during analysis: {str(e)}")
        # Also print to console for more detailed debugging when running locally
        print(f"Full error in analyze_youtube_video: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback to console
        return None  # Return None on error


# --- Streamlit App UI ---

# Page configuration: Set to centered layout (default) instead of "wide"
st.set_page_config(page_title="Social Sentiment Analysis", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #d6d6d6;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("📊 SOCIAL SENTIMENT ANALYSIS")
# A little description for the user
st.write(
    """
    Welcome to the **Social Sentiment Analyzer!** 👋

    This application uses a fine-tuned RoBERTa model to predict the sentiment (Positive, Neutral, or Negative) expressed in text.

    Use the tabs below to choose your input method:
    * **Analyze Text Input:** Paste or type any English text directly.
    * **YouTube Analysis:** Enter a YouTube video URL or ID to analyze its comments.
    * **Twitter/X Analysis:** Support for analyzing Twitter/X posts is coming soon!

    Select a tab to begin!
    """
)

# Tabs for different platforms, makes it easy to add Twitter later
tab_text_input, tab_youtube, tab_twitter = st.tabs(
    ["Analyze Text Input", "YouTube Analysis", "Twitter/X Analysis (Coming Soon!)"]
)

with tab_text_input:
    # Header for this tab
    st.header("Analyze Sentiment of Your Text")
    st.write(
        "Enter a sentence or a short paragraph below to see its predicted sentiment distribution."
    )

    # Use text_area for potentially longer input
    # Giving it a unique key helps maintain state if needed
    user_text = st.text_area(
        "Enter text here:",
        key="text_input_area_key",
        height=100,
        placeholder="Type or paste your text...",
    )

    # Button to trigger the analysis
    if st.button("Analyze Text", key="text_input_analyze_btn"):
        # Check if the user actually entered something (not just whitespace)
        if user_text and not user_text.isspace():
            # Show a spinner while processing
            with st.spinner("Analyzing your text..."):
                try:
                    # Call the prediction function from predict.py
                    # Pass the input text as a list with one element
                    prediction_results = predict_sentiments([user_text])

                    # Check if prediction was successful and returned expected format
                    if (
                        prediction_results
                        and isinstance(prediction_results, list)
                        and len(prediction_results) > 0
                    ):
                        # Get the result dictionary for the single input text
                        result = prediction_results[0]
                        predicted_label = result.get("label")
                        scores = result.get(
                            "scores"
                        )  # This should be a dict like {'negative': 0.1, ...}

                        # Make sure got a valid label and scores dictionary
                        if (
                            predicted_label
                            and scores
                            and isinstance(scores, dict)
                            and predicted_label != "Error"
                        ):

                            # Display the top predicted sentiment
                            st.subheader("Predicted Sentiment:")
                            # Using Streamlit's built-in status elements for color
                            if predicted_label == "positive":
                                st.success(
                                    f"The model thinks the sentiment is: **{predicted_label.capitalize()}** 👍"
                                )
                            elif predicted_label == "negative":
                                st.error(
                                    f"The model thinks the sentiment is: **{predicted_label.capitalize()}** 👎"
                                )
                            else:  # Neutral or potentially "Unknown" if mapping failed
                                st.info(
                                    f"The model thinks the sentiment is: **{predicted_label.capitalize()}** 😐"
                                )

                            st.write("---")  # Adding a small separator
                            st.subheader(
                                "Detailed Probabilities:"
                            )  # Subheader for this section
                            if scores and isinstance(scores, dict):
                                # Using columns here helps align the probabilities nicely
                                prob_col_neg, prob_col_neu, prob_col_pos = st.columns(3)

                                # Helper to get score safely
                                def get_score(sentiment_name):
                                    return scores.get(
                                        sentiment_name.lower(), 0.0
                                    )  # Use lowercase to be safe

                                value_font_size = "22px"
                                value_font_weight = "bold"

                                with prob_col_neg:
                                    neg_prob = get_score("negative")
                                    # Display label "Negative"
                                    st.markdown("**Negative 👎:**")
                                    # Display the probability, larger font, red color
                                    st.markdown(
                                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:red;'>{neg_prob:.1%}</p>",
                                        unsafe_allow_html=True,
                                    )

                                with prob_col_neu:
                                    neu_prob = get_score("neutral")
                                    # Display label "Neutral"
                                    st.markdown("**Neutral 😐:**")
                                    # Display the probability, larger font, grey color
                                    st.markdown(
                                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:grey;'>{neu_prob:.1%}</p>",
                                        unsafe_allow_html=True,
                                    )

                                with prob_col_pos:
                                    pos_prob = get_score("positive")
                                    # Display label "Positive"
                                    st.markdown("**Positive 👍:**")
                                    # Display the probability, larger font, green color
                                    st.markdown(
                                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:green;'>{pos_prob:.1%}</p>",
                                        unsafe_allow_html=True,
                                    )

                            else:
                                # If scores dict is missing or invalid
                                st.write("Could not retrieve probability scores.")
                            st.write("---")  # Another separator before the chart

                            # --- Display Pie Chart of Probabilities ---
                            st.subheader("Sentiment Probabilities:")
                            # Convert the scores dictionary to a DataFrame suitable for Plotly
                            # Ensure keys match class_names for consistency if possible
                            # Assuming scores keys are 'negative', 'neutral', 'positive'
                            score_items = list(scores.items())
                            if score_items:  # Check if scores dict is not empty
                                df_scores = pd.DataFrame(
                                    score_items,
                                    columns=["Sentiment", "Probability"],
                                )
                                # Convert Probability to numeric just in case
                                df_scores["Probability"] = pd.to_numeric(
                                    df_scores["Probability"]
                                )

                                # Define colors (ensure keys match Sentiment names case)
                                color_map = {
                                    "positive": "green",
                                    "neutral": "grey",
                                    "negative": "red",
                                }
                                # Make keys lowercase for robust mapping
                                df_scores["Sentiment"] = df_scores[
                                    "Sentiment"
                                ].str.capitalize()
                                df_scores["Sentiment_Lower"] = df_scores[
                                    "Sentiment"
                                ].str.lower()
                                color_map_lower = {
                                    k.lower(): v for k, v in color_map.items()
                                }

                                # Create the pie chart
                                fig_pie_text = px.pie(
                                    df_scores,
                                    values="Probability",  # Use the probability column
                                    names="Sentiment",  # Labels for the slices
                                    title="Probability Distribution per Class",
                                    color="Sentiment_Lower",  # Use lowercase for mapping
                                    color_discrete_map=color_map_lower,
                                )  # Map colors

                                # Update how text is shown on slices
                                fig_pie_text.update_traces(
                                    textposition="inside",
                                    textinfo="percent+label",
                                    hovertemplate="Sentiment: %{label}<br>Probability: %{percent}",
                                )
                                # Maybe add hover info too
                                fig_pie_text.update_layout(
                                    uniformtext_minsize=16,
                                    uniformtext_mode="hide",
                                )  # Improve text fitting

                                st.plotly_chart(fig_pie_text, use_container_width=True)

                            else:  # If scores dictionary was empty
                                st.warning("Received empty scores, cannot plot chart.")

                        else:
                            # This handles cases where predict_sentiments returned an error label
                            st.error(
                                f"Sentiment analysis failed for the input text. Result: {result}"
                            )

                    else:
                        # This handles cases where predict_sentiments returned None or empty list
                        st.error(
                            "Received no valid result from the prediction function."
                        )

                except Exception as analysis_e:
                    # Catch-all for other errors during analysis for this tab
                    st.error(
                        f"An error occurred during text analysis: {str(analysis_e)}"
                    )
                    print(f"Full error during text input analysis: {analysis_e}")
                    import traceback

                    traceback.print_exc()

        else:
            # If user clicks button without entering text
            st.warning("Please enter some text in the text area first!")

with tab_youtube:
    st.header("YouTube Comment Sentiment Analyzer")
    # Input field for URL or ID
    video_url_input = st.text_input(
        "Enter YouTube Video URL or Video ID:",
        key="youtube_url_input_key",  # Giving it a unique key
        placeholder="e.g., Z9kGRMglw-I or full URL",
    )

    # Button to trigger analysis
    if st.button("Analyze YouTube Comments", key="youtube_analyze_button_key"):
        if video_url_input:  # Check if user actually entered something
            # analyze_youtube_video handles spinners internally now
            analysis_results = analyze_youtube_video(video_url_input)

            if (
                analysis_results and analysis_results["summary"]
            ):  # Check if got valid results
                summary = analysis_results["summary"]
                comments_data = analysis_results["comments_data"]
                video_title_display = summary.get(
                    "video_title", "Video Title Not Available"
                )

                st.markdown("---")
                # Displaying the video title using markdown for potential formatting later
                st.markdown(f"### Analyzing Video: **{video_title_display}**")
                st.markdown("---")

                st.subheader("📊 Sentiment Summary")

                # Define desired font sizes (you can adjust these)
                label_font_size = "24px"
                value_font_size = "28px"  # Font size for the actual count like "137"
                value_font_weight = "bold"  # Make the count bold

                # Define colors for the sentiment counts
                positive_color = "green"
                neutral_color = "grey"
                negative_color = "red"

                # Using 5 columns
                col_fetched, col_analyzed, col_pos, col_neu, col_neg = st.columns(5)

                # Metric 1: Comments Fetched
                with col_fetched:
                    # Label for fetched comments
                    st.markdown(
                        f"<p style='font-size: {label_font_size}; margin-bottom: 0px;'>Comments Fetched</p>",
                        unsafe_allow_html=True,
                    )
                    # The number of fetched comments
                    st.markdown(
                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; margin-top: 0px;'>{summary.get('num_comments_fetched', 0)}</p>",
                        unsafe_allow_html=True,
                    )

                # Metric 2: Comments Analyzed
                with col_analyzed:
                    # Label for analyzed comments
                    st.markdown(
                        f"<p style='font-size: {label_font_size}; margin-bottom: 0px;'>Comments Analyzed</p>",
                        unsafe_allow_html=True,
                    )
                    # The number of analyzed comments
                    st.markdown(
                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; margin-top: 0px;'>{summary.get('num_comments_analyzed', 0)}</p>",
                        unsafe_allow_html=True,
                    )

                # Metric 3: Positive
                with col_pos:
                    # Label for positive comments, with emoji
                    st.markdown(
                        f"<p style='font-size: {label_font_size}; margin-bottom: 0px;'>Positive 👍</p>",
                        unsafe_allow_html=True,
                    )
                    # The count of positive comments, green and bold
                    st.markdown(
                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:{positive_color}; margin-top: 0px;'>{summary.get('positive', 0)}</p>",
                        unsafe_allow_html=True,
                    )

                # Metric 4: Neutral
                with col_neu:
                    # Label for neutral comments
                    st.markdown(
                        f"<p style='font-size: {label_font_size}; margin-bottom: 0px;'>Neutral 😐</p>",
                        unsafe_allow_html=True,
                    )
                    # The count of neutral comments, grey and bold
                    st.markdown(
                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:{neutral_color}; margin-top: 0px;'>{summary.get('neutral', 0)}</p>",
                        unsafe_allow_html=True,
                    )

                # Metric 5: Negative
                with col_neg:
                    # Label for negative comments
                    st.markdown(
                        f"<p style='font-size: {label_font_size}; margin-bottom: 0px;'>Negative 👎</p>",
                        unsafe_allow_html=True,
                    )
                    # The count of negative comments, red and bold
                    st.markdown(
                        f"<p style='font-size: {value_font_size}; font-weight: {value_font_weight}; color:{negative_color}; margin-top: 0px;'>{summary.get('negative', 0)}</p>",
                        unsafe_allow_html=True,
                    )

                # Add a visual separator before charts
                st.markdown("---")

                # Data for charts - make sure it has counts > 0
                if summary.get("num_valid_predictions", 0) > 0:
                    # Prepare DataFrame for Plotly charts
                    sentiment_data_for_plot = [
                        {"Sentiment": "Positive", "Count": summary.get("positive", 0)},
                        {"Sentiment": "Neutral", "Count": summary.get("neutral", 0)},
                        {"Sentiment": "Negative", "Count": summary.get("negative", 0)},
                    ]
                    sentiment_counts_df = pd.DataFrame(sentiment_data_for_plot)
                    # Filter out rows where Count is 0 for cleaner charts
                    sentiment_counts_df_for_plot = sentiment_counts_df[
                        sentiment_counts_df["Count"] > 0
                    ].copy()

                    # Define the color map for charts
                    # Keys should match the 'Sentiment' column values
                    color_map = {
                        "Positive": "green",
                        "Neutral": "grey",
                        "Negative": "red",
                    }

                    if not sentiment_counts_df_for_plot.empty:
                        st.subheader("📈 Sentiment Distribution Charts")
                        # Pie Chart (Corrected data input for Plotly)
                        # Plotly pie chart expects a DataFrame where one column is values, another is names
                        fig_pie = px.pie(
                            sentiment_counts_df_for_plot,  # Use the filtered DataFrame
                            values="Count",  # Column for pie slice values
                            names="Sentiment",  # Column for pie slice names
                            title="Pie Chart: Comment Sentiments",
                            color="Sentiment",  # Color slices based on the 'Sentiment' category
                            color_discrete_map=color_map,
                        )  # Apply custom colors

                        fig_pie.update_traces(
                            textposition="inside",
                            textinfo="percent+label",
                            hovertemplate="Sentiment: %{label}<br>Count: %{value}<br>Percentage: %{percent}",
                        )

                        fig_pie.update_layout(
                            uniformtext_minsize=16, uniformtext_mode="hide"
                        )

                        st.plotly_chart(fig_pie, use_container_width=True)

                        # Bar Chart (Using Plotly for consistent coloring)
                        fig_bar = px.bar(
                            sentiment_counts_df_for_plot,  # Use the filtered DataFrame
                            x="Sentiment",  # Categories on X-axis
                            y="Count",  # Values on Y-axis
                            title="Bar Chart: Comment Sentiments",
                            color="Sentiment",  # Color bars based on 'Sentiment'
                            color_discrete_map=color_map,  # Apply custom colors
                            labels={
                                "Count": "Number of Comments",
                                "Sentiment": "Sentiment Category",
                            },
                        )  # Custom labels
                        st.plotly_chart(fig_bar, use_container_width=True)

                    else:
                        # This message shows if all sentiment counts are zero
                        st.write(
                            "No sentiment data (Positive, Neutral, Negative all zero) to display in charts."
                        )
                else:
                    # This message shows if no comments were analyzed successfully
                    st.write(
                        "Not enough valid sentiment data to display distribution charts."
                    )

                # Display comments and their sentiments
                if comments_data:
                    st.subheader(
                        f"🔍 Analyzed Comments (showing first {len(comments_data)} results)"
                    )
                    comments_display_df = pd.DataFrame(comments_data)

                    if "Confidence" in comments_display_df.columns:
                        try:
                            # Format as percentage with 1 decimal place
                            comments_display_df["Confidence"] = comments_display_df[
                                "Confidence"
                            ].map("{:.1%}".format)
                        except (TypeError, ValueError):
                            st.warning(
                                "Could not format confidence scores."
                            )  # Handle potential errors if confidence is not numeric

                    st.dataframe(
                        comments_display_df, use_container_width=True, height=400
                    )
                else:
                    st.write("No comments were analyzed to display.")
        else:
            # If user clicks button without entering URL
            st.warning("Please enter a YouTube URL or Video ID first!")

with tab_twitter:
    st.header("Twitter/X Post Analysis")
    st.info("This feature is currently under construction. Please check back later!")
    # Placeholder for future Twitter input
    # twitter_url_input = st.text_input("Enter Twitter/X Post URL:", key="twitter_url_input_key")
    # if st.button("Analyze Tweets", key="twitter_analyze_button_key"):
    #     st.write("Imagine amazing Twitter analysis happening here... Tweet tweet!")
