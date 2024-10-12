import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import jovian


st.set_page_config(
    page_title="Spotify Streaming History EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title of the app
st.title("🎧 Spotify Streaming History: Exploratory Data Analysis")

# Sidebar for user inputs
st.sidebar.header("Upload Your Spotify Streaming History")

# File uploader allows users to upload their Spotify data
uploaded_file = st.sidebar.file_uploader(
    "Upload your Spotify Streaming History JSON or CSV file",
    type=["json", "csv"]
)

# Sidebar option for delimiter (only applicable for CSV)
delimiter_option = None
custom_delimiter = None

if uploaded_file is not None and uploaded_file.name.endswith('.csv'):
    delimiter_option = st.sidebar.selectbox(
        "Select CSV Delimiter",
        ("Comma (,)", "Tab (\\t)", "Semicolon (;)", "Other")
    )

    # Input for custom delimiter
    if delimiter_option == "Other":
        custom_delimiter = st.sidebar.text_input("Enter custom delimiter", value=",")
    else:
        delimiter_mapping = {
            "Comma (,)": ",",
            "Tab (\\t)": "\t",
            "Semicolon (;)": ";"
        }
        custom_delimiter = delimiter_mapping[delimiter_option]

# Function to load data
@st.cache_data(show_spinner=False)
def load_data(file, delimiter=None):
    if file is not None:
        try:
            if file.name.endswith('.json'):
                df = pd.read_json(file)
                st.success("Successfully loaded JSON file!")
            elif file.name.endswith('.csv'):
                if delimiter:
                    df = pd.read_csv(file, sep=delimiter)
                else:
                    df = pd.read_csv(file)
                st.success("Successfully loaded CSV file!")
            else:
                st.error("Unsupported file type.")
                return None
            return df
        except pd.errors.EmptyDataError:
            st.error("No columns to parse from file. Please check the file format and delimiter.")
            return None
        except pd.errors.ParserError as e:
            st.error(f"Error parsing file: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return None
    return None

# Load data with selected delimiter
df = load_data(uploaded_file, custom_delimiter)

if df is not None:
    st.success("Data successfully loaded!")

    # Display raw data
    if st.checkbox("Show Raw Data"):
        st.subheader("Raw Spotify Streaming Data")
        st.write(df.head(10))

    # Data Cleaning and Formatting
    st.subheader("🔧 Data Cleaning and Formatting")

    # Identify the timestamp column
    timestamp_cols = ['endTime', 'Play Time', 'timestamp', 'dateTime']
    timestamp_col = None
    for col in timestamp_cols:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        df["Play-Time"] = pd.to_datetime(df[timestamp_col])
    else:
        st.error("Timestamp column not found. Please ensure your data has a timestamp column named 'endTime' or 'Play Time'.")
        st.stop()

    # Extract date components
    df['year'] = df['Play-Time'].dt.year
    df['month'] = df['Play-Time'].dt.month
    df['day'] = df['Play-Time'].dt.day
    df['weekday'] = df['Play-Time'].dt.weekday
    df['time'] = df['Play-Time'].dt.time
    df['hour'] = df['Play-Time'].dt.hour
    df['day_name'] = df['Play-Time'].dt.day_name()
    df['count'] = 1  # For counting purposes

    # Convert msPlayed to timedelta if available
    duration_cols = ['msPlayed', 'Duration_ms', 'duration_ms']
    duration_col = None
    for col in duration_cols:
        if col in df.columns:
            duration_col = col
            break

    if duration_col:
        df["Time-Played (hh-mm-ss)"] = pd.to_timedelta(df[duration_col], unit='ms')
    else:
        st.warning("Duration column not found. Skipping duration conversions.")

    # Calculate listening time in hours and minutes
    def hours(td):
        return td.seconds / 3600 + td.days * 24

    def minutes(td):
        return (td.seconds / 60) % 60 + td.days * 24 * 60

    if "Time-Played (hh-mm-ss)" in df.columns:
        df["Listening Time (Hours)"] = df["Time-Played (hh-mm-ss)"].apply(hours).round(3)
        df["Listening Time (Minutes)"] = df["Time-Played (hh-mm-ss)"].apply(minutes).round(3)

    # Drop unnecessary columns
    columns_to_drop = [timestamp_col, "Time-Played (hh-mm-ss)", duration_col] if duration_col else [timestamp_col]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    st.write("### Cleaned DataFrame")
    st.write(df.head(10))

    # Data Overview
    st.subheader("📊 Data Overview")
    st.write(df.describe())

    # Visualizations
    st.subheader("📈 Exploratory Analysis and Visualization")

    # Setting up visualization styles
    sns.set_style('darkgrid')  # Seaborn styling
    # Removed plt.style.use('seaborn-darkgrid') to avoid the OSError
    plt.rcParams['font.size'] = 14
    plt.rcParams['figure.figsize'] = (9, 5)
    plt.rcParams['figure.facecolor'] = '#FFFFFF'  # White background

    # Sidebar for visualization options
    st.sidebar.header("Visualization Options")
    viz_option = st.sidebar.selectbox(
        "Select Visualization",
        ("Artist Analysis", "Track Analysis", "Day-wise Usage", "Hourly Usage", "Listening Time Stats")
    )

    if viz_option == "Artist Analysis":
        st.subheader("🎤 Artist Analysis")

        # Unique Artists Percentage
        unique_artists = df["artistName"].nunique()
        total_artists = df["artistName"].count()
        unique_artist_percentage = unique_artists / total_artists * 100

        st.write(f"**Unique Artists:** {unique_artists} out of {total_artists} plays ({unique_artist_percentage:.2f}%)")

        # Pie Chart for Unique Artists
        unique_artist_list = np.array([unique_artists, total_artists - unique_artists])
        unique_artist_labels = ["Unique Artists", "Non-Unique Artists"]

        fig1, ax1 = plt.subplots()
        ax1.pie(
            unique_artist_list,
            labels=unique_artist_labels,
            autopct='%1.1f%%',
            explode=[0.05, 0.05],
            startangle=180,
            shadow=True
        )
        ax1.set_title("Unique Artist Percentage")
        st.pyplot(fig1)

        # Top 10 Artists by Listening Time
        st.write("### Top 10 Artists by Listening Time (Hours)")
        top_artists_time = df.groupby("artistName")[["Listening Time (Hours)", "Listening Time (Minutes)", "count"]].sum().sort_values(by="Listening Time (Minutes)", ascending=False).head(10)
        st.write(top_artists_time)

        # Bar Chart for Top 10 Artists (Hours)
        fig2, ax2 = plt.subplots(figsize=(12,8))
        sns.barplot(
            x=top_artists_time.index,
            y="Listening Time (Hours)",
            data=top_artists_time,
            palette="Greens_d"
        )
        ax2.set_title("Top 10 Artists by Listening Time (Hours)")
        ax2.set_xlabel("Artists")
        ax2.set_ylabel("Hours")
        plt.xticks(rotation=75)
        st.pyplot(fig2)

        # Top 10 Artists by Count
        st.write("### Top 10 Artists by Play Count")
        top_artists_count = df.groupby("artistName")[["count"]].count().sort_values(by="count", ascending=False).head(10)
        st.write(top_artists_count)

        # Bar Chart for Top 10 Artists (Count)
        fig3, ax3 = plt.subplots(figsize=(12,8))
        sns.barplot(
            x=top_artists_count.index,
            y="count",
            data=top_artists_count,
            palette="Oranges_d"
        )
        ax3.set_title("Top 10 Artists by Play Count")
        ax3.set_xlabel("Artists")
        ax3.set_ylabel("Play Count")
        plt.xticks(rotation=75)
        st.pyplot(fig3)

    elif viz_option == "Track Analysis":
        st.subheader("🎶 Track Analysis")

        # Unique Tracks Percentage
        unique_tracks = df["trackName"].nunique()
        total_tracks = df["trackName"].count()
        unique_tracks_percentage = unique_tracks / total_tracks * 100

        st.write(f"**Unique Tracks:** {unique_tracks} out of {total_tracks} plays ({unique_tracks_percentage:.2f}%)")

        # Pie Chart for Unique Tracks
        unique_tracks_list = np.array([unique_tracks, total_tracks - unique_tracks])
        unique_tracks_labels = ["Unique Tracks", "Non-Unique Tracks"]

        fig4, ax4 = plt.subplots()
        ax4.pie(
            unique_tracks_list,
            labels=unique_tracks_labels,
            autopct='%1.1f%%',
            explode=[0.05, 0.05],
            startangle=180,
            shadow=True
        )
        ax4.set_title("Unique Tracks Percentage")
        st.pyplot(fig4)

        # Top 10 Tracks by Listening Time
        st.write("### Top 10 Tracks by Listening Time (Hours)")
        top_tracks_time = df.groupby("trackName")[["Listening Time (Hours)", "Listening Time (Minutes)", "count"]].sum().sort_values(by="Listening Time (Minutes)", ascending=False).head(10)
        st.write(top_tracks_time)

        # Bar Chart for Top 10 Tracks (Hours)
        fig5, ax5 = plt.subplots(figsize=(12,8))
        sns.barplot(
            x=top_tracks_time.index,
            y="Listening Time (Hours)",
            data=top_tracks_time,
            palette="Blues_d"
        )
        ax5.set_title("Top 10 Tracks by Listening Time (Hours)")
        ax5.set_xlabel("Tracks")
        ax5.set_ylabel("Hours")
        plt.xticks(rotation=75)
        st.pyplot(fig5)

        # Top 10 Tracks by Count
        st.write("### Top 10 Tracks by Play Count")
        top_tracks_count = df.groupby("trackName")[["count"]].count().sort_values(by="count", ascending=False).head(10)
        st.write(top_tracks_count)

        # Bar Chart for Top 10 Tracks (Count)
        fig6, ax6 = plt.subplots(figsize=(12,8))
        sns.barplot(
            x=top_tracks_count.index,
            y="count",
            data=top_tracks_count,
            palette="Purples_d"
        )
        ax6.set_title("Top 10 Tracks by Play Count")
        ax6.set_xlabel("Tracks")
        ax6.set_ylabel("Play Count")
        plt.xticks(rotation=75)
        st.pyplot(fig6)

    elif viz_option == "Day-wise Usage":
        st.subheader("📅 Day-wise Spotify Usage")

        # Pie Chart for Day-wise Usage
        day_counts = df["day_name"].value_counts()

        fig7, ax7 = plt.subplots()
        ax7.pie(
            day_counts,
            labels=day_counts.index,
            autopct='%1.1f%%',
            startangle=180,
            shadow=True
        )
        ax7.set_title("Spotify Usage by Day of the Week")
        st.pyplot(fig7)

        # Word Cloud for Days
        wordcloud_days = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(day_counts)
        fig8, ax8 = plt.subplots(figsize=(15, 7.5))
        ax8.imshow(wordcloud_days, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title("Word Cloud of Days Based on Play Count")
        st.pyplot(fig8)

    elif viz_option == "Hourly Usage":
        st.subheader("⏰ Hourly Spotify Usage")

        # Histogram for Hourly Usage
        fig9, ax9 = plt.subplots(figsize=(12,8))
        sns.histplot(
            df["hour"],
            bins=24,
            kde=True,
            color="darkgreen",
            ax=ax9
        )
        ax9.set_title("Average Distribution of Streaming Over Hours")
        ax9.set_xlabel("Hour of the Day (24-hour format)")
        ax9.set_ylabel("Number of Songs Played")
        st.pyplot(fig9)

    elif viz_option == "Listening Time Stats":
        st.subheader("📈 Listening Time Statistics")

        # Total Listening Time
        total_hours = df["Listening Time (Hours)"].sum()
        st.write(f"**Total Listening Time:** {total_hours:.2f} Hours")

        # Percentage of Listening Time
        date_df = df["Play-Time"]
        if len(date_df) > 1:
            time_difference_days = (date_df.iloc[-1] - date_df.iloc[0]) / np.timedelta64(1, "D")
            time_difference_hours = time_difference_days * 24
            listening_percentage = (total_hours / time_difference_hours) * 100
            st.write(f"**Listening Time Percentage:** {listening_percentage:.2f}% of possible listening hours")
        else:
            st.warning("Not enough data to calculate listening time percentage.")
            time_difference_hours = None

        # Pie Chart for Listening Time Percentage
        if time_difference_hours:
            listening_time_list = np.array([total_hours, time_difference_hours - total_hours])
            listening_time_labels = ["Actual Hours Spent", "Possible Hours"]

            fig10, ax10 = plt.subplots()
            ax10.pie(
                listening_time_list,
                labels=listening_time_labels,
                autopct='%1.1f%%',
                explode=[0.2, 0.2],
                startangle=180,
                shadow=True
            )
            ax10.set_title("Listening Time Percentage")
            st.pyplot(fig10)

        # Average Songs Played Daily
        if time_difference_hours:
            average_songs_daily = (df["trackName"].count() / time_difference_days).round()
            st.write(f"**Average Songs Played Daily:** {average_songs_daily} songs/day")

        # Scatter Plot for Maximum Songs Played in a Day
        df['date'] = df['Play-Time'].dt.date
        songs_per_day = df.groupby("date")["count"].sum().sort_values(ascending=False)
        if not songs_per_day.empty:
            max_songs_day = songs_per_day.head(1)
            st.write(f"**Maximum Songs Played in a Day:** {max_songs_day.values[0]} songs on {max_songs_day.index[0]}")

            fig11, ax11 = plt.subplots(figsize=(15,8))
            sns.scatterplot(
                x=songs_per_day.index,
                y=songs_per_day.values,
                ax=ax11,
                color='blue',
                alpha=0.6
            )
            ax11.axhline(songs_per_day.mean(), linestyle='--', color='red', label='Average Songs Played')
            ax11.set_title("Maximum Number of Songs Played in a Day")
            ax11.set_xlabel("Date")
            ax11.set_ylabel("Number of Songs Played")
            plt.xticks(rotation=90)
            plt.legend()
            st.pyplot(fig11)
        else:
            st.warning("No data available for songs per day.")

    # Additional Visualizations (e.g., Word Cloud for Artists)
    st.subheader("🌟 Additional Insights")

    if st.checkbox("Show Word Cloud of Top Artists by Play Count"):
        st.write("### Word Cloud of Top 100 Artists by Play Count")
        top_artists_wc = df['artistName'].value_counts().head(100)
        wordcloud_artists = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_artists_wc)

        fig12, ax12 = plt.subplots(figsize=(15,7.5))
        ax12.imshow(wordcloud_artists, interpolation='bilinear')
        ax12.axis('off')
        ax12.set_title("Word Cloud of Top 100 Artists by Play Count")
        st.pyplot(fig12)


else:
    st.warning("Please upload your Spotify Streaming History JSON or CSV file to begin the analysis.")
