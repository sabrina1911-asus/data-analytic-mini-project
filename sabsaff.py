import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Display image at the top
st.image("students image.jpg", caption="Students in Extracurricular Activities", use_container_width=True)

# Load cleaned dataset
df = pd.read_csv("anissabrinacleaned_data.csv")

# Clean missing values in Activity column
df["Activity"] = df["Activity"].fillna("unknown")

# Define intensity extraction safely and case-insensitive
def get_intensity(activity):
    activity = str(activity).lower()
    if "high" in activity:
        return "High"
    elif "low" in activity:
        return "Low"
    else:
        return "Medium"

# Create intensity level column if not already there
if "Intensity_Level" not in df.columns:
    df["Intensity_Level"] = df["Activity"].apply(get_intensity)

# Sidebar filters
st.sidebar.title("🎓 Dashboard Settings")
activity_types = df['Activity'].unique()

# Optionally exclude 'unknown' by default in filters:
default_activities = [act for act in activity_types if act != "unknown"]
selected_activities = st.sidebar.multiselect("Select Activity Type:", activity_types, default=default_activities)

intensity_levels = df['Intensity_Level'].unique()
selected_intensity = st.sidebar.multiselect("Select Intensity Level:", intensity_levels, default=intensity_levels)

# Filtered DataFrame
filtered_df = df[
    (df['Activity'].isin(selected_activities)) &
    (df['Intensity_Level'].isin(selected_intensity))
]

# Warn if no data
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust your selection.")
else:
    # Title
    st.title("📊 Impact of Extracurricular Activities on Academic Performance")

    # Objective 1
    st.header("🎯 Objective 1: Activity Type vs GPA")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average GPA by Activity")
        avg_gpa = filtered_df.groupby('Activity')['GPA'].mean().sort_values()
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        sns.barplot(x=avg_gpa.values, y=avg_gpa.index, palette="Blues_d", ax=ax1)
        ax1.set_xlabel("Average GPA")
        ax1.set_ylabel("Activity")
        ax1.set_title("Average GPA by Activity")
        st.pyplot(fig1)

    with col2:
        st.subheader("GPA Share by Activity")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(avg_gpa, labels=avg_gpa.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
        ax2.set_title("GPA Share by Activity")
        st.pyplot(fig2)

    # Objective 2
    st.header("🎯 Objective 2: Intensity Level vs Wellbeing")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Wellbeing Score by Involvement")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='Intensity_Level', y='Well-being', data=filtered_df, palette='cool', ax=ax3,
                    order=["Low", "Medium", "High"])
        ax3.set_xlabel("Intensity Level")
        ax3.set_ylabel("Well-being Score")
        ax3.set_title("Wellbeing Score by Intensity Level")
        st.pyplot(fig3)

    with col4:
        st.subheader("GPA by Involvement")
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        sns.boxplot(x='Intensity_Level', y='GPA', data=filtered_df, palette='pastel', ax=ax4,
                    order=["Low", "Medium", "High"])
        ax4.set_xlabel("Intensity Level")
        ax4.set_ylabel("GPA")
        ax4.set_title("GPA by Intensity Level")
        st.pyplot(fig4)

    # Objective 3
    st.header("🎯 Objective 3: Best Activities for Wellbeing")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Average Wellbeing Score by Activity")
        df_avg_wellbeing = filtered_df.groupby("Activity")["Well-being"].mean().reset_index().sort_values(by='Well-being', ascending=False)
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        sns.barplot(data=df_avg_wellbeing, x="Well-being", y="Activity", palette="coolwarm", ax=ax5)
        ax5.set_xlabel("Average Well-being Score")
        ax5.set_ylabel("Activity")
        ax5.set_title("Average Wellbeing Score by Activity")
        st.pyplot(fig5)

    with col6:
        st.subheader("Top 3 Beneficial Activities")
        st.table(df_avg_wellbeing.head(3))

    # Bonus: Correlation
    st.header("📌 GPA vs Wellbeing Correlation")
    correlation = filtered_df['GPA'].corr(filtered_df['Well-being'])
    st.markdown(f"**Correlation Coefficient:** `{correlation:.2f}`")

    try:
        import statsmodels.api as sm  # Required for trendline='ols'
        fig6 = px.scatter(
            filtered_df,
            x='GPA',
            y='Well-being',
            color='Activity',
            title='GPA vs Wellbeing with Trendline',
            trendline='ols'
        )
        st.plotly_chart(fig6, use_container_width=True)
    except ModuleNotFoundError:
        st.error("The 'statsmodels' package is required to show the trendline. Please add it to your requirements.txt file: \n\nstatsmodels")
