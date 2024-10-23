import streamlit as st
import pickle
import numpy as np
import pandas as pd  
import plotly.express as px

# Function to load the model
def load_model():
    with open('model/gpa_updated.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess input before passing to the model
def preprocess_input(Age, StudyTimeWeekly, Absences, ParentalSupport, GradeClass, Extracurricular, Sports, Music, Volunteering, Tutoring, OSmarks, DBMSmarks, MLmarks, Pythonmarks):
    Extracurricular = 1 if Extracurricular == "Yes" else 0
    Sports = 1 if Sports == "Yes" else 0
    Music = 1 if Music == "Yes" else 0
    Volunteering = 1 if Volunteering == "Yes" else 0
    Tutoring = 1 if Tutoring == "Yes" else 0

    # Create an array of features for prediction
    features = np.array([[Age, StudyTimeWeekly, Absences, ParentalSupport, GradeClass, Extracurricular, Sports, Music, Volunteering, Tutoring, OSmarks, DBMSmarks, MLmarks, Pythonmarks]])
    return features

def data_chart(std_name, OverallGPA):
    # Create a DataFrame for the input data, including GPA
    data = {
        "Feature": ["Predicted GPA"],
        "Value": [OverallGPA]
    }
    
    df = pd.DataFrame(data)

    # Create a bar chart using Plotly Express
    fig = px.bar(df, x="Feature", y="Value", title=f"{std_name}'s GPA Prediction", text="Value")
    fig.update_traces(marker=dict(line=dict(width=2))) 
    fig.update_layout(bargap=0.7)

    return fig

# Main function for the Streamlit app
def main():
    st.set_page_config(
        page_title="GPA PREDICTION",
        page_icon="💯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load model
    model = load_model()

    # Page title and description
    st.title("💯 GPA Prediction App")
    st.subheader("Welcome to the GPA Prediction App!")
    st.markdown("""
    ### Grading System
    **Marks >= 90**: Grade 4.0 (A)  
    **Marks >= 80**: Grade 3.0 (B)  
    **Marks >= 70**: Grade 2.0 (C)  
    **Marks >= 60**: Grade 1.0 (D)  
    **Marks < 60**: Grade 0.0 (F)  
    """)

    
    # Background image styling
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://i.postimg.cc/L5xDBvbw/Screenshot-2024-10-23-225504.png");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Input form for student details
    with st.container():
        # Create columns for layout
        buff, col, buff2 = st.columns([1, 3, 1])

        # Form starts here
        with col.form("std_form"):
            std_name = st.text_input("Enter the student's name:")
            Age = st.number_input("Enter the student's age (min value=16, max value=19)", min_value=16, max_value=19, step=1)
            StudyTimeWeekly = st.number_input("Enter the student's study hours (min value=0hrs, max value=21hrs):", min_value=0, max_value=21, step=1)
            Absences = st.number_input("Enter the student's absence (min value=0 days, max value=30 days):", min_value=0, max_value=30, step=1)
            ParentalSupport = st.number_input("Enter the student's parental support (min value=0, max value=4):", min_value=0, max_value=4, step=1)
            GradeClass = st.number_input("Enter the student's Grade class (min value=0, max value=4):", min_value=0, max_value=4, step=1)

            # Additional inputs for marks and GPAs
            OSmarks = st.number_input("Enter Operating system marks:", min_value=0, max_value=100, step=1)
            DBMSmarks = st.number_input("Enter Database management marks:", min_value=0, max_value=100, step=1)
            MLmarks = st.number_input("Enter Machine learning marks:", min_value=0, max_value=100, step=1)
            Pythonmarks = st.number_input("Enter Python marks:", min_value=0, max_value=100, step=1)

            # Create columns for radio buttons
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                Extracurricular = st.radio("Extracurricular", ["Yes", "No"])
            with col2:
                Sports = st.radio("Sports", ["Yes", "No"])
            with col3:
                Music = st.radio("Music", ["Yes", "No"])
            with col4:
                Volunteering = st.radio("Volunteering", ["Yes", "No"])
            with col5:
                Tutoring = st.radio("Tutoring", ["Yes", "No"])

            # Form submission button
            submit_button = st.form_submit_button("Submit")

        if submit_button:
            if std_name == "":
                st.warning("Your input is empty")
            else:
                features = preprocess_input(Age, StudyTimeWeekly, Absences, ParentalSupport, GradeClass, Extracurricular, Sports, Music, Volunteering, Tutoring, OSmarks, DBMSmarks, MLmarks, Pythonmarks)

                try:
                    # Predict GPA using the model
                    predicted_gpa = model.predict(features)[0]

                    # Ensure GPA is not negative
                    predicted_gpa = max(predicted_gpa, 0)

                    # Display the predicted value
                    st.success(f"Predicted GPA for {std_name}: {predicted_gpa:.2f}")

                    # Prepare data for download
                    report_data = f"""
                    Student's Name: {std_name}
                    Age: {Age}
                    Study Time Weekly: {StudyTimeWeekly} hours
                    Absence: {Absences} days
                    Parental Support: {ParentalSupport}
                    Grade Class: {GradeClass}
                    Extracurricular: {Extracurricular}
                    Sports: {Sports}
                    Music: {Music}
                    Volunteering: {Volunteering}
                    Tutoring: {Tutoring}
                    Operating system marks: {OSmarks}
                    Database Management marks: {DBMSmarks}
                    Machine Learning marks: {MLmarks}
                    Python marks: {Pythonmarks}
                    Predicted GPA: {predicted_gpa:.2f}
                    """

                    # Download button
                    st.download_button(
                        label="Download Report",
                        data=report_data,
                        file_name=f"{std_name}_report.txt",
                        mime="text/plain"
                    )

                    # Generate and display data chart
                    chart = data_chart(std_name, predicted_gpa)
                    st.plotly_chart(chart)

                except ValueError as e:
                    st.error(f"Prediction error: {e}")

if __name__ == '__main__':
    main()



