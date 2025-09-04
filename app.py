import streamlit as st
import cv2
import face_recognition as fr
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import io
import time
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Live Attendance System",
    page_icon="📸",
    layout="wide"
)

# --- CACHED DATA LOADING (SUPER DEBUG VERSION) ---
@st.cache_data
def load_all_data():
    """Loads encodings and the student database, printing debug info to the terminal."""
    print("\n--- DEBUG: Starting data loading process ---")
    
    # --- 1. Check for encodings.pkl ---
    encodings_path = "encodings.pkl"
    print(f"DEBUG: Looking for '{encodings_path}' at full path: {os.path.abspath(encodings_path)}")
    
    if not os.path.exists(encodings_path):
        print(">>> FATAL ERROR: 'encodings.pkl' NOT FOUND. <<<")
        st.error(f"❌ '{encodings_path}' not found. Please check your terminal for the full path.")
        return None, None, None
    try:
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
        print("--- DEBUG: 'encodings.pkl' loaded successfully. ---")
    except Exception as e:
        print(f">>> FATAL ERROR: Failed to read 'encodings.pkl'. Details: {e} <<<")
        st.error(f"❌ Error reading 'encodings.pkl': {e}")
        return None, None, None

    # --- 2. Check for <student_list_name> ---
    excel_path = "<student_list_name>"
    print(f"DEBUG: Looking for '{excel_path}' at full path: {os.path.abspath(excel_path)}")

    if not os.path.exists(excel_path):
        print(">>> FATAL ERROR: '<student_list_name>' NOT FOUND. <<<")
        st.error(f"❌ '{excel_path}' not found. Please check your terminal for the full path.")
        return None, None, None
    try:
        df = pd.read_excel(excel_path)
        if 'Roll_Number' not in df.columns:
            print(">>> FATAL ERROR: 'Roll_Number' column missing from Excel file. <<<")
            st.error("❌ '<student_list_name>' must have a column with the header 'Roll_Number'.")
            return None, None, None
        print("--- DEBUG: '<student_list_name>' loaded successfully. ---")
    except Exception as e:
        print(f">>> FATAL ERROR: Failed to read '<student_list_name>'. Details: {e} <<<")
        st.error(f"❌ Error reading '<student_list_name>': {e}")
        return None, None, None
        
    print("--- DEBUG: Data loading process finished successfully. ---\n")
    return known_encodings, known_names, df

# --- SESSION STATE INITIALIZATION ---
def initialize_session_state():
    """Initializes session state variables."""
    if 'attendance_df' not in st.session_state:
        known_encodings, known_names_from_files, df_from_excel = load_all_data()

        if df_from_excel is not None and known_names_from_files is not None:
            st.session_state.known_encodings = known_encodings
            st.session_state.known_names = known_names_from_files
            
            name_map = {name_roll.split('_')[-1]: '_'.join(name_roll.split('_')[:-1]) for name_roll in known_names_from_files}
            df_from_excel['Roll_Number'] = df_from_excel['Roll_Number'].astype(str)
            df_from_excel['Name'] = df_from_excel['Roll_Number'].map(name_map).fillna('N/A - No Image Found')
            
            today_str = datetime.now().strftime("%Y-%m-%d")
            if today_str not in df_from_excel.columns:
                df_from_excel[today_str] = "Absent"
            
            cols = ['Roll_Number', 'Name'] + [col for col in df_from_excel.columns if col not in ['Roll_Number', 'Name']]
            st.session_state.attendance_df = df_from_excel[cols]
        else:
            st.session_state.attendance_df = pd.DataFrame()
            
    if 'run_attendance' not in st.session_state:
        st.session_state.run_attendance = False
    
    if 'last_recognized_info' not in st.session_state:
        st.session_state.last_recognized_info = None

# --- LIVE ATTENDANCE FUNCTION ---
def run_live_attendance():
    """
    Captures video, performs face recognition, and displays a confirmation
    box for 2 seconds upon successful recognition.
    """
    video_placeholder = st.empty()
    table_placeholder = st.empty()
    
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        st.error("Could not open video capture. Please check your webcam connection.")
        return

    frame_count = 0
    today_col = datetime.now().strftime("%Y-%m-%d")
    CONFIRMATION_TIME_SECONDS = 2.0  # Display confirmation for 2 seconds

    while st.session_state.run_attendance:
        ret, frame = videocapture.read()
        if not ret:
            st.warning("Could not read frame from camera. Stopping.")
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Face Recognition (every 3rd frame for performance) ---
        if frame_count % 3 == 0:
            live_face_locations = fr.face_locations(rgb_frame)
            live_face_encodings = fr.face_encodings(rgb_frame, live_face_locations)

            for face_encoding, face_location in zip(live_face_encodings, live_face_locations):
                matches = fr.compare_faces(st.session_state.known_encodings, face_encoding, tolerance=0.5)
                face_distances = fr.face_distance(st.session_state.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    full_name_id = st.session_state.known_names[best_match_index]
                    roll_number = full_name_id.split('_')[-1]
                    
                    # Mark present in the dataframe
                    st.session_state.attendance_df.loc[st.session_state.attendance_df['Roll_Number'] == roll_number, today_col] = "Present"
                    
                    # Store recognition info in session state
                    st.session_state.last_recognized_info = {
                        "roll_number": roll_number,
                        "location": face_location,
                        "timestamp": time.time()
                    }

        # Drawing logic runs on every frame to show the persistent box.
        if st.session_state.last_recognized_info:
            info = st.session_state.last_recognized_info
            # Check if the 2-second confirmation window is still active
            if time.time() - info["timestamp"] < CONFIRMATION_TIME_SECONDS:
                top, right, bottom, left = info["location"]
                # Draw green box and display the roll number
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, info["roll_number"], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # If time is up, clear the info
                st.session_state.last_recognized_info = None

        frame_count += 1
        video_placeholder.image(frame, channels="BGR")
        table_placeholder.dataframe(st.session_state.attendance_df, use_container_width=True)
        
        # Small sleep to prevent high CPU usage, can be adjusted
        time.sleep(0.01)

    # Cleanup
    videocapture.release()
    cv2.destroyAllWindows()
    video_placeholder.empty()
    table_placeholder.empty()
    st.info("Live attendance has stopped.")


# --- MAIN APP LAYOUT ---
st.title("Live Attendance System 📸")
initialize_session_state()

if not st.session_state.attendance_df.empty:
    # Sidebar for manual controls
    st.sidebar.header("Manual Controls")
    with st.sidebar.form("manual_present_form"):
        present_roll = st.text_input("Enter Roll Number to Mark PRESENT")
        submit_present = st.form_submit_button("Mark Present")
        if submit_present and present_roll:
            today_col = datetime.now().strftime("%Y-%m-%d")
            if str(present_roll) in st.session_state.attendance_df['Roll_Number'].values:
                st.session_state.attendance_df.loc[st.session_state.attendance_df['Roll_Number'] == str(present_roll), today_col] = "Present"
                st.success(f"✅ Marked {present_roll} as Present.")
                # REMOVED: st.rerun() - The form submission handles the rerun automatically.
            else:
                st.error(f"❌ Roll Number {present_roll} not found.")

    with st.sidebar.form("manual_absent_form"):
        absent_roll = st.text_input("Enter Roll Number to Mark ABSENT")
        submit_absent = st.form_submit_button("Mark Absent")
        if submit_absent and absent_roll:
            today_col = datetime.now().strftime("%Y-%m-%d")
            if str(absent_roll) in st.session_state.attendance_df['Roll_Number'].values:
                st.session_state.attendance_df.loc[st.session_state.attendance_df['Roll_Number'] == str(absent_roll), today_col] = "Absent"
                st.success(f"✔️ Marked {absent_roll} as Absent.")
                # REMOVED: st.rerun() - The form submission handles the rerun automatically.
            else:
                st.error(f"❌ Roll Number {absent_roll} not found.")

    # Main content area
    st.header("Live Video Feed & Attendance Record")
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("Start Attendance")
    with col2:
        stop_button = st.button("Stop and Save Attendance")

    if start_button:
        st.session_state.run_attendance = True
        st.rerun() # This rerun is necessary to immediately start the camera loop

    if stop_button:
        st.session_state.run_attendance = False
        df_to_save = st.session_state.attendance_df.copy()
        df_to_save.drop(columns=['Name'], inplace=True)
        try:
            df_to_save.to_excel("<student_list_name>", index=False)
            st.success("✅ Master attendance sheet ('<student_list_name>') has been updated.")
        except Exception as e:
            st.error(f"Could not save the file. Error: {e}")

        today_col = datetime.now().strftime("%Y-%m-%d")
        present_df = st.session_state.attendance_df[st.session_state.attendance_df[today_col] == "Present"]
        if not present_df.empty:
            output = io.BytesIO()
            present_df_to_export = present_df[['Roll_Number', 'Name']]
            present_df_to_export.to_excel(output, index=False, sheet_name='PresentStudents')
            output.seek(0)
            st.download_button(
                label="📥 Download List of Present Students",
                data=output,
                file_name=f"present_students_{today_col}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No students were marked as 'Present' to export.")
        st.rerun() # This rerun is necessary to show the static dataframe after stopping

    # Conditional execution of the live feed
    if st.session_state.get('run_attendance', False):
        run_live_attendance()
    else:
        # Show the static dataframe when not running
        st.dataframe(st.session_state.attendance_df, use_container_width=True)

else:

    st.error("Application cannot start. Please fix the file errors shown above and check the terminal for more details.")
