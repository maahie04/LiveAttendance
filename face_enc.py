import cv2
import face_recognition as fr
import pickle
import os

def encode_student_images():
    """
    Scans a directory of student images, creates face encodings,
    and saves them to a pickle file.
    """
    # Path to the directory containing student images
    DB_PATH = "Images"
    
    # Lists to store encodings and corresponding names
    known_encodings = []
    known_names = []

    print("Processing student images...")

    # Loop through each file in the image directory
    for filename in os.listdir(DB_PATH):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            # Construct the full image path
            image_path = os.path.join(DB_PATH, filename)
            
            # Extract the name and roll number from the filename
            # e.g., "Maahie_Gupta_22BLC1165.jpg" -> "Maahie_Gupta_22BLC1165"
            student_name = os.path.splitext(filename)[0]
            
            try:
                # Load the image
                img = fr.load_image_file(image_path)
                
                # Find face encodings. We assume one face per image.
                encodings = fr.face_encodings(img)
                
                if len(encodings) > 0:
                    known_encodings.append(encodings[0])
                    known_names.append(student_name)
                    print(f"✔️ Encoded: {student_name}")
                else:
                    print(f"⚠️ No face found in {filename}, skipping.")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

    # Save the encodings and names to a file
    with open("encodings.pkl", "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    print("\n✅ All images processed and encodings saved to 'encodings.pkl'")


if __name__ == "__main__":
    encode_student_images()