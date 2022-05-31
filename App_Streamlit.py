import streamlit as st
import cv2
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import numpy as np
from collections import deque

all_rows = open('classes.txt').read().strip().split('\n')
CLASSES = [r[r.find(' ')+1:] for r in all_rows]
classes_list = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),dtype="uint8")
COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

st.title("Video  Classification")
uploaded_video1 = st.file_uploader("Choose video for classification", type=["mp4"])

writer = None
frame_skip = 10

model1 = load_model('model_classification.h5')
if uploaded_video1 is not None: # run only when user uploads video
    vid = uploaded_video1.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video1.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
def predict_on_live_video(video_file_path, output_file_path, window_size):
    
    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting the width and height of the video 
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Writing the Overlayed Video Files Using the VideoWriter Object
    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'x264'), 24, (original_video_width, original_video_height))

    while True: 

        # Reading The Frame
        status, frame = video_reader.read() 

        if not status:
            break

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (64, 64))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame / 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model1.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        # Assuring that the Deque is completely filled before starting the averaging process
        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Predicted Labels Probabilities Deque into Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            # Calculating Average of Predicted Labels Probabilities Column Wise 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

            # Converting the predicted probabilities into labels by returning the index of the maximum value.
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            # Accessing The Class Name using predicted label.
            predicted_class_name = classes_list[predicted_label]
          
            # Overlaying Class Name Text Ontop of the Frame
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Writing The Frame
        video_writer.write(frame)


        # cv2.imshow('Predicted Frames', frame)

        # key_pressed = cv2.waitKey(10)

        # if key_pressed == ord('q'):
        #     break

    # cv2.destroyAllWindows()

    
    # Closing the VideoCapture and VideoWriter objects and releasing all resources held by them. 
    video_reader.release()
    video_writer.release()
# Setting sthe Window Size which will be used by the Rolling Average Proces

window_size = 1

# Constructing The Output YouTube Video Path
output_video_file_path = 'testoutput1.mp4'
if uploaded_video1 is not None:
    predict_on_live_video(uploaded_video1.name, output_video_file_path, window_size)
    st.header("The output video")
    st.video('testoutput1.mp4')

# Play Video File in the Notebook
# VideoFileClip(output_video_file_path).ipython_display(width = 700)
all_rows = open('classes.txt').read().strip().split('\n')
CLASSES = [r[r.find(' ')+1:] for r in all_rows]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES) - 1, 3),dtype="uint8")
COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")

st.title("Video Segmentation ")
uploaded_video = st.file_uploader("Choose video for segmentation", type=["mp4"])

writer = None
frame_skip = 10

model = load_model('model_segmentation.h5')

if uploaded_video is not None: # run only when user uploads video
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read()) # save video to disk

    st.markdown(f"""
    ### Files
    - {vid}
    """,
    unsafe_allow_html=True) # display file name

    vidcap = cv2.VideoCapture(vid) # load video from disk
    cur_frame = 0
    success = True
    st.video(vid)
    img_array = []
    
    while success:
        success, frame = vidcap.read() # get next frame from video
        if cur_frame % frame_skip == 0: # only analyze every n=10 frames

            imgarr = np.asarray(frame, dtype=np.uint8)
            st.header('********************')
            img = cv2.resize(imgarr,(256,256))
            img = img/127.5-1
            img = img.astype(np.float32)
            img= np.expand_dims(img, axis=0)
            pred_label = model.predict(img)
            #(height, width , numClasses) = pred_label.shape[1:4]
            #st.header(( height, width , numClasses))
            classMap = np.argmax(pred_label[0], axis=-1)
            mask = COLORS[classMap]
            mask = cv2.resize(mask, (imgarr.shape[1], imgarr.shape[0]), interpolation=cv2.INTER_NEAREST)
            output = ((0.3 * imgarr) + (0.7 * mask)).astype("uint8")
            img_array.append(output)
            
        cur_frame += 1
    st.text("real number of frames : {}".format(cur_frame))
    out = cv2.VideoWriter('testoutput.mp4',cv2.VideoWriter_fourcc(*'x264'), 5, (output.shape[1], output.shape[0]), True)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    st.header("The output video")
    st.video('testoutput.mp4')
    