import streamlit as st
import cv2
import numpy as np
# from PIL import Image
# from io import BytesIO
# import base64

# The main title
st.title("ECE 253 - Face Detection")
# Load an image &  a mask
uploaded_image = st.file_uploader("Choose an (RGB) image to upload", type=['jpg', 'jpeg', 'png'])
uploaded_mask = st.file_uploader("Choose a maks (image) to upload", type=['jpg', 'jpeg', 'png'])

# Select button
select_val = st.selectbox("Select method(s) to find faces",["Cross Correlation", "Deep Learning", "Both"])

# Function for cross correlation
def CC(img, temp, threshold):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    h,w = gray_temp.shape

    # Cross correlation
    res = cv2.matchTemplate(gray_img, gray_temp, cv2.TM_CCOEFF_NORMED)

    # Based on the threshold
    vals = np.where(res >= threshold)

    copy_img = np.copy(img)

    for pt in zip(*vals[::-1]):
        cv2.rectangle(copy_img, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 1)

    return copy_img

# Function for detecting facses in an image.
def detectFaceOpenCVDnn(net, frame):
    # Create a blob from the image and apply some pre-processing.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    # Set the blob as input to the model.
    net.setInput(blob)
    # Get Detections.
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5):
    bboxes = []
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # Loop over all detections and draw bounding boxes around each face.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            bb_line_thickness = max(1, int(round(frame_h / 200)))
            # Draw bounding boxes around detected faces.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), bb_line_thickness, cv2.LINE_8)
    return frame, bboxes

# Function to load the DNN model.
@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

if uploaded_image is not None and uploaded_mask is not None:
    # Read the image file and convert it to opencv Image.
    img_raw_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    # Read the template file and convert it to opencv Image.
    template_raw_bytes = np.asarray(bytearray(uploaded_mask.read()), dtype=np.uint8)

    # Loads image & template in a BGR channel order.
    image = cv2.imdecode(img_raw_bytes, cv2.IMREAD_COLOR)
    template = cv2.imdecode(template_raw_bytes, cv2.IMREAD_COLOR)

    # Or use PIL Image (which uses an RGB channel order)
    # image = np.array(Image.open(img_file_buffer))

    # Create placeholders to display input and output images.
    num_cols = 3 if select_val == "Both" else 2
    placeholders = st.columns(num_cols)
    # Display Input image in the first placeholder.
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")

    # Create a Slider and get the threshold from the slider.
    conf_threshold = st.slider("SET Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)

    # Do detection based on the method required

    # Call the face detection model to detect faces in the image.
    if select_val == "Deep Learning":
        detections_dl = detectFaceOpenCVDnn(net, image)
        # Process the detections based on the current confidence threshold.
        out_image, _ = process_detections(image, detections_dl, conf_threshold=conf_threshold)
        # Display Detected faces.
        placeholders[1].image(out_image, channels='BGR')
        placeholders[1].text("Output (DNN) Image")
    
    elif select_val == "Cross Correlation":
        detections_cc = CC(image, template, conf_threshold)
        placeholders[1].image(detections_cc, channels='BGR')
        placeholders[1].text("Output (CC) Image")
    
    else: # select_val == "Both"
        detections_dl = detectFaceOpenCVDnn(net, image)
        detections_cc = CC(image, template, conf_threshold)

        out_image, _ = process_detections(image, detections_dl, conf_threshold=conf_threshold)
        # Display Detected faces.
        placeholders[1].image(out_image, channels='BGR')
        placeholders[1].text("Output (DNN) Image")

        placeholders[2].image(detections_cc, channels='BGR')
        placeholders[2].text("Output (CC) Image")

    

    # # Process the detections based on the current confidence threshold.
    # out_image, _ = process_detections(image, detections_dl, conf_threshold=conf_threshold)

    # # Display Detected faces.
    # placeholders[1].image(out_image, channels='BGR')
    # placeholders[1].text("Output Image")

    # Convert opencv image to PIL.
    # out_image = Image.fromarray(out_image[:, :, ::-1])
    # Create a link for downloading the output file.
    # st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'),
    #             unsafe_allow_html=True)