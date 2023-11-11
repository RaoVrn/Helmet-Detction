from flask import Flask, request, jsonify, render_template, send_from_directory
import cv2
import numpy as np

app = Flask(__name__)

# Initialize YOLO model and label paths
helmet_detection_path = 'helmet-detection/'
wts1_path = helmet_detection_path + 'yolov3-helmet.weights'
cfgn1_path = helmet_detection_path + 'yolov3-helmet.cfg'
labels1 = open(helmet_detection_path + 'helmet.names').read().strip().split('\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If the user does not select a file, return an error message
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Call detect_helmet to get the helmet image
    helmet_image = detect_helmet(image)

    # Save the detected helmet image
    helmet_image_path = 'static/helmet_result.jpg'
    cv2.imwrite(helmet_image_path, helmet_image)

    return jsonify({'image_path': '/uploads/' + file.filename, 'helmet_image_path': '/' + helmet_image_path}) 

def detect_helmet(img0):
    p_min = 0.5
    threshold = 0.3
    net1 = cv2.dnn.readNetFromDarknet(cfgn1_path, wts1_path)
    ln1_all = net1.getLayerNames()
    ln1_output = [ln1_all[lyr-1] for lyr in net1.getUnconnectedOutLayers()]

    clr1 =[[179, 61, 234]]

    bb1 = []
    conf1 = []
    c_no_1 = []

    h,w = img0.shape[:2]

    blob = cv2.dnn.blobFromImage(img0, 1/255.0, (416, 416), swapRB=True, crop=False)
    blob_to_show = blob[0,:,:,:].transpose(1,2,0)
    net1.setInput(blob)
    output_from_net1 = net1.forward(ln1_output)

    # For Helmet
    for result in output_from_net1:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > p_min:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width/2))
                y_min = int(y_center - (box_height/2))

                bb1.append([x_min, y_min, int(box_width), int(box_height)])
                conf1.append(float(confidence_current))
                c_no_1.append(class_current)
    results1 = cv2.dnn.NMSBoxes(bb1, conf1, p_min, threshold)

    if len(results1) > 0:
        for i in results1.flatten():
            x_min, y_min = bb1[i][0], bb1[i][1]
            box_width, box_height = bb1[i][2], bb1[i][3]
            colour_box_current = [int(j) for j in clr1[c_no_1[i]]]
            cv2.rectangle(img0, (x_min, y_min), (x_min+box_width, y_min+box_height), colour_box_current, 5)
            text_box_current1 = '{}: {:.4f}'.format(labels1[int(c_no_1[i])], conf1[i])
            cv2.putText(img0, text_box_current1, (x_min, y_min-7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box_current, 5)

    return img0

if __name__ == '__main__':
    app.run(debug=True)
