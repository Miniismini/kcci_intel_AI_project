
import collections
import sys
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np

from openvino import runtime as ov


import time

sys.path.append("../utils")

base_model_dir = Path("detection_model")





ie_core = ov.Core()

model = ie_core.read_model(model="ssdlite_mobilenet_v2_fp16.xml")

compiled_model = ie_core.compile_model(model=model, device_name="CPU")


input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


height, width = list(input_layer.shape)[1:3]

input_layer.any_name, output_layer.any_name



classes = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet",
    "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()


def process_results(frame, results, thresh=0.6):
   
    h, w = frame.shape[:2]
    
    results = results.squeeze()
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        
        boxes.append(
            tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h)))
        )
        labels.append(int(label))
        scores.append(float(score))

    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6
    )

    # If there are no boxes.
    if len(indices) == 0:
        return []

    # Filter detected objects.
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

global a
global b
global c
a=0
b=0
c=0

global d
global e
global f
d=0
e=0
f=0

def draw_boxes(frame, boxes):
    

    state =True
    state_1=True
    temp=0
    temp_1=0
    for label, score, box in boxes:
        # Choose color for the label.
        color = tuple(map(int, colors[label]))
        # Draw a box.
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)
        
        # Draw a label name inside the box.
        cv2.putText(
            img=frame,
            text=f"{classes[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        global c
        if time.time()-c>1:
            if classes[label] =="cell phone": 
                state ==True
                while state ==True:
                    
                    center_x = (box[0]+x2)/2
                    center_y= (box[1]+y2)/2                
                    
                    if center_x>0 and center_x<640 and center_y >0 and center_y<200:                
                        global a
                        a=time.time()                    
                
                    if center_x>0 and center_x<640 and center_y >280 and center_y<480:                
                        global b
                        b=time.time()               
                    
                    if a != 0 and b !=0 :                  

                        if a-b<0:
                            print("cell phone in")
                            a=0
                            b=0
                            c=time.time()                        

                        if a-b>0:
                            print("cell phone out")
                            a=0
                            b=0
                            c=time.time()   
                    state =False
        global f
        if time.time()-f>1:
            if classes[label] =="bottle": 
                state_1 ==True
                while state_1 ==True:
                    
                    center_x = (box[0]+x2)/2
                    center_y= (box[1]+y2)/2                
                    
                    if center_x>0 and center_x<640 and center_y >0 and center_y<200:                
                        global d
                        d=time.time()                    
                
                    if center_x>0 and center_x<640 and center_y >280 and center_y<480:                
                        global e
                        e=time.time()               
                    
                    if d != 0 and e !=0 :                  

                        if d-e<0:
                            print("bottle in")
                            d=0
                            e=0
                            f=time.time()                        

                        if d-e>0:
                            print("bottle out")
                            d=0
                            e=0
                            f=time.time()   
                    state_1 =False

        
        else:
            continue
    return frame



# Main processing function to run object detection.
def run_object_detection(source=0, flip=False, use_popup=False, skip_first_frames=0):
    
    player = None
    try:
        
        
        cap = cv2.VideoCapture(0)
        w = 640#1280#1920
        h = 480#720#1080
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        
        processing_times = collections.deque()
        while True:
           

            ret, frame = cap.read()

            
            if ret is False:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            # Resize the image and change dims to fit neural network input.
            input_img = cv2.resize(
                src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA
            )
            # Create a batch of images (size = 1).
            input_img = input_img[np.newaxis, ...]

            # Measure processing time.

            start_time = time.time()
            # Get the results.
            results = compiled_model([input_img])[output_layer]
            stop_time = time.time()
            # Get poses from network results.
            boxes = process_results(frame=frame, results=results)

            # Draw boxes on a frame.
            frame = draw_boxes(frame=frame, boxes=boxes)

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=f_width / 1000,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(winname="Press ESC to Exit", mat=frame)
                key = cv2.waitKey(1)
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(
                    ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
                )
                # Create an IPython image.
                #i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                #display.clear_output(wait=True)
                #display.display(i)
                cv2.imshow("camera", encoded_img)

    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        #if player is not None:
            # Stop capturing.
        #    player.stop()
        #if use_popup:
        #    cv2.destroyAllWindows()
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


## run live object detection

run_object_detection(source=0, flip=False, use_popup=True)
