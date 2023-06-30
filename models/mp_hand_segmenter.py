
#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt
import torch
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def init_hand_kp_model():
    # STEP 2: Create an HandLandmarker object.
    base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.01)
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

# detector = init_hand_kp_model()

def create_segmenter(hand=True):
    # Create the options that will be used for ImageSegmenter
    model_asset_path='models/selfie_multiclass_256x256.tflite' if hand else 'models/deeplab_v3.tflite'
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)

    # Create the image segmenter
    segmenter = vision.ImageSegmenter.create_from_options(options)
    return segmenter 

# hand_segmenter = create_segmenter(True)
# obj_segmenter = create_segmenter(False)

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    # cv2.putText(annotated_image, f"{handedness[0].category_name}",
    #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
    #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def calculate_offset(x_coordinates, y_coordinates):
    edges = list(solutions.hands.HAND_CONNECTIONS)
    offset = 0
    for i in range(len(edges)):
        fr, to = edges[i][0], edges[i][1]
        if fr == 0 or to == 0:
            continue
        distance = np.sqrt((x_coordinates[fr] - x_coordinates[to])**2 + (y_coordinates[fr] - y_coordinates[to])**2)
        offset += distance

    offset = int(offset / len(x_coordinates))
    return offset

def get_hand_results(image, detection_result):
    image_np = image.numpy_view()
    height, width, _ = image_np.shape
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    # Loop through the detected hands to visualize.
    
    cropped_image = np.zeros(image_np.shape, dtype=np.uint8)
    bbs = []
    pose2d = torch.zeros((2, 21, 2))

    for idx, hand in enumerate(['Left', 'Right']):
        indexes = [i for i, handedness in enumerate(handedness_list) if handedness[0].category_name == hand]
        if len(indexes) == 0:
            pose2d[idx] = torch.zeros((21, 2))
            continue
        
        max_index = max(indexes, key=lambda x: handedness_list[x][0].score)
            
        hand_landmarks = hand_landmarks_list[max_index]
        x_coordinates = [int(landmark.x * width) for landmark in hand_landmarks]
        y_coordinates = [int(landmark.y * height) for landmark in hand_landmarks]
        
        pose_2d_side = torch.stack((torch.tensor(x_coordinates).view(-1, 21), torch.tensor(y_coordinates).view(-1, 21)), dim=2)[0]
        pose2d[idx] = pose_2d_side

        offset = calculate_offset(x_coordinates, y_coordinates)
        min_x, min_y, max_x, max_y = min(x_coordinates) - offset, min(y_coordinates) - offset, max(x_coordinates) + offset, max(y_coordinates) + offset

        bbs.append([min_x, min_y, max_x, max_y])
        cropped_image[min_y:max_y, min_x:max_x, :] = image_np[min_y:max_y, min_x:max_x, :]

    return pose2d, cropped_image, bbs

def get_segmentation_mask(segmenter, image, class_id):
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask.numpy_view()

    mask = category_mask == class_id
    mask_arr = np.copy(image.numpy_view())
    mask_arr[~mask] = 0

    return mask_arr

    
def detect_hand(image, detector=None, segmenter=None):

    # Load the input image from numpy array
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    # Detect hand landmarks from the input image.
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Process the classification result. 
    pose2d, cropped_image, bb = get_hand_results(image, detection_result)

    # Segment the hand from the cropped image
    cropped_image, masked_hand = None, None
    if segmenter is not None:
        cropped_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_image)
        masked_hand = get_segmentation_mask(segmenter, cropped_image, 2)

    return annotated_image, pose2d, cropped_image, bb, masked_hand

if __name__ == '__main__':
    path = '/home2/HO3D_v3/train/ABF10/rgb/0882.jpg'
    path = '/home2/HO3D_v3/train/SS2/rgb/0885.jpg'
    path = '/home2/HO3D_v3/train/ShSu13/rgb/0002.jpg'
    
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    annotated_image, _, _, cropped_image, _, masked_hand = detect_hand(image)
    
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    axs[0].imshow(annotated_image)
    axs[0].set_title('RGB Image')
    
    axs[1].imshow(masked_hand)
    axs[1].set_title('Hand Segmentation Mask')

    # axs[2].imshow(masked_obj)
    # axs[2].set_title('Object Segmentation Mask')

    plt.show()
