#import sys
#sys.path.append("D:\OPENCV\myvenv\Lib\site-packages")
import mediapipe.python.solutions.hands as mpHands
import numpy as np
import cv2 as cv
import math

cap = cv.VideoCapture(0)
width = int(cap.get(3))
height = int(cap.get(4))

hands = mpHands.Hands(False, 1)
img = cv.imread("D:\OPENCV\mediapipe_hand_gesture\hand_landmarks_text\SARPPAKKAAVIL.png", cv.IMREAD_UNCHANGED)
img2 = cv.imread("D:\OPENCV\mediapipe_hand_gesture\hand_landmarks_text\SRAPPA_PRETHAM.png", cv.IMREAD_UNCHANGED)

img_height, img_width, ch = img.shape
alpha_channel = img[:, :, 3] / 255  # Normalize alpha channel to 0-1 range
color_channel = img[:, :, :3]

img2_height, img2_width, ch = img2.shape
alpha2_channel = img2[:, :, 3] / 255  # Normalize alpha channel to 0-1 range
color2_channel = img2[:, :, :3]

alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))
alpha2_mask = np.dstack((alpha2_channel, alpha2_channel, alpha2_channel))
count = 1
while True:
    ret, frames = cap.read()
    frames = cv.flip(frames, 1)
    if not ret:
        print("Error reading frame")
        break

    frames2RGB = cv.cvtColor(frames, cv.COLOR_BGR2GRAY)
    gray_3_channel = cv.merge([frames2RGB, frames2RGB, frames2RGB])
    frames= gray_3_channel
    cv.imshow ("gray", frames)
    result = hands.process(gray_3_channel)

    if result.multi_hand_landmarks:
        for idx, landmarks in enumerate(result.multi_hand_landmarks):
            thumb_pos = (
                int(landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].x * width),
                int(landmarks.landmark[mpHands.HandLandmark.THUMB_TIP].y * height)
            )
            index_pos = (
                int(landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * width),
                int(landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * height)
            )

            middle_pos = (
                int(landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].x * width),
                int(landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP].y * height)
            )
            index2_pos = (
                int(landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * width),
                int(landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * height)

            )
            # Midpoint between thumb and index finger
            center_x = (thumb_pos[0] + index_pos[0]) // 2
            center_y = (thumb_pos[1] + index_pos[1]) // 2

            # Midpoint between index finger and middle finger
            center2_x = (middle_pos[0] + index2_pos[0]) // 2
            center2_y = (middle_pos[1] + index2_pos[1]) // 2

            # Calculate angle and distance between the two fingers
            delta_x = index_pos[0] - thumb_pos[0]
            delta_y = index_pos[1] - thumb_pos[1]
            angle_rad = math.atan2(delta_y, delta_x)
            angle_deg = math.degrees(angle_rad)
            distance = int(math.hypot(delta_x, delta_y))

            delta2_x = index2_pos[0] - middle_pos[0]
            delta2_y = index2_pos[1] - middle_pos[1]
            angle2_rad = math.atan2(delta2_y, delta2_x)
            angle2_deg = math.degrees(angle2_rad+math.pi)
            distance2 = int(math.hypot(delta2_x, delta2_y))




            # Resize image based on distance between thumb and index
            scale_factor = distance / img_width
            img_resized = cv.resize(color_channel,None, fx=scale_factor, fy=1)
            alpha_resized = cv.resize(alpha_mask, None, fx=scale_factor, fy=1)

            scale2_factor = distance2 / img2_width
            img2_resized = cv.resize(color2_channel,None, fx=scale2_factor, fy=1)
            cv.imshow("img_resized", img2_resized)
            alpha2_resized = cv.resize(alpha2_mask, None, fx=scale2_factor, fy=1)

            #To change the pivor of resize
            # pivot_x, pivot_y = 0,0

            # translation_matrix_to_origin = np.float32([[1, 0, -pivot_x], [0, 1, -pivot_y]])
            # translated_image = cv.warpAffine(color_channel, translation_matrix_to_origin, (img_width, img_height))
            # #color_resized_image = cv.resize(translated_image, (0, 0), fx=scale_factor, fy=scale_factor)
            # img_resized = cv.resize(translated_image, (0, 0), fx=scale_factor, fy=1)
            # alpha_translated_image = cv.warpAffine(alpha_mask, translation_matrix_to_origin, (img_width, img_height))
            # #alpha_resized_image = cv.resize(alpha_translated_image, (0, 0), fx=scale_factor, fy=scale_factor)
            # alpha_resized = cv.resize(alpha_translated_image, (0, 0), fx=scale_factor, fy=1)

            # translation_matrix_back = np.float32([[1, 0, pivot_x], [0, 1, pivot_y]])
            # img_resized = cv.warpAffine(color_resized_image, translation_matrix_back, (img_width, img_height))
            # alpha_translation_matrix_back = np.float32([[1, 0, pivot_x], [0, 1, pivot_y]])
            # alpha_resized = cv.warpAffine(color_resized_image, translation_matrix_back, (img_width, img_height))


            # Get dimensions of resized image
            img_h, img_w = img_resized.shape[:2]
            img2_h, img2_w = img2_resized.shape[:2]


            # Compute the diagonal length of the resized image
            diagonal_length = int(math.sqrt(img_w**2 + img_h**2))
            diagonal2_length = int(math.sqrt(img2_w**2 + img2_h**2))


            # Create a larger blank canvas to fit the rotated image
            large_canvas = np.zeros((diagonal_length, diagonal_length, 3), dtype=np.uint8)
            large_alpha = np.zeros((diagonal_length, diagonal_length, 3), dtype=np.float32)
           
            large2_canvas = np.zeros((diagonal2_length, diagonal2_length, 3), dtype=np.uint8)
            large2_alpha = np.zeros((diagonal2_length, diagonal2_length, 3), dtype=np.float32)

            # Center the resized image on the larger canvas [PADDING]
            x_offset = (diagonal_length - img_w) // 2
            y_offset = (diagonal_length - img_h) // 2
            large_canvas[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = img_resized
            large_alpha[y_offset:y_offset + img_h, x_offset:x_offset + img_w] = alpha_resized

            x2_offset = (diagonal2_length - img2_w) // 2
            y2_offset = (diagonal2_length - img2_h) // 2
            large2_canvas[y2_offset:y2_offset + img2_h, x2_offset:x2_offset + img2_w] = img2_resized
            large2_alpha[y2_offset:y2_offset + img2_h, x2_offset:x2_offset + img2_w] = alpha2_resized
            cv.imshow("img2resized", img2_resized)


            # Rotate the larger canvas and alpha channel
            center = (diagonal_length // 2, diagonal_length // 2)
            M = cv.getRotationMatrix2D(center, -angle_deg, 1)
            rotated_img = cv.warpAffine(large_canvas, M, (diagonal_length, diagonal_length), flags=cv.INTER_LINEAR)
            rotated_alpha = cv.warpAffine(large_alpha, M, (diagonal_length, diagonal_length), flags=cv.INTER_LINEAR)

            center2 = (diagonal2_length // 2, diagonal2_length // 2)
            M2 = cv.getRotationMatrix2D(center2, -angle2_deg, 1)
            rotated2_img = cv.warpAffine(large2_canvas, M2, (diagonal2_length, diagonal2_length), flags=cv.INTER_LINEAR)
            rotated2_alpha = cv.warpAffine(large2_alpha, M2, (diagonal2_length, diagonal2_length), flags=cv.INTER_LINEAR)

            # Determine position to overlay the rotated image on the original frame
            x_offset = center_x - diagonal_length // 2
            y_offset = center_y - diagonal_length // 2
            y1, y2 = max(0, y_offset), min(frames.shape[0], y_offset + diagonal_length)
            x1, x2 = max(0, x_offset), min(frames.shape[1], x_offset + diagonal_length)

            x2_offset = center2_x - diagonal2_length // 2
            y2_offset = center2_y - diagonal2_length // 2
            y21, y22 = max(0, y2_offset), min(frames.shape[0], y2_offset + diagonal2_length)
            x21, x22 = max(0, x2_offset), min(frames.shape[1], x2_offset + diagonal2_length)

            # Crop the rotated image and alpha mask to fit within the frame
            cropped_color = rotated_img[max(0, -y_offset):rotated_img.shape[0], max(0, -x_offset):rotated_img.shape[1]]
            print(f"cropped image shape: {cropped_color.shape}")
            cropped_alpha = rotated_alpha[max(0, -y_offset):rotated_alpha.shape[0], max(0, -x_offset):rotated_alpha.shape[1]]

            cropped2_color = rotated2_img[max(0, -y2_offset):rotated2_img.shape[0], max(0, -x2_offset):rotated2_img.shape[1]]
            print(f"cropped image shape: {cropped2_color.shape}")
            cropped2_alpha = rotated2_alpha[max(0, -y2_offset):rotated2_alpha.shape[0], max(0, -x2_offset):rotated2_alpha.shape[1]]


            # Blend the image onto the frame
            try:
                alpha_mask_crop = cropped_alpha[:, :, 0]
                alpha_inv = 1.0 - alpha_mask_crop
                for c in range(3): 
                    frames[y1:y2, x1:x2, c] = (
                        alpha_mask_crop * cropped_color[:, :, c] + alpha_inv * frames[y1:y2, x1:x2, c]
                )
            except Exception as e:
                print(f"VALUE ERROR: {e}")
                pass


            try:
                alpha2_mask_crop = cropped2_alpha[:, :, 0]
                alpha2_inv = 1.0 - alpha2_mask_crop
                for c in range(3): 
                    frames[y21:y22, x21:x22, c] = (
                        alpha2_mask_crop * cropped2_color[:, :, c] + alpha2_inv * frames[y21:y22, x21:x22, c]
                )
            except Exception as e:
                print(f"VALUE ERROR: {e}")
                pass
    #frames_final = cv.cvtColor(frames, cv.COLOR_RGB2GRAY)
    cv.imshow("Valicho", frames)
    cv.imwrite('D:/OPENCV/mediapipe_hand_gesture/frames_out/frame%d.jpg' % count, frames)

    if cv.waitKey(1) == ord('q'):
        break
    count = count+1
cap.release()
cv.destroyAllWindows()