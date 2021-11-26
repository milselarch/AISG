import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
)

basedir = '../datasets/extract/mtcnn-sync/0a3e16d70c766db6'
image_path = f'{basedir}/0-11.jpg'

image = cv2.imread(image_path)
results = detector.process(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
)

annotated_image = image.copy()
for detection in results.detections:
    print('Mouth center:')
    print(mp_face_detection.get_key_point(
        detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER
    ))

    mp_drawing.draw_detection(annotated_image, detection)

print(results)
print('DONE')