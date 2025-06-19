from media_func import *
from test_face import *
import onnxruntime as ort

print("!!!",ort.get_device())
def get_results(image, holistic):
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_skel(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
            .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
            .get_default_hand_landmarks_style())


width = 1280
height = 720
if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("../video.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    print("시작")
    with mp_holistic.Holistic(static_image_mode=True,model_complexity=1,enable_segmentation=True,min_detection_confidence=0.9) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            ###
            i,count = 0, 0
            image = cv2.flip(image, 1)
            h, w, c = image.shape
            src = image.copy()
            where_ppl = {}
            ###
            #화면에 인식된 모든 사람들의 스켈레톤 객체 얻음
            while True:
                try:
                    src, results = get_results(src, holistic)
                    if not results.pose_landmarks.landmark[0]:
                        break

                    a, b, c, d, g = results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[6], \
                                 results.pose_landmarks.landmark[9], results.pose_landmarks.landmark[1],results.pose_landmarks.landmark[0]
                    e, f = results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[8]

                    cv2.rectangle(src, (int(a.x * w), int((b.y - (c.y - d.y)) * h)),
                                  (int(e.x * w), int((2 * a.y - f.y) * h)), (0, 255, 0), 2)

                    src[int((b.y - (c.y - d.y)) * h):int((a.y + a.y - f.y) * h),int(a.x * w):int(e.x * w)] = 0
                    where_ppl[(results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y,
                               results.pose_landmarks.landmark[0].z)] = results
                    i += 1

                except AttributeError or AssertionError:
                    break
            cv2.imshow("aa", src)

            #얼굴 인식 및 얼굴 인증
            if  count % 24 == 0:
                flag, bboxes = recognize_faces(image)  ############
                count=0
            else:
                draw_faces(image)
            count+=1
            
            #인증된 사람의 얼굴부분 스켈레톤이 얼굴 영역 안에 있는지
            xx, yy, ww, hh, _ = map(int, bboxes[0])
            for i,person in enumerate(where_ppl.values()):
                 aa, bb, cc, dd = person.pose_landmarks.landmark[3], person.pose_landmarks.landmark[6], \
                                  person.pose_landmarks.landmark[9], person.pose_landmarks.landmark[10]
                 e = person.pose_landmarks.landmark[0]
                 cv2.circle(image, (int(xx), int(yy)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                 cv2.circle(image, (int(ww), int(hh)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                 #print(e.x * w, e.y * h, "//", xx, yy, ww, hh)

                 if xx < aa.x * w < ww and yy < aa.y * h < hh:
                    if xx < bb.x * w < ww and yy < bb.y * h < hh:
                        if xx < cc.x * w < ww and yy < cc.y * h < hh:
                            if xx < dd.x * w < ww and yy < dd.y * h < hh:
                                results = person
                                print(len(where_ppl), i, "번째 얼굴이 박스안에 존재함.")
            
            #인증된 사람일 경우
            if flag == True:
                # 프레임 확인
                frame_count += 1
                if time.time() - frame_time >= 1:
                    # print("frame : %d" %(frame_count/(time.time()-frame_time)))
                    frame_result = int(frame_count / (time.time() - frame_time))
                    frame_count = 0
                    frame_time = time.time()
                cv2.putText(image, str(frame_result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (20, 255, 155), 3, 5)

                if results.left_hand_landmarks or results.right_hand_landmarks:
                    left_hand = results.left_hand_landmarks
                    right_hand = results.right_hand_landmarks

                    # 손가락 숫자 확인
                    hands_counts, fingers_statuses = count_hand_finger(left_hand, right_hand)

                    # 손의 정적 제스처 탐지
                    gesture = hands_static_gesture(results, left_hand, right_hand, fingers_statuses, hands_counts)
                    # print("main~~",gesture)

                    # 손의 정적 제스처를 저장
                    update_static_gesture(gesture)

                    # 펜 모드 확인
                    check_pen_mode(gesture)

                    # 손의 동적 제스처 탐지
                    action = hands_dynamic_gesture(results, left_hand, right_hand)

                    image = hand_drawing(image, results, gesture, action, left_hand, right_hand)

                    # menu_update(action, hands_counts)

            #image = menu_drawing(image)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                #print(count)
                break
    cap.release()