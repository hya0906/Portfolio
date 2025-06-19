from media_func import *
from test_face import *

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

global bboxes
if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    count=0

    print("시작")
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)
            h, w, c = image.shape
            if not success:
                print("Ignoring empty camera frame.")
                continue
            try:
                if count % 6 == 0:
                    flag, bboxes = recognize_faces(image)  ############
                    count = 0
                else:
                    draw_faces(image)
                count += 1
            except:
                pass

            xx,yy,ww,hh,_ = map(int,bboxes[0])
            image, results = get_results(image, holistic)

            a, b, cc, d = results.pose_landmarks.landmark[12], results.pose_landmarks.landmark[6], results.pose_landmarks.landmark[9], results.pose_landmarks.landmark[1]
            e = results.pose_landmarks.landmark[0]
            cv2.circle(image, (int(e.x * w), int(e.y * h)), 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (int(xx), int(yy)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (int(ww), int(hh)), 5, (255, 255, 255), -1, cv2.LINE_AA)
            #print(e.x*w, e.y*h, "//", xx,yy,ww,hh)

            if xx < e.x*w < ww and yy < e.y*h < hh:
                pass
                ##print("해당 얼굴이 박스안에 존재함.")
                #image, results = get_results(image, holistic) #해당 사람을 주체로 results저장필요

            if flag == True and xx < e.x*w < ww and yy < e.y*h < hh:
                print("flag True 실행됨")
                print("해당 얼굴이 박스안에 존재함.")

                # 프레임 확인
                frame_count += 1
                if time.time() - frame_time >= 1:
                    # print("frame : %d" %(frame_count/(time.time()-frame_time)))
                    frame_result = int(frame_count / (time.time() - frame_time))
                    frame_count = 0
                    frame_time = time.time()
                cv2.putText(image, str(frame_result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (20, 255, 155), 3, 5)
                try:
                    if results.left_hand_landmarks or results.right_hand_landmarks:
                        print("yes")
                        left_hand = results.left_hand_landmarks
                        right_hand = results.right_hand_landmarks

                        # 손가락 숫자 확인
                        hands_counts, fingers_statuses = count_hand_finger(left_hand, right_hand)
                        print(hands_counts)

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
                except:
                    pass
            else:
                print("존재하지 않음XXXXXXX")
            #image = menu_drawing(image)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                print(count)
                break
    cap.release()