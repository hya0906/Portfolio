import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(3)) # 가로 길이 가져오기
height = int(cap.get(4)) # 세로 길이 가져오기
#img = cv2.imread("D:\insightface_folder\lab_test\data\gong.jpg")
#cv2.imshow("img",img)
print("2")

while(True) :
    print("1")
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Fail to read frame!")
        break

cap.release()
cv2.destroyAllWindows()
