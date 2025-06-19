from tkinter import *
from draw_UI import *
from tkinter import filedialog
from tkinter.messagebox import *
import os
import numpy as np
import math
from playsound import playsound
import threading
import time
###playsound==1.2.2
###python==3.8

# #한 악보당 16개
# SOUND_FOLDER = r"C:\Users\USER\Desktop\elec_piano\soundfiles"
SOUND_FOLDER = r".\elec_piano\soundfiles"
notes = {"do-c4.wav":0, "re-d4.wav":1, "mi-e4.wav":2, "fa-f4.wav":3, "sol-g4.wav":4, "la-a4.wav":5, "si-b4.wav":6, "do-c5.wav":7,
         "xylophone-c1.wav":8, "xylophone-d1.wav":9, "xylophone-e1.wav":10, "xylophone-f1.wav":11, "xylophone-g1.wav":12,
         "xylophone-a1.wav":13, "xylophone-b1.wav":14, "xylophone-c2.wav":15}
instrument = 0
note_pos = 1    #음표 위치
note_gap = 60   #음표간 간격
start_pos = 120 #첫번째 음표 위치
last_pos = start_pos + 30
notes_list = []
npos_list = []
sound_list = [] #전체 음포 실행 리스트
page = 1        #전체 페이지
now_p = 1       #현재 페이지

#모든 음표 지우기
def remove_all():
    global canvas, note_pos, now_p, p1, page, sound_list
    for a in notes_list:
        if isinstance(a, list): #가온 도일 경우
            canvas.delete(a[0]) #note
            canvas.delete(a[1]) #line
        canvas.delete(a)        #가온도 제외 모두
    notes_list.clear()
    npos_list.clear()
    sound_list.clear()
    note_pos = page = now_p = 1 #음표위치,전체페이지,현재페이지 초기화
    canvas.delete(p1)           #페이지 글씨객체 삭제
    p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")


#마지막 음표 하나 지우기
def remove_one():
    global note_pos, now_p, p1, page, sound_list
    if note_pos == 1: #마지막 장에 음표가 하나 있을때
        if now_p <= 1:
            showwarning("경고", "첫 페이지입니다")
            return
        print(now_p)
        now_p -= 1
        page -= 1
        note_pos = 16
        showinfo("알림", f"page {now_p}로 넘어갑니다.")
        for i,(a,b,c,d) in enumerate(npos_list[16*(now_p-1):(16*now_p)]): #그 전 페이지 그림 다시 그리기
            n = canvas.create_oval(a, b, c, d,fill="black")
            if (b == 160) and (d == 185):
                l = canvas.create_line(a - 10, 170, c + 10, 170, fill="black")
                notes_list[16*(now_p-1)+i] = [n,l]
            else:
                notes_list[16*(now_p-1)+i] = n
            canvas.delete(p1)
            p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")
    else:              #마지막장에 음표가 1개 이상 있을때
        note_pos -= 1

    a = notes_list[-1]
    if isinstance(a, list):
        canvas.delete(a[0])
        canvas.delete(a[1])
    canvas.delete(notes_list[-1])
    del notes_list[-1]
    del npos_list[-1]
    del sound_list[-1]


def n_draw_save(a,b,c,d):
    global npos_list, note_pos
    if (b == 160) and (d == 185): #가온 도 일경우
        line = canvas.create_line(a - 10, 170, c + 10, 170, fill="black")
        note = canvas.create_oval(a, b, c, d, fill="black")
        notes_list.append([note, line])
        npos_list.append([a, b, c, d])
    else:                         #가온 도 제외하고 모든것
        note = canvas.create_oval(a, b, c, d, fill="black")
        notes_list.append(note)
        npos_list.append([a, b, c, d])
    note_pos += 1

def save_data():
    global npos_list, sound_list
    os.makedirs("piano_data", exist_ok=True)
    npos_list = np.array(npos_list)
    sound_list = np.array(sound_list)
    np.save("./piano_data/pos_list1.npy",npos_list)
    np.save("./piano_data/sound_list1.npy", sound_list)
    npos_list = npos_list.tolist()
    sound_list = sound_list.tolist()
    showinfo("알림", "데이터를 저장하였습니다")

def load_data():
    global npos_list, note_pos, page, now_p, sound_list, p1
    if note_pos == page == now_p == 1:
        data_path = filedialog.askopenfilename(initialdir='./', title='npos_list파일선택', filetypes=(('npy files', '*.npy'), ('all files', '*.*')))
        npos_list = np.load(data_path).tolist()
        data_path = filedialog.askopenfilename(initialdir='./', title='sound_list파일선택', filetypes=(('npy files', '*.npy'), ('all files', '*.*')))
        sound_list = np.load(data_path).tolist()

        if len(npos_list) % 16 == 0: #악보가 다 꽉 찼을 경우
            note_pos = 17
        else:                   #악보가 비어있을 경우
            note_pos = (len(npos_list) % 16) + 1
        page =  math.ceil(len(npos_list)/16)
        now_p = 1
        print("~~~~~~~~~~~~",page,now_p, note_pos)

        for i, (a,b,c,d) in enumerate(npos_list[(16*(now_p-1)):(16*now_p)]):
            n = canvas.create_oval(a, b, c, d,fill="black")
            if (b == 160) and (d == 185):
                l = canvas.create_line(a - 10, 170, c + 10, 170, fill="black")
                notes_list.append([n,l])
            else:
                notes_list.append(n)
            canvas.delete(p1)
            p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")
        showinfo("알림", "데이터를 불러왔습니다")

    else:
        showerror("경고", "음표가 아무것도 저장되어있지 않아야 합니다")

def play_sound(file_name, flag = 0):
    global sound_list
    # wav파일 재생
    path = os.path.join(SOUND_FOLDER, file_name)
    if flag == 0:
        sound_list.append(notes[file_name])
    else:
        pass
    playsound(path)


def play_all(): #전체 음표 실행
    global sound_list
    reverse_dict = dict(map(reversed, notes.items()))
    for i in sound_list:
        if isinstance(sound_list[0], int): # 불러온 데이터일때
            name = reverse_dict[i]
            t = threading.Thread(target=play_sound, args=(name, 1))
            t.start()
            time.sleep(0.7)

        else:                              # 직접 친 데이터일때
            t = threading.Thread(target=play_sound, args=(i, 1))
            t.start()
            time.sleep(0.7)
        continue

def callback_mouse(event):
    global canvas, note_pos, npos_list, p1, page, now_p, sound_list, instrument
    print(event.x, event.y)
    # 피아노변경
    if (70 < event.x < 230) and (300 < event.y < 360):
        instrument = 0
        showinfo("알림", "피아노로 변경하였습니다")
    # 실로폰변경
    if (70 < event.x < 230) and (380 < event.y < 440):
        instrument = 1
        showinfo("알림", "실로폰으로 변경하였습니다")
    # 모든 음표 지우기
    if (960 < event.x < 1180) and (300 < event.y < 360):
        remove_all()

    # 음표 하나 지우기
    elif (960 < event.x < 1180) and (380 < event.y < 440):
        remove_one()

    # 음표 데이터 저장
    elif (960 < event.x < 1180) and (460 < event.y < 540):
        save_data()

    # 음표 데이터 불러오기
    elif (960 < event.x < 1180) and (540 < event.y < 620):
        load_data()

    # 전체 실행하기
    elif (960 < event.x < 1180) and (620 < event.y < 700):
        t = threading.Thread(target=play_all)
        t.start()

    # 악보 오른쪽으로 넘기기
    elif (1210 < event.x < 1245) and (85 < event.y < 150):
        if now_p == page:
            showwarning("경고", "다음 장이 없습니다")
            return
        now_p += 1
        print("오른쪽#######",now_p,(16*(now_p-1)),16*now_p)
        for a in notes_list[16*((now_p-1)-1):(16*(now_p-1)+1)]: #전 장 지우기
            if isinstance(a, list):
                canvas.delete(a[0])
                canvas.delete(a[1])
            else:
                canvas.delete(a)
        try: #마지막 페이지가 꽉 차있지 않을 경우에 index error남
            for i, (a, b, c, d) in enumerate(npos_list[(16 * (now_p - 1)):(16 * now_p)]): # 현재 장 그리기
                print("iiii",npos_list[16+i], npos_list[(16 * (now_p - 1)):(16 * now_p)])
                n = canvas.create_oval(a, b, c, d, fill="black")
                if (b == 160) and (d == 185):
                    l = canvas.create_line(a - 10, 170, c + 10, 170, fill="black")
                    notes_list.append([n, l])
                else:
                    notes_list.append(n)
        except IndexError:
            canvas.delete(p1)
            p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")
            return
        canvas.delete(p1)
        p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")


    # 악보 왼쪽으로 넘기기
    elif (35 < event.x < 70) and (85 < event.y < 120):
        if now_p <= 1:
            showwarning("경고", "첫 번째 페이지입니다")
            return
        print(now_p)
        now_p -= 1
        print("왼쪽지우기",16*now_p,16*(now_p+1))
        for a in notes_list[16*now_p:16*(now_p+1)]: #전 장 지우기
            if isinstance(a, list):
                canvas.delete(a[0])
                canvas.delete(a[1])
            else:
                canvas.delete(a)

        print("왼쪽#######",now_p, 16*(now_p-1),(16*now_p))
        for i,(a,b,c,d) in enumerate(npos_list[16*(now_p-1):(16*now_p)]): #현재 장 그리기
            n = canvas.create_oval(a, b, c, d,fill="black")
            if (b == 160) and (d == 185):
                l = canvas.create_line(a - 10, 170, c + 10, 170, fill="black")
                notes_list[16*(now_p-1)+i] = [n,l]
            else:
                notes_list[16*(now_p-1)+i] = n
        print(notes_list)
        canvas.delete(p1)
        p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")

    # 건반
    elif 300 < event.y < 600:
        if 270 < event.x < 350: #도
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 160, last_pos + (note_gap*note_pos), 185)
            #note_pos+=1
            if instrument == 0:
                note = "do-c4.wav"
            else:
                note = "xylophone-c1.wav"
            t = threading.Thread(target=play_sound, args=([note]))
            t.start()

        elif 350 < event.x < 430: #레
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 148, last_pos + (note_gap*note_pos), 173) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "re-d4.wav"
            else:
                note = "xylophone-d1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 430 < event.x < 510: #미
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 136, last_pos + (note_gap*note_pos), 161) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "mi-e4.wav"
            else:
                note = "xylophone-e1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 510 < event.x < 590: #파
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 124, last_pos + (note_gap*note_pos), 149) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "fa-f4.wav"
            else:
                note = "xylophone-f1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 590 < event.x < 670: #솔
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 112, last_pos + (note_gap*note_pos), 137) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "sol-g4.wav"
            else:
                note = "xylophone-g1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 670 < event.x < 750: #라
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 100, last_pos + (note_gap*note_pos), 125) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "la-a4.wav"
            else:
                note = "xylophone-a1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 750 < event.x < 830: #시
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 88, last_pos + (note_gap*note_pos), 113) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "si-b4.wav"
            else:
                note = "xylophone-b1.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()

        elif 830 < event.x < 910: #도
            if note_pos % 17 == 0:
                for a in notes_list:
                    if isinstance(a, list):
                        canvas.delete(a[0])
                        canvas.delete(a[1])
                    canvas.delete(a)
                note_pos = 1
                page += 1
                now_p = page
                canvas.delete(p1)
                p1 = canvas.create_text(630, 200, text=f"page {page}", font=("나눔고딕코딩", 20), fill="blue")

            n_draw_save(start_pos + (note_gap*note_pos), 76, last_pos + (note_gap*note_pos), 101) #음표 그리기
            #note_pos += 1
            if instrument == 0:
                note = "do-c5.wav"
            else:
                note = "xylophone-c2.wav"
            t = threading.Thread(target=play_sound, args=([note])) #동시에 음계 소리 실행
            t.start()
    print(len(npos_list))

def main():
    global canvas, page, p1
    width = 1300; height = 700
    app = Tk()
    app.title("electric piano")
    canvas = Canvas(app, width=width, height=height, bg="white")
    canvas.pack(fill=BOTH, expand=True)
    photoImage = PhotoImage(file="./high.png").subsample(3) #draw high note
    canvas.create_image(80, 45, anchor=NW, image=photoImage)

    p1 = draw_screen(canvas)
    app.bind("<Button-1>", callback_mouse)
    app.mainloop()

if __name__ == "__main__":
    main()