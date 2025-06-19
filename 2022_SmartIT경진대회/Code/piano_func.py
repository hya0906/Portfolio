import winsound
from playsound import playsound
playsound('C:/Users/USER/Desktop/elec_piano/soundfiles/do-c4.mp3')
#winsound.PlaySound("C:/Users/USER/Desktop/elec_piano/soundfiles/do-c4.wav", winsound.SND_NOWAIT)

#한 악보당 16개
note_pos = 1
note_gap = 60
start_pos = 120
last_pos = start_pos + 30
notes_list = []
npos_list = []
page = 1
now_p = 1

def get_canvas(c):
    global canvas
    canvas = c

def get_p1(p):
    global p1
    p1 = p


def remove_all():
    global canvas, note_pos, now_p, p1, page
    for a in notes_list:
        if isinstance(a, list):
            canvas.delete(a[0])
            canvas.delete(a[1])
        canvas.delete(a)
    notes_list.clear()
    npos_list.clear()
    note_pos = 1
    page = 1
    now_p = 1
    canvas.delete(p1)
    p1 = canvas.create_text(630, 200, text=f"page {now_p}", font=("나눔고딕코딩", 20), fill="blue")


def remove_one():
    global note_pos
    a = notes_list[-1]
    if isinstance(a, list):
        canvas.delete(a[0])
        canvas.delete(a[1])
    canvas.delete(notes_list[-1])
    del notes_list[-1]
    del npos_list[-1]
    note_pos -= 1



