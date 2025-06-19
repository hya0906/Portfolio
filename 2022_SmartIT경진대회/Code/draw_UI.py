
def draw_screen(canvas):
    for x in range(50, 170, 25): #lines
        canvas.create_line(80, x, 1200, x, fill="black")

    for x, note in zip(range(270, 891, 80), ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]): #white
        canvas.create_rectangle(x, 300, x + 80, 600, fill='white', outline='black', width=1)
        canvas.create_text(x + 40, 570, text=note, font=("나눔고딕코딩", 20), fill="blue")

    for x in range(320,721, 80):#black
        if x == 480:
            continue
        canvas.create_rectangle(x, 300, x+60, 450, fill='black', outline='black', width=1)
    canvas.create_rectangle(880, 300, 880+30, 450, fill='black', outline='black', width=1)
    p1 = canvas.create_text(630, 200, text=f"page 1", font=("나눔고딕코딩", 20), fill="blue")
    draw_buttons(canvas)
    return p1

def draw_buttons(canvas):
    canvas.create_rectangle(960, 300, 1180, 360, fill='white', outline='black', width=1)
    canvas.create_text(1070, 330, text= "모든 음표 지우기", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(960, 380, 1180, 440, fill='white', outline='black', width=1)
    canvas.create_text(1070, 410, text="음표 하나 지우기", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(960, 460, 1180, 520, fill='white', outline='black', width=1)
    canvas.create_text(1070, 490, text="악보데이터 저장", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(960, 540, 1180, 600, fill='white', outline='black', width=1)
    canvas.create_text(1070, 570, text="데이터 불러오기", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(960, 620, 1180, 680, fill='white', outline='black', width=1)
    canvas.create_text(1070, 650, text="전체 실행", font=("나눔고딕코딩", 20), fill="black")

    canvas.create_rectangle(70, 300, 230, 360, fill='white', outline='black', width=1)
    canvas.create_text(150, 330, text="피아노", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(70, 380, 230, 440, fill='white', outline='black', width=1)
    canvas.create_text(150, 410, text="실로폰", font=("나눔고딕코딩", 20), fill="black")


    canvas.create_rectangle(35, 85, 70, 120, fill='white', outline='black', width=1)
    canvas.create_text(50, 102, text="◀", font=("나눔고딕코딩", 20), fill="black")
    canvas.create_rectangle(1210, 85, 1245, 120, fill='white', outline='black', width=1)
    canvas.create_text(1228, 102, text="▶", font=("나눔고딕코딩", 20), fill="black")