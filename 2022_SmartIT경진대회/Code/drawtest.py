'''
from tkinter import *
global canvas, page, p1
width = 1300
    height = 700
    app = Tk()
    app.title("electric piano")
    canvas = Canvas(app, width=width, height=height, bg="white")
    canvas.pack(fill=BOTH, expand=True)
    print(canvas.winfo_reqheight(), canvas.winfo_reqwidth())
    photoImage = PhotoImage(file="C:/Users/USER/Desktop/elec_piano/high.png").subsample(3)
    canvas.create_image(80, 45, anchor=NW, image=photoImage)
    app.bind("<Button-1>", callback_mouse)
    app.mainloop()
'''