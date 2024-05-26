#coding:utf-8
import time
from tkinter import *#图形开发界面的库
#import importlib
#importlib.reload(sys)
#sys.setdefaultencoding('utf_8')#设置默认编码系统，python3中已经不需要了
master=Tk()#创建主窗口对象master
master.title('汉诺塔演示程序(最多可演示8个盘子)')
master.geometry('800x460')#初始化主窗口大小
canvas_width=800
canvas_height=400
w=Canvas(master,width=canvas_width,height=canvas_height,bg="gray")
w.pack()#创建并初始化一个画布对象w

positionX=[150,400,650]#设置三个柱子的水平位置
y=int(canvas_height-50)#设置水平基线的垂直位置

A=0#A,B,C代表三个柱子
B=1
C=2
myCount=[8,0,0]#记录每个柱子上圆盘的数量（最多是8个）
myPlate=[[8,7,6,5,4,3,2,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]#用二维数组记录三个柱子上盘子的摆放情况
currentPlate=1#保存刚刚取下的盘子（数字表示盘子的大小）
sumPlate=3#盘子总数
sumMove=0
str=StringVar();

def hanoi_paintOne(x):#绘制某一个柱子上所有盘子的函数
    for i in range(0,8):
        if myPlate[x][i]>0 and myPlate[x][i]<=sumPlate:
            k=myPlate[x][i]-1;
            LeftUpX=positionX[x]-80+(k*10);
            LeftUpY=y-30*(i+1)-5;
            RightDownX=positionX[x]+80-(k*10)
            RightDownY=y-30*(i+1)+5;
            w.create_rectangle(LeftUpX,LeftUpY,RightDownX,RightDownY,fill="Maroon",outline="Maroon");
    return;

def hanoi_start():#按用户输入盘子数设置A柱上的盘子并清空B和C柱上盘子的函数
    global sumPlate#用户输入的盘子数
    if sumPlate<3:
        sumPlate=3;
    if sumPlate>8:
        sumPlate=8;
    myCount[0]=sumPlate;
    myCount[1]=0;
    myCount[2]=0;
    for i in range(0,8):#清空三个柱子上的盘子
        myPlate[0][i]=0;
        myPlate[1][i]=0;
        myPlate[2][i]=0;
    for i in range(0,sumPlate):#设置A柱上的盘子
        myPlate[0][i]=i+1;
    return;

def clearCanvas():#清空画布的函数
    w.create_rectangle(0,0,800,600,fill="Gray")
    w.pack();

def get_plate_from(X):#从某个柱子上取下一个盘子的函数，取下的盘子放在currentPlate中
    global currentPlate;
    pox=myCount[X]-1;
    currentPlate=myPlate[X][pox]
    myPlate[X][pox]=0;
    myCount[X]=myCount[X]-1;
    return;

def set_plate_to(X):#将当前盘子放在某一个柱子上的函数，待放置的盘子在currentPlate中
    global currentPlate;
    pox=myCount[X];
    myPlate[X][pox]=currentPlate;
    myCount[X]=myCount[X]+1;
    return;

def hanoi_moveplate(X,Y):#将一个X上的盘子移动到另一个柱子Y上的函数
    global sumMove;
    get_plate_from(X);#从X柱上取下盘子
    set_plate_to(Y);#放置到Y柱上
    hanoi_paint();#重新绘制画布上的柱子和盘子
    sumMove=sumMove+1;
    return;

def hanoi(N,X,Y,Z):#将N个盘子从X柱借助Z柱移动到Y柱上的函数
    if(N==1):
        hanoi_moveplate(X,Y);#只有一个盘子时，直接把它从X柱移动到Y柱上     
    else:
        hanoi(N-1,X,Z,Y);#先把N-1个盘子从X移动到Z上(借助Y)
        hanoi_moveplate(X,Y);#然后把X最下面的盘子放到Z
        hanoi(N-1,Z,Y,X);#最后把N-1个盘子再从Z移动到Y上(借助X)

#设置输入和按钮控制的界面要素
w0=Label(master,text='盘子数：',width=10,height=2);
w0.pack(padx=1,pady=1,side=LEFT);
w1=Entry(master,width=12,bg='white');
w1.pack(padx=1,pady=1,side=LEFT);
#exit
w2=Button(master,text="重置",command=(lambda x=ALL:reload_hanoi()),width=12,height=2,bg='red');#“重置”按钮绑定reload_hanoi()
w2.pack(padx=1,pady=1,side=LEFT);
w6=Button(master,text="单步执行",command=(lambda x=ALL: StepByStep()),width=12,height=2,bg='red')#创建“单步执行”按钮并绑定函数 StepByStep()
w6.pack(padx=1,pady=1,side=LEFT);
w3=Button(master,text="自动移动",command=(lambda x=ALL:redo_hanoi()),width=12,height=2,bg='red');#“自动移动”按钮绑定redo_hanoi()
w3.pack(padx=1,pady=1,side=LEFT);
w4=Button(master,text="退出",command=(lambda x=ALL:exit()),width=12,height=2,bg='red');#“退出”按钮绑定exit()
w4.pack(padx=1,pady=1,side=LEFT);
w5=Label(master,text='总共移动次数：',width=20,height=2);
w5.pack(padx=1,pady=1,side=LEFT);

pause=0
def key(event):#处理事件的函数，功能是pause如果为0，则置其为1，如果为1，则置其为0，控制单步执行
    global pause
    pause=(pause+1)%2

w.bind("<Key>",key)#将函数绑定到相应的事件上，Button-1为单击鼠标左键，Key为任何键盘按键，当发生任何一件，就执行key函数
w.bind("<Button-1>",key)
w.focus_set()

def hanoi_paint():#绘制当前所有柱子上的所有盘子的函数
    global pause;
    clearCanvas();#每次重绘后需先清空画布
    y=int(canvas_height-50)
    w.create_rectangle(0,y-5,canvas_width,y+5,fill="#476042")
    w.create_rectangle(positionX[0]-2,y-300,positionX[0]+2,y,fill="#476042")
    w.create_rectangle(positionX[1]-2,y-300,positionX[1]+2,y,fill="#476042")
    w.create_rectangle(positionX[2]-2,y-300,positionX[2]+2,y,fill="#476042")
    hanoi_paintOne(A)
    hanoi_paintOne(B)
    hanoi_paintOne(C)
    master.update()
    time.sleep(0.2)#每次重绘后留有一定的延迟时间，使读者能体验移动过程
    while step_or_continue and pause:#单步执行时，循环延时等待直到pause为0
        master.update();
        time.sleep(0.1);
    pause=1;#这条语句很重要，它将使单击鼠标后pause能够为0进而退出循环。
    return;

def StepByStep():#设置单步执行汉诺塔移动的函数
    global step_or_continue
    step_or_continue=1#值为1则单步执行，为0则自动执行
    global sumMove;
    sumMove=0;
    hanoi_start();
    hanoi_paint();
    hanoi(sumPlate,A,B,C);
    str.set(sumMove)
    w8=Label(master,textvariable=str,width=20,height=2,fg="red");
    w8.pack(padx=1,pady=1,side=LEFT);

def redo_hanoi():#自动执行汉诺塔移动的函数
    global step_or_continue;
    step_or_continue=0;#值为1则单步执行，为0则自动执行
    global sumMove;
    sumMove=0;
    hanoi_start();
    hanoi_paint();
    hanoi(sumPlate,A,B,C); 
    str.set(sumMove)
    w7=Label(master,textvariable=str,width=20,height=2,fg="red");
    w7.pack(padx=1,pady=1,side=LEFT);
   
def reload_hanoi():#重置游戏参数的函数
    global sumPlate;
    global step_or_continue;

    step_or_continue=0;
    temp=w1.get();#从盘子数输入框中获得用户的输入值
    if temp.strip()=='':#判断输入是否为空
        sumPlate=3;
    if temp.isdigit():#判断输入是否是数字
        sumPlate=int(w1.get(),10)#将用户输入的字符转换为数字
    else:
        sumPlate=3;
    clearCanvas();
    hanoi_start();
    hanoi_paint();
reload_hanoi();
redo_hanoi();
#StepByStep()
master.mainloop();
