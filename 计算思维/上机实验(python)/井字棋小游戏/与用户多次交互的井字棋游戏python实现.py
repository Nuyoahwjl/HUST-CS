# #使用博弈树、估价函数，确定电脑最佳落子处

# def printBoard(board):
#     '''
#     显示棋盘当前状态
#     '''
#     boardShow='''
#                     |   |
#                   %s | %s | %s
#                     |   |
#                 -------------
#                     |   |
#                   %s | %s | %s
#                     |   |
#                 -------------
#                     |   |
#                   %s | %s | %s
#                     |   |
#             '''
#     charShow=[]
#     for c in board:
#         if c==0:
#             charShow.append(' ')
#         elif c==1:
#             charShow.append('O')
#         else:
#             charShow.append('X')
#     print(boardShow%tuple(charShow[6:9]+charShow[3:6]+charShow[0:3]))


# def playerDrop(board):
#     '''
#     玩家落子，改变棋盘状态即可
#     '''
#     gridNo=int(input('请选择你要落子的位置(1-9):'))
#     while gridNo not in range(1,10) or board[gridNo-1]!=0:
#         gridNo=int(input('非法落子位置，请重新选择(1-9):'))
#     return gridNo-1


# def chessCountValue(chessCount):
#     """
#     返回对应棋子状态的估价
#     """
#     if chessCount[1]==3:
#         return 1000
#     if chessCount[2]==3:
#         return -1000
#     if chessCount[1]==2 and chessCount[0]==1:
#         return 50
#     elif chessCount[1]==1 and chessCount[0]==2:
#         return 10
#     if chessCount[2]==2 and chessCount[0]==1:
#         return -50
#     elif chessCount[2]==1 and chessCount[0]==2:
#         return -10
#     return 0


# def evaluate(board):
#     '''
#     估价函数，判断当前棋盘的估价值
#     '''
#     # 判断每一行、每一列、每条对角线的棋子状态，不同的状态有不同的估价值
#     # 棋子状态估价排序（高到低）：
#     #   三子一线，直接返回最大估价
#     #   两子一个空
#     #   一子两个空
#     board=[board[6:9],board[3:6],board[0:3]] # 将board转化为二维矩阵便于判断棋子状态
#     chessCount=[0,0,0] # 记录棋子状态
#     value=0
#     for i in range(0,3):
#         chessCount=[0,0,0]
#         for j in range(0,3):
#             chessCount[board[i][j]]+=1 # 判断行
        
#         tempValue=chessCountValue(chessCount)
#         if abs(tempValue)==1000:
#             return tempValue
#         else:
#             value+=tempValue
        
#         chessCount=[0,0,0]
#         for j in range(0,3):
#             chessCount[board[j][i]]+=1 # 判断列
        
#         tempValue=chessCountValue(chessCount)
#         if abs(tempValue)==1000:
#             return tempValue
#         else:
#             value+=tempValue
        
#     chessCount=[0,0,0]
#     for p in range(0,3):
#         chessCount[board[p][p]]+=1 # 判断正对角线

#     tempValue=chessCountValue(chessCount)
#     if abs(tempValue)==1000:
#         return tempValue
#     else:
#         value+=tempValue
    
#     chessCount=[0,0,0]
#     for p in range(0,3):
#         chessCount[board[p][2-p]]+=1 # 判断负对角线

#     tempValue=chessCountValue(chessCount)
#     if abs(tempValue)==1000:
#         return tempValue
#     else:
#         value+=tempValue
        
#     return value


# def searchTree(board,whosTurn):
#     '''
#     深度优先搜索，在落子一个位置后，直到棋盘满，这个过程产生的估价值
#     param:
#         board:棋盘状态
#         whosTurn:当前落子方，等于1表示当前由电脑落子，等于2表示由玩家落子
#     '''
#     maxValue=-100000 # 用来存储对当前落子方来说最大的价值
#     minValue=100000 # 用来存储对当前落子方来说最小的价值
    
#     value=evaluate(board) # 计算当前棋盘状态的价值
#     isFull=True # 棋盘是否已经下满
#     if abs(value)>=1000: # 若当前棋盘状态已经达到最大值或最小值，返回即可
#         return value
    
#     # 遍历每个棋盘位置
#     for i in range(0,9):
#         if board[i]!=0: # 不等于0表示已经有棋子落在了该处，则继续判断下一个位置
#             continue
#         isFull=False # 执行到这一步说明棋盘没有满，将isFull赋值为False
#         board[i]=whosTurn # 将当前落子方的棋子下在当前遍历的位置

#         # 递归调用，继续向下进行搜索
#         # 请在此添加代码，勿改动其它代码
#         #-----------Begin----------
#         tempValue=searchTree(board,((whosTurn-1)^1)+1)
#         #------------End-----------

#         maxValue=max(maxValue,tempValue) # 判断下在当前遍历的位置后产生的价值是否大于maxValue
#         minValue=min(minValue,tempValue) # 判断下在当前遍历的位置后产生的价值是否小于minValue
#         board[i]=0 # 将被落子的位置清空，继续循环判断下在其他位置后棋盘的价值
#     if isFull: # 若isFull为True，表示期满已满，返回进入该函数时棋盘的价值
#         return value
#     if whosTurn==1: # 若当前落子方为电脑，返回最大价值
#         return maxValue
#     if whosTurn==2: # 若当前落子方为玩家，返回最小价值
#         return minValue


# def computerDrop(board):
#     '''
#     遍历棋盘，计算在可以落子的位置落子后的估价函数值，选择值最高的位置落子
#     '''
#     mapValueToGrid=[]
#     for i in range(0,9):
#         if board[i]!=0:
#             continue
#         board[i]=1
#         mapValueToGrid.append([searchTree(board,2),i])
#         board[i]=0
#     mapValueToGrid=sorted(mapValueToGrid,key=lambda x: x[0],reverse=True)
#     # board[mapValueToGrid[0][1]]=1
#     # printBoard(board)
#     # print('各位置估价：',mapValueToGrid)
#     return mapValueToGrid[0][1]


# def thisRoundState(board):
#     '''
#     判断刚刚落子的一方是否获胜（电脑胜出还是玩家胜出）
#     '''
#     # chessType=whosTurn # 表示刚刚落子的一方的棋子类型
#     value=evaluate(board)
#     emptyGrid=0
#     if value==1000:
#         return 1 # 电脑获胜
#     elif value==-1000:
#         return 2 # 玩家获胜
#     else:
#         for i in range(0,9):
#             if board[i]==0:
#                 emptyGrid+=1
#         if emptyGrid==0:
#             return 3 # 平局
#         else:
#             return 0 # 棋局未结束


# def gameBegin():
#     '''
#     主循环，电脑玩家依次落子，电脑每次落子前选择最优位置落子，玩家根据输入落子
#     '''
#     continueGame=True
#     while continueGame:
#         print('''
#             \n\t 游戏说明：
#             \n\t 1、游戏棋盘是以 3 X 3 的九宫格形式。每格坐标与键盘上的小键盘 1 - 9 数字键一一对应。
#             \n\t 2、游戏中，玩家与电脑轮流下棋，输入数字，将棋下在九宫格的相应格里。
#             \n\t 3、首先实现横线、竖线或斜线连续三个格棋子一样的获胜。
#             ''')
#         whoGoesFirst=(input('谁执先手？(输入1电脑先手，玩家先手请按其他键):'))
#         computersTurn=1
#         playersTurn=2
#         whosTurn=computersTurn if whoGoesFirst=='1' else playersTurn

#         # 0表示可以落子，1表示电脑已落子，2表示玩家已落子
#         board=[0,0,0,0,0,0,0,0,0]

#         # 游戏开始
#         gameRunning=True
#         printBoard(board)
#         while gameRunning:
#             if whosTurn==playersTurn:
#                 playerDropGrid=playerDrop(board)
#                 board[playerDropGrid]=2
#                 whosTurn=computersTurn
#             elif whosTurn==computersTurn:
#                 computerDropGrid=computerDrop(board)
#                 board[computerDropGrid]=1
#                 whosTurn=playersTurn
#             printBoard(board)
#             state=thisRoundState(board)
#             if state==1:
#                 print('电脑获胜，你输了！')
#             elif state==2:
#                 print('你赢了，恭喜！')
#             elif state==3:
#                 print('平局！')
#             if state!=0:
#                 continueGameCode=(input('继续游戏请输入1，退出请按其他键：'))
#                 if continueGameCode=='1':
#                     continueGame=True
#                 else:
#                     continueGame=False
#                 break


# gameBegin()
print(' ')