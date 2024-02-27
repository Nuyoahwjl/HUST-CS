import numpy as np
a =float(input('请输入第一位同学的成绩:'))
b =float(input('请输入第二位同学的成绩:'))
c =float(input('请输入第三位同学的成绩:'))
d =float(input('请输入第四位同学的成绩:'))
e =float(input('请输入第五位同学的成绩:'))

score = [a, b, c,d,e]
score = np.array(score) # 不转类型也能运行
for k in range(len(score)):
    for i in range(len(score) - 1):
        if score[i] < score[i+1]:
            temp = score[i]
            score[i] = score[i+1]
            score[i+1] = temp
        print(score)
