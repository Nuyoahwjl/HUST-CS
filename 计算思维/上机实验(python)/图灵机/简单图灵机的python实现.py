import re
if __name__=="__main__":
    initial_state = "S1" #图灵机初始状态
    final_states = "S7"
    #请根据图灵机的状态转换图把下面状态转移表的字典填写完整
    #图灵机状态转移表如下
    transition_function = {("S1", " "): ("", "R", "S1"),
                           ("S1", "V"): ("V", "R", "S2"),
                           ("S2", " "): ("", "R", "S2"),
                           ("S2", "="): ("=", "R", "S3"),
                           ("S3", " "): ("", "R", "S3"),
                           ("S3", "C"): ("C", "R", "S4"),
                           ("S4", " "): ("", "R", "S4"),
                           ("S4", "+"): ("+", "R", "S5"),
                           ("S4", ";"): ("", "N", "S7"),
                           ("S5", " "): ("", "R", "S5"),
                           ("S5", "C"): ("C", "R", "S6"),
                           ("S6", " "): ("", "R", "S6"),
                           ("S6", ";"): ("", "N", "S7")
                           }
    tape_alphabet = ["V", "C", " ", "=", "+", ";"] #图灵机字母表
    
    print("请输入纸带上内容：")
    s = input()
    print("Input on Tape:")
    print(s)
    s1 = re.sub(r'[_a-zA-Z]\w*', "V", s) #通过正则表达式匹配变量标识符并转换为字母'V'
    s2 = re.sub(r'0|[1-9]\d*', "C", s1)  #通过正则表达式匹配正整数并转换为字母'C'
    position = 0 #图灵机的当前读取位置
    current_state = initial_state #初始化图灵机当前状态
    tape = list(s2) #构建图灵机要识别的字母列表，内容如[' ', 'V', ' ', '=', 'C', ' ']
    flag = True #是否匹配特定模式的标记
    while current_state != final_states:
        char = tape[position] #图灵机当前读取到的字母
        x = (current_state, char)
        if x in transition_function: #查询图灵机状态转移表确定下一步操作
            y = transition_function[x]
            print(str(x+y)) 
            #请根据注释在此添加代码，勿改动其它代码
            #改变纸带上字母
            #-----------Begin----------
            tape[position] = y[0]
            #------------End-----------
            if y[1] == "R": #右移
                # 请在此添加代码，勿改动其它代码
                #-----------Begin----------
                position += 1
                #------------End-----------
            elif y[1] == "N": #左移
                position -= 1
            
            #请根据注释在此添加代码，勿改动其它代码
            #图灵机进入下一个状态
            #-----------Begin----------
            current_state = y[2]
            #------------End-----------          
            if position < 0 or position >= len(tape): #超出纸带范围
                print("最终状态不是S7: false")
                flag = False
                break;
        else: #当前状态与当前输入不在状态转移表中，模式匹配失败
            print(str(x) + ": false")
            flag = False
            break;
    if flag:
        print("Result of Turing machine calculation:")
        print(''.join(tape))