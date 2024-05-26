def sayhello2techer_Chen(name):
    sayhello = name + "hello, how do you do?"
    print(sayhello)
    return sayhello

if __name__ == '__main__': # 以下内容无法被其他.py所import
    sayhello = sayhello2techer_Chen("Jiazhong Chen, ")
    def omg(student):
        print(student, "says:", sayhello)

    name = input("你的姓名: ")
    omg(name)