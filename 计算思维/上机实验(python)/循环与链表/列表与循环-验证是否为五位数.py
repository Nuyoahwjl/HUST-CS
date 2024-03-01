#请验证输入的列表N_list中的整数是否为五位数，并返回五位数整数的最高位数值


N_list = [int(i) for i in input().split(',')]
#   请在此添加实现代码   #
# ********** Begin *********#
# def validate_five_digit_numbers(N_list):
highest_list = []
for num in N_list:
    if len(str(num)) == 5:
         highest_list.append(int(num/10000))
print(highest_list)

# N_list = [int(i) for i in input().split(',')]
# result=validate_five_digit_numbers(N_list)
# print(result)




# ********** End **********#

