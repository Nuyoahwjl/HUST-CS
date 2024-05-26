# regular_expression.py
# 写出一个正则表达式来匹配是否是手机号
import re
phone_rule = re.compile('1\d{10}') # 第一个必须是1. 后面10个字符必须是数字, d{10}: 规则要匹配的是10个数字
phone_num = input('请输入一个手机号: ') #通过规则去匹配字符串
if len(phone_num) != 11:
    print('手机号应该是11位数字')
elif len(phone_num) == 11:
    sample_result = phone_rule.search(phone_num) # 判断phone_num的第一个数字是否为1, 后面字符是否为数字.
    if sample_result != None:
        print('这是一个手机号')
        print(sample_result.group())
    else:
        print('这不是一个手机号')
