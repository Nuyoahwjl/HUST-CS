
item_one = 'one+'
item_two = 'two+'
item_three = 'three'
total = item_one + \
        item_two + \
        item_three
print(total)

days = ['Monday', 'Tuesday', 'Wednesday', 'thursday', 'Friday',
        'Saturday', 'Sunday']
print(days)

you_love_LaTeX = 1
you_dont_love_LateX = 1 - you_love_LaTeX
if you_love_LaTeX == 1:
    score = 95
    print(score)
elif you_dont_love_LateX == 1:
    print('Please work hard on LaTeX')
else:
    score = 65
    print(score)

def plus(a,b):
    c = a + b
    return c

d = plus(1,2)
print(d)

def PLUS(a=1,b=2):
    c = a + b
    return c

d = PLUS(b=1)
print(d)
