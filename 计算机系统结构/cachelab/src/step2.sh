#!/bin/bash
rm ./task2/* > /dev/null 2<&1
rm ./src/trans.o  > /dev/null 2<&1
rm ./src/test-trans  > /dev/null 2<&1
rm ./src/tracegen  > /dev/null 2<&1
cd ./src  > /dev/null 2<&1
gcc -g -Wall -Werror -std=c99 -m64 -O0 -c trans.c >/dev/null 2>&1
gcc -g -Wall -Werror -std=c99 -m64 -o test-trans test-trans.c cachelab.c trans.o  >/dev/null 2>&1
gcc -g -Wall -Werror -std=c99 -m64 -O0 -o tracegen tracegen.c trans.o cachelab.c >/dev/null 2>&1
python driver_tans.py > info 2>&1
cat info | grep "程序没有改进" >result 2>&1
if [ $? -eq 0 ]; then
    cat info
else
    tail info -n 3
fi
cat info > ../task2/step2.txt   2>&1