#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <string.h>

FILE *infile = NULL;
int cookie = 0; /* unique cookie computed from userid */
int gencookie(char *s);

char *Gets(char *, char *, int);
int getbuf(char *, int);
unsigned char *convert_to_byte_string(FILE *file_in, int *size);

struct env_info
{
    char file_name[100];
    char userid[20];
    unsigned int cookie;
    unsigned int level;
};

void initialize_bomb(char *userid)
{
    int len;
    int x = 10, y = 20, z = 30, u = 40, v = 50, w = 60;
    len = strlen(userid);
    if (len != 10)
    {
        printf("The student number has 10 characters, such as U202215001 \n");
        exit(0);
    }
    printf("welcome  %s \n", userid);
#ifdef U0
    if (userid[len - 1] == '0')
    {
        x = 100;
        x = ~x;
        w = 2 * x;
        u = x + y + z;
        return;
    }
#endif
#ifdef U1
    if (userid[len - 1] == '1')
    {
        w = 2 * x + 3 * y;
        return;
    }
#endif
#ifdef U2
    if (userid[len - 1] == '2')
    {
        w = 2 * x + 3 * y + 4 * u;
        return;
    }
#endif
#ifdef U3
    if (userid[len - 1] == '3')
    {
        w = 2 * x + 3 * y + 4 * u + x; // 250
        return;
    }
#endif
#ifdef U4
    if (userid[len - 1] == '4')
    {
        w = 2 * x + 3 * y + 4 * u + x + 5 * w;
        return;
    }
#endif
#ifdef U5
    if (userid[len - 1] == '5')
    {
        w = -u;
        return;
    }
#endif
#ifdef U6
    if (userid[len - 1] == '6')
    {
        w = -u - v;
        return;
    }
#endif
#ifdef U7
    if (userid[len - 1] == '7')
    {
        return;
    }
#endif
#ifdef U8
    if (userid[len - 1] == '8')
    {
        x = 4 * w + 10;
        return;
    }
#endif
#ifdef U9
    if (userid[len - 1] == '9')
    {
        x = 4 * w + 5 * y;
        return;
    }
#endif
    printf(" gcc  -g -o binarybomb -D U* bomb.c  support.c  phase.o\n");
    printf(" U* : * is the last number of your Student Id . \n");
    printf(" example :  U202215001  ->   -D U1 . \n");
    exit(0);
}

/* 第 0 级 ： smoke
 * 从函数  getbuf() 返回时, 要求执行 smoke 函数 ，而不是返回到主调函数 test().
 */
void smoke()
{
    printf("Smoke!: You called smoke()\n");
    exit(0);
}

/* 第 1 级 ： fizz
 * 要求执行该函数（或者函数的部分语句） ，能够显示出 "Fizz!:.....”，显示的val值应正确
 */
void fizz(int val)
{
    if (val == cookie)
    {
        printf("Fizz!: You called fizz(0x%x)\n", val);
    }
    else
        printf("Misfire: You called fizz(0x%x)\n", val);
    exit(0);
}

/* 第 2 级 ： bang
 * 要求执行该函数（或者函数的部分语句） ，能够显示出 "Bang!:.....”，
 *       显示的 global_value 值应正确
 */
int global_value = 0;
/*
void ftemp()
{
     global_value=cookie;
}
*/

void bang(int val)
{
    if (global_value == cookie)
    {
        printf("Bang!: You set global_value to 0x%x\n", global_value);
    }
    else
        printf("Misfire: global_value = 0x%x\n", global_value);
    exit(0);
}

/* 第 3 级 ： boom
 * 要求执行 getbuf 函数后，将cookie 值返回，能够正确回到  test 的调用处继续执行
 *       显示  Boom!: ...... , 显示的值应正确
 */

void test(struct env_info *p)
{
    unsigned char *byte_buffer;
    int byte_buffer_size;
    int val;
    FILE *fp;
    int level = 0;

    if (p->level == 0 || p->level == 1 || p->level == 2 || p->level == 3)
    {
        fp = fopen(p->file_name, "r");
        if (fp == NULL)
        {
            printf("please check file name %s \n", p->file_name);
            return;
        }
        byte_buffer = convert_to_byte_string(fp, &byte_buffer_size);
        fclose(fp);
        val = getbuf(byte_buffer, byte_buffer_size);
    }

    if (val == cookie)
    {
        printf("Boom!: getbuf returned 0x%x\n", val);
    }
    else
    {
        printf("Dud: getbuf returned 0x%x\n", val);
    }
}

/*
 * Gets - Like gets(), except that can optionally (when hexformat
 * nonzero) accept format where characters are typed as pairs of hex
 * digits.  Nondigit characters are ignored.  Stops when encounters
 * newline.  In addition, it stores the string in global buffer
 * gets_buf.
 */
#define GETLEN 1024

int gets_cnt = 0;
char gets_buf[3 * GETLEN + 1];

static char trans_char[16] =
    {'0', '1', '2', '3', '4', '5', '6', '7',
     '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

static void save_char(char c)
{
    if (gets_cnt < GETLEN)
    {
        gets_buf[3 * gets_cnt] = trans_char[(c >> 4) & 0xF];
        gets_buf[3 * gets_cnt + 1] = trans_char[c & 0xF];
        gets_buf[3 * gets_cnt + 2] = ' ';
        gets_cnt++;
    }
}

static void save_term()
{
    gets_buf[3 * gets_cnt] = '\0';
}

char *Gets(char *dest, char *src, int len)
{
    memcpy(dest, src, len);
    return dest;
}

int main(int argc, char *argv[])
{
    struct env_info input_args;

    if (argc < 4)
    {
        printf("usage : %s <stuid> <string_file>  <level> \n", argv[0]);
        printf("Example :  ./bufbomb  U202115001 smoke_hex.txt  0  \n");
        return 0;
    }

    strcpy(input_args.userid, argv[1]);
    strcpy(input_args.file_name, argv[2]);
    input_args.level = atoi(argv[3]);
    printf("user id : %s \n", input_args.userid);
    cookie = gencookie(input_args.userid);
    printf("cookie : 0x%x \n", cookie);
    printf("hex string file : %s \n", input_args.file_name);
    printf("level : %d \n", input_args.level);

    printf("smoke : 0x%p   fizz : 0x%p  bang : 0x%p \n", smoke, fizz, bang);

    initialize_bomb(input_args.userid);

    test(&input_args);

    printf("bye bye , %s\n", input_args.userid);
    return 0;
}
