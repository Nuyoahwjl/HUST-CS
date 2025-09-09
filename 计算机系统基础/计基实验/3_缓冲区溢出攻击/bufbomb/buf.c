/* Buffer procedures used by buffer bomb */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>

char *Gets(char *, char *, int);

#define NORMAL_BUFFER_SIZE 32

/* Buffer size for getbufn */
#define KABOOM_BUFFER_SIZE 512
#define INPUT_STRING_SIZE 4096
#define BYTE_BUFFER_INITIAL_SIZE 1024
#define COMMENT_START "/*"
#define COMMENT_END "*/"

unsigned char convert_to_hex_value(char *input)
{
     unsigned val;
     sscanf(input, "%x", &val);
     return (unsigned char)val;
}

unsigned char *convert_to_byte_string(FILE *file_in, int *size)
{
     int comment_level = 0;
     int input_string_size = INPUT_STRING_SIZE;
     char input[input_string_size];

     int byte_buffer_size = BYTE_BUFFER_INITIAL_SIZE;
     int byte_buffer_offset = 0;

     unsigned char *byte_buffer = (unsigned char *)malloc(BYTE_BUFFER_INITIAL_SIZE * sizeof(*byte_buffer));
     if (byte_buffer == NULL)
          return NULL;

     while (fscanf(file_in, "%s", input) > 0)
     {
          // Case 1: 进入注释部分
          if (!strcmp(input, COMMENT_START))
          {
               comment_level++;
               continue;
          }

          // Case 2:  注释部分结束
          if (!strcmp(input, COMMENT_END))
          {
               if (comment_level <= 0)
               {
                    // make sure we are in a comment-block
                    fprintf(stderr, "Error: stray %s found.\n", COMMENT_END);
                    free(byte_buffer);
                    return NULL;
               }
               comment_level--;
               continue;
          }

          // Case 3: Convert data to hex value and store
          if (comment_level == 0)
          {
               // we should have read a hex value and print it out.
               if (!isxdigit(input[0]) || !isxdigit(input[1]) || (input[2] != '\0'))
               {
                    fprintf(stderr, "Invalid hex value [%s]. "
                                    "Please specify only single byte hex values separated by whitespace.\n",
                            input);
                    free(byte_buffer);
                    return NULL;
               }

               unsigned char b = convert_to_hex_value(input);
               // see if we have enough room in the buffer...
               if (byte_buffer_offset == byte_buffer_size)
               {
                    byte_buffer = (unsigned char *)realloc(byte_buffer, 2 * byte_buffer_size);
                    if (byte_buffer == NULL)
                         return NULL;

                    byte_buffer_size *= 2;
               }
               byte_buffer[byte_buffer_offset++] = b;
          }
     }

     *size = byte_buffer_offset;
     return byte_buffer;
}

int getbuf(char *src, int len)
{
#ifdef U0
     char temp0[12] = "good luck";
#endif
#ifdef U1
     char temp1[] = "have a good day";
#endif
#ifdef U2
     char temp2[] = "The future will be better tomorrow";
#endif
#ifdef U3
     char temp3[10] = "computer";
#endif
#ifdef U4
     char temp4[12] = "language";
#endif
#ifdef U5
     char temp5[] = "hello";
#endif
#ifdef U6
     char temp6[12] = "foundation";
#endif
#ifdef U7
     char temp7[] = "Believe in yourself";
#endif
#ifdef U8
     char temp8[8] = "123456";
#endif
#ifdef U9
     char temp9[] = "bomb";
#endif

     char buf[NORMAL_BUFFER_SIZE];
     printf("buf_location:%p\n",buf);
     Gets(buf, src, len);
     return 1;
}

int gencookie(char *s)
{
     if (strlen(s) != 10)
     {
          printf("length of userid  must be 10. \n");
          return 0;
     }
     if (s[0] != 'U' && s[0] != 'u')
     {
          printf("student id  satrt with U. \n");
          return 0;
     }
     for (int i = 1; i < 10; i++)
          if (s[i] < '0' || s[i] > '9')
          {
               printf("stuid must be digitals. \n");
               return 0;
          }
     return atoi(s + 1);
}
