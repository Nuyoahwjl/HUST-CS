 /********** 使用int变量x,p和n，在下面写出满足任务要求的表达式**********/
 /**********Begin**********/
 //x=(x>>p+1<<p+1)|((unsigned)x<<sizeof(int)*8-1-p+n>>sizeof(int)*8-1-p+n)|((unsigned)(~x)>>p-n+1<<sizeof(int)*8-n>>sizeof(int)*8-1-p)
 x=(0xffffffff<<p-n+1)&(0xffffffff>>31-p)^x
 /**********End**********/