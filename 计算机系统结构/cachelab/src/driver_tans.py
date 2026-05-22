#coding:utf8
#
# driver.py - The driver tests the correctness of the student's cache
#     simulator and the correctness and performance of their transpose
#     function. It uses ./test-csim to check the correctness of the
#     simulator and it runs ./test-trans on three different sized
#     matrices (32x32, 64x64, and 61x67) to test the correctness and
#     performance of the transpose function.
#
import subprocess;
import re;
import os;
import sys;
import optparse;

#
# computeMissScore - compute the score depending on the number of
# cache misses
#
def computeMissScore(miss, lower, upper, full_score):
    if miss <= lower:
        return full_score
    if miss >= upper: 
        return 0

    score = (miss - lower) * 1.0 
    range = (upper- lower) * 1.0
    return round((1 - score / range) * full_score, 1)

#
# main - Main function
#
def main():

    # Configure maxscores here
    maxscore= {};
    maxscore['csim'] = 27
    maxscore['transc'] = 1
    maxscore['trans32'] = 8
    maxscore['trans64'] = 8
    maxscore['trans61'] = 10

    # Parse the command line arguments 
    p = optparse.OptionParser()
    p.add_option("-A", action="store_true", dest="autograde", 
                 help="emit autoresult string for Autolab");
    opts, args = p.parse_args()
    autograde = opts.autograde

    # Check the correctness and performance of the transpose function
    # 32x32 transpose
    print " 测试矩阵转置"
    #print "运行 ./test-trans -M 32 -N 32"
    p = subprocess.Popen("./test-trans -M 32 -N 32 | grep TEST_TRANS_RESULTS", 
                         shell=True, stdout=subprocess.PIPE)
    stdout_data = p.communicate()[0]
    result32 = re.findall(r'(\d+)', stdout_data)
    
    # 64x64 transpose
    #print "运行 ./test-trans -M 64 -N 64"
    p = subprocess.Popen("./test-trans -M 64 -N 64 | grep TEST_TRANS_RESULTS", 
                         shell=True, stdout=subprocess.PIPE)
    stdout_data = p.communicate()[0]
    result64 = re.findall(r'(\d+)', stdout_data)
    
    # 61x67 transpose
    #print "运行 ./test-trans -M 61 -N 67"
    p = subprocess.Popen("./test-trans -M 61 -N 67 | grep TEST_TRANS_RESULTS", 
                         shell=True, stdout=subprocess.PIPE)
    stdout_data = p.communicate()[0]
    result61 = re.findall(r'(\d+)', stdout_data)
    
    # Compute the scores for each step
    trans_cscore = int(result32[0]) * int(result64[0]) * int(result61[0]);
    miss32 = int(result32[1])
    miss64 = int(result64[1])
    miss61 = int(result61[1])
    trans32_score = computeMissScore(miss32, 300, 600, maxscore['trans32']) * int(result32[0])
    trans64_score = computeMissScore(miss64, 1300, 2000, maxscore['trans64']) * int(result64[0])
    trans61_score = computeMissScore(miss61, 2000, 3000, maxscore['trans61']) * int(result61[0])
    total_score =  trans32_score + trans64_score + trans61_score

    # Summarize the results
    print "\n实验汇总:"
    print "%-22s%8s%10s%12s" % ("转置矩阵类型", "得分", "该项分值", "是否有效")


    misses = str(miss32)
    if miss32 == 2**31-1 :
        misses = "invalid"
    print "%-22s%8.1f%10d%12s" % ("32x32", trans32_score, 
                                  maxscore['trans32'], misses)

    misses = str(miss64)
    if miss64 == 2**31-1 :
        misses = "invalid"
    print "%-22s%8.1f%10d%12s" % ("64x64", trans64_score, 
                                  maxscore['trans64'], misses)

    misses = str(miss61)
    if miss61 == 2**31-1 :
        misses = "invalid"
    print "%-22s%8.1f%10d%12s" % ("61x67", trans61_score, 
                                  maxscore['trans61'], misses)

    print "  %22s%8.1f%10d" % ("合  计", total_score,
                             maxscore['trans32'] + 
                             maxscore['trans64'] +
                             maxscore['trans61'])
    if total_score<=0.0:
        print "程序没有改进"
    if trans32_score >=1:
        print "矩阵32x32转置优化完成，"
    if trans64_score >=1:
        print "矩阵64x64转置优化完成，"
    if trans61_score >=1 :
        print "矩阵61x67转置优化完成!"
    
    # Emit autoresult string for Autolab if called with -A option
    if autograde:
        autoresult="%.1f:%d:%d:%d" % (total_score, miss32, miss64, miss61)
        print "\nAUTORESULT_STRING=%s" % autoresult
    
    
# execute main only if called as a script
if __name__ == "__main__":
    main()

