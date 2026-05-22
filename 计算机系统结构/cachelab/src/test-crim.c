/*
 * test-csim.c - Checks the correctness of a student's test cache
 * simulator (csim) by comparing its output to a reference simulator
 * provided by the instructors (csim-ref).
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <errno.h>
#include <getopt.h>

#define MAX_STR 1024  /* Max string size */

/* 
 * usage - Prints usage info
 */
void usage(char *argv[]){
    printf("Usage: %s [-hv]\n", argv[0]);
    printf("Options:\n");
    printf("  -h    Print this help message.\n");
}

/*
 * SIGALRM handler 
 */
void sigalrm_handler(int signum)
{
    printf("Error: Program timed out.\n");
    printf("TEST_CSIM_RESULTS=0,0,0,0,0,0\n");
    exit(1);
}

/* 
 * runtrace - Runs the reference and trace simulators on a particular
 * trace and set of cache parameters, and collects the results for the
 * caller. Return 0 if any problems, 1 if OK.
 */
int runtrace(int s, int E, int b, char *trace, /* in */
             int *ref_hits, int *ref_misses, int *ref_evictions,    /* out */
             int *test_hits, int *test_misses, int *test_evictions) /* out */
{
    FILE *fp;
    int status;
    char cmd[MAX_STR];

    /* Run the reference simulator */
    sprintf(cmd, "./csim-ref -s %d -E %d -b %d -t %s > /dev/null", 
            s, E, b, trace);
    system("rm -rf .csim_results");
    status = system(cmd);
    if (status == -1) {
        fprintf(stderr, "Error invoking system() for reference sim: %s\n", 
                strerror(errno));
        return 0;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        fprintf(stderr, "Error running reference simulator: Status %d\n", 
                WEXITSTATUS(status));
        return 0;
    }

    /* Get the results from the reference simulator */
    fp = fopen(".csim_results", "r");
    if (!fp) {
        fprintf(stderr, "Error: Results for reference simulator not found. Use the printSummary() function\n");
        return 0;
    }
    fscanf(fp, "%d %d %d", ref_hits, ref_misses, ref_evictions);
    fclose(fp);

    /* Run the test simulator */
    sprintf(cmd, "./csim -s %d -E %d -b %d -t %s > /dev/null", 
            s, E, b, trace);
    system("rm -rf .csim_results");
    status = system(cmd);
    if (status == -1) {
        fprintf(stderr, "Error invoking system() for test sim: %s\n", strerror(errno));
        return 0;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
        fprintf(stderr, "Error running test simulator: Status %d\n", 
                WEXITSTATUS(status));
        return 0;
    }

    /* Get the results from the test simulator */
    fp = fopen(".csim_results", "r");
    if (!fp) {
        fprintf(stderr, "Error: Results for test simulator not found. Use the printSummary() function\n");
        return 0;
    }
    fscanf(fp, "%d %d %d", test_hits, test_misses, test_evictions);
    fclose(fp);

    return 1;
}

/*
 * test_csim - Check the student's test simulator for correctness by
 * comparing its results to the reference simulator.
 */

#define N 8  /* Number of tests */

void test_csim()
{
    int i;
    char buf[MAX_STR];

    /* Specify the tests */
    int s[N] = {1, 4, 2, 2, 2, 2, 5, 5};
    int E[N] = {1, 2, 1, 1, 2, 4, 1, 1};
    int b[N] = {1, 4, 4, 3, 3, 3, 5, 5};
    int weight[N] = {1, 1, 1, 1, 1, 1, 1, 2}; 
    char *trace[N] = {"traces/yi2.trace", "traces/yi.trace", "traces/dave.trace", 
                      "traces/trans.trace", "traces/trans.trace", 
                      "traces/trans.trace", "traces/trans.trace", 
                      "traces/long.trace"};

    /* Output results */
    int ref_hits[N], ref_misses[N], ref_evictions[N];
    int test_hits[N], test_misses[N], test_evictions[N];
    int status[N];
    int points[N];
    int total_points = 0;

    /* Run the individual tests */
    for (i=0; i < N; i++) {

        /* Run test */
        /*printf("Testing (s,E,b)=(%d,%d,%d) on file %s\n", 
          s[i], E[i], b[i], trace[i]);*/
        status[i] = runtrace(s[i], E[i], b[i], trace[i], 
                             &ref_hits[i], &ref_misses[i], 
                             &ref_evictions[i],
                             &test_hits[i], &test_misses[i], 
                             &test_evictions[i]);

        /* If the test had any problems, null everything out */
        if (status[i] == 0) {
            ref_hits[i] = ref_misses[i] = ref_evictions[i] = -1;
            test_hits[i] = test_misses[i] = test_evictions[i] = -1;
        }
    }

    /* Compute the points earned for each trace */
    for (i=0; i < N; i++) {
        points[i] = 0;
        if (status[i] != 0 && ref_hits[i] != -1 && ref_misses[i] != -1 &&
            ref_evictions[i] != -1 && test_hits[i] != -1 && 
            test_misses[i] != -1 && test_evictions[i] != -1) {
            points[i] += (ref_hits[i] == test_hits[i]) * weight[i];
            points[i] += (ref_misses[i] == test_misses[i]) * weight[i];
            points[i] += (ref_evictions[i] == test_evictions[i]) * weight[i];
        }
        total_points += points[i];
    }

    /* Display a summary of results */
    printf("%38s%24s\n", "Your simulator", "Reference simulator");
    printf("%6s%8s%8s%8s%8s%8s%8s%8s\n", 
           "Points", "(s,E,b)",
           "Hits", "Misses", "Evicts",
           "Hits", "Misses", "Evicts");

    for (i=0; i < N; i++) {
        sprintf(buf, "(%d,%d,%d)", s[i], E[i], b[i]);
        printf("%6d%8s", points[i], buf);
        printf("%8d%8d%8d%8d%8d%8d  %s\n", 
               test_hits[i], test_misses[i], test_evictions[i], 
               ref_hits[i], ref_misses[i], ref_evictions[i],
               trace[i]);
    }
    printf("%6d\n", total_points);

    /* Print a compact summary string for the driver */
    printf("\nTEST_CSIM_RESULTS=%d\n", total_points);
}

/*
 * main - Main routine
 */
int main(int argc, char* argv[]){
    char c;

    /* Parse command line args */
    while ((c = getopt(argc, argv, "h")) != -1) {
        switch(c) {
        case 'h':
            usage(argv);
            exit(0);
        default:
            usage(argv);
            exit(1);
        }
    }

    /* Install timeout handler */
    if (signal(SIGALRM, sigalrm_handler) == SIG_ERR) {
        fprintf(stderr, "Unable to install SIGALRM handler\n");
        exit(1);
    }

    /* Time out and give up after a while */
    alarm(20);

    /* Evaluate the student's cache simulator for correctness */
    test_csim();

    exit(0);
}
