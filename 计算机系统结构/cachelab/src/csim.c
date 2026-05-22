/*
 *csim.c-使用C编写一个Cache模拟器，它可以处理来自Valgrind的跟踪和输出统计
 *信息，如命中、未命中和逐出的次数。更换政策是LRU。
 *设计和假设:
 *  1. 每个加载/存储最多可导致一个缓存未命中。（最大请求是8个字节。）
 *  2. 忽略指令负载（I），因为我们有兴趣评估trace.c内容中数据存储性能。
 *  3. 数据修改（M）被视为加载，然后存储到同一地址。因此，M操作可能导致两次缓存命中，或者一次未命中和一次命中，外加一次可能的逐出。
 *使用函数printSummary() 打印输出，输出hits, misses and evictions 的数，这对结果评估很重要
 */
#include "cachelab.h"
//                    请在此处添加代码
//****************************Begin*********************
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <errno.h>

// #define DEBUG_ON
#define ADDRESS_LENGTH 64

/* 类型：内存地址 */
typedef unsigned long long int mem_addr_t;

/* 类型：Cache行
LRU是用于实现LRU替换策略的计数器*/
typedef struct cache_line
{
    char in_use;                // 有效位
    mem_addr_t tag_id;          // 标记位
    unsigned long long int age; // LRU计数器
} line_t;

typedef line_t *set_t;
typedef set_t *cache_space_t;

/* 全局变量由命令行参数设置 */
int verbose_mode = 0; /* 如果设置，则打印跟踪 */
int s_bits = 0;       /* 组索引位数 */
int b_bits = 0;       /* 块偏移位数 */
int assoc_ways = 0;   /* 相联度 */
char *trace_path = NULL;

/* 从命令行参数派生 */
int set_count; /* 组数 */

/* 用于记录缓存统计信息的计数器 */
int miss_total = 0;                  // 缺失计数器
int hit_total = 0;                   // 命中计数器
int eviction_total = 0;              // 逐出计数器
unsigned long long int lru_tick = 1; // LRU计数器

/* cache */
cache_space_t cache_bank;
mem_addr_t current_set_index;

static inline mem_addr_t pick_set_index(mem_addr_t addr)
{
    return (addr >> b_bits) & ((1ULL << s_bits) - 1ULL);
}

static inline mem_addr_t pick_tag(mem_addr_t addr)
{
    return addr >> (s_bits + b_bits);
}

static int probe_hit_slot(mem_addr_t set_index, mem_addr_t tag_id)
{
    for (int i = 0; i < assoc_ways; ++i)
    {
        if (cache_bank[set_index][i].in_use && cache_bank[set_index][i].tag_id == tag_id)
        {
            return i;
        }
    }
    return -1;
}

static int probe_free_slot(mem_addr_t set_index)
{
    for (int i = 0; i < assoc_ways; ++i)
    {
        if (!cache_bank[set_index][i].in_use)
        {
            return i;
        }
    }
    return -1;
}

static int choose_victim_slot(mem_addr_t set_index)
{
    int victim_slot = 0;
    unsigned long long int oldest_age = cache_bank[set_index][0].age;
    for (int i = 1; i < assoc_ways; ++i)
    {
        if (cache_bank[set_index][i].age < oldest_age)
        {
            oldest_age = cache_bank[set_index][i].age;
            victim_slot = i;
        }
    }
    return victim_slot;
}

static void refresh_slot(mem_addr_t set_index, int slot_index, mem_addr_t tag_id)
{
    cache_bank[set_index][slot_index].in_use = 1;
    cache_bank[set_index][slot_index].tag_id = tag_id;
    cache_bank[set_index][slot_index].age = ++lru_tick;
}

/*
 * initCache - 分配内存，将valid、tag和LRU写入0，
 * 同时计算set_index_mask
 */
void build_cache()
{
    if (s_bits < 0)
    {
        printf("set number error!\n");
        exit(0);
    }
    cache_bank = (cache_space_t)malloc(set_count * sizeof(set_t));
    if (cache_bank == NULL)
    {
        printf("No set memory!\n");
        exit(0);
    }
    for (int i = 0; i < set_count; ++i)
    {
        cache_bank[i] = (line_t *)malloc(assoc_ways * sizeof(line_t)); // 为行申请空间
        if (!cache_bank[i])
        {
            printf("No line memory!\n");
            exit(0);
        }
        for (int j = 0; j < assoc_ways; ++j)
        {
            cache_bank[i][j].age = 0;
            cache_bank[i][j].tag_id = 0;
            cache_bank[i][j].in_use = 0;
        }
    }
}

/*
 * 释放已分配的内存
 */
void destroy_cache()
{
    if (cache_bank == NULL)
    {
        return;
    }
    for (int i = 0; i < set_count; ++i)
    {
        free(cache_bank[i]);
    }
    free(cache_bank);
}

/*
 * 访问地址为 addr 的数据.
 * 如果该数据已在缓存中，增加 hit_count
 * 如果该数据不在缓存中，将其载入缓存，增加 miss_count.
 * 如果有一行被替换，增加 eviction_count.
 */
void access_cache(mem_addr_t addr)
{
    mem_addr_t tag_now = pick_tag(addr);
    current_set_index = pick_set_index(addr);
    int hit_slot = probe_hit_slot(current_set_index, tag_now);
    if (hit_slot >= 0)
    {
        hit_total++;
        cache_bank[current_set_index][hit_slot].age = ++lru_tick;
        return;
    }

    miss_total++;
    int free_slot = probe_free_slot(current_set_index);
    if (free_slot >= 0)
    {
        refresh_slot(current_set_index, free_slot, tag_now);
        return;
    }

    eviction_total++;
    int victim_slot = choose_victim_slot(current_set_index);
    refresh_slot(current_set_index, victim_slot, tag_now);
}

/*
 * 对给定的跟踪文件进行回放
 */
void process_trace(char *trace_fn)
{
    char trace_buf[1000];
    mem_addr_t address = 0;
    unsigned int size = 0;
    FILE *trace_fp = fopen(trace_fn, "r");
    if (!trace_fp)
    {
        fprintf(stderr, "%s: %s\n", trace_fn, strerror(errno));
        exit(1);
    }
    while (fgets(trace_buf, 1000, trace_fp) != NULL)
    {
        char opcode = '\0';
        if (sscanf(trace_buf, " %c %llx,%u", &opcode, &address, &size) == 3)
        {
            if (opcode == 'S' || opcode == 'L' || opcode == 'M')
            {
                if (verbose_mode)
                    printf("%c %llx,%u ", opcode, address, size);
                access_cache(address);
                /* 如果指令为读写，则再次访问 */
                if (opcode == 'M')
                    access_cache(address);
                if (verbose_mode)
                    printf("\n");
            }
        }
    }
    fclose(trace_fp);
}

/*
 * 打印使用信息
 */
void show_usage(char *argv[])
{
    printf("Usage: %s [-hv] -s <num> -E <num> -b <num> -t <file>\n", argv[0]);
    printf("Options:\n");
    printf("  -h         Print this help message.\n");
    printf("  -v         Optional verbose flag.\n");
    printf("  -s <num>   Number of set index bits.\n");
    printf("  -E <num>   Number of lines per set.\n");
    printf("  -b <num>   Number of block offset bits.\n");
    printf("  -t <file>  Trace file.\n");
    printf("\nExamples:\n");
    printf("  linux>  %s -s 4 -E 1 -b 4 -t traces/yi.trace\n", argv[0]);
    printf("  linux>  %s -v -s 8 -E 2 -b 4 -t traces/yi.trace\n", argv[0]);
    exit(0);
}

int main(int argc, char *argv[])
{
    char c;
    while ((c = getopt(argc, argv, "s:E:b:t:vh")) != -1)
    {
        switch (c)
        {
        case 's':
            s_bits = atoi(optarg);
            break;
        case 'E':
            assoc_ways = atoi(optarg);
            break;
        case 'b':
            b_bits = atoi(optarg);
            break;
        case 't':
            trace_path = optarg;
            break;
        case 'v':
            verbose_mode = 1;
            break;
        case 'h':
            show_usage(argv);
            exit(0);
        default:
            show_usage(argv);
            exit(1);
        }
    }
    /* 所有必需的命令行参数都已指定 */
    if (s_bits == 0 || assoc_ways == 0 || b_bits == 0 || trace_path == NULL)
    {
        printf("%s: Missing required command line argument\n", argv[0]);
        show_usage(argv);
        exit(1);
    }
    /* 从命令行参数计算缓存规模 */
    set_count = 1 << s_bits;
    /* 初始化缓存 */
    build_cache();
    process_trace(trace_path);
    /* 释放分配的内存 */
    destroy_cache();
    /* 输出自动测试程序的命中和未命中统计信息 */
    printSummary(hit_total, miss_total, eviction_total);
    return 0;
}
//****************************End**********************#