#include <stdio.h>  
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <dlfcn.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>

double (*latency)(int);
int *ninst;

int foo(int N) {
    int checksum = 0;
    int i = 0;
    while(i<N) {
        asm( "addl %2, %0;" : "=r" (checksum) : "r" (checksum) , "i" (1) );
        //asm( "subl %2, %0;" : "=r" (i) : "r" (i) , "i" (13) );
        //asm( "subl %2, %0;" : "=r" (i) : "r" (i) , "i" (10) );
        i++;
    }
    return checksum;
}

void benchmark(int N, int (*f)(int)) {
    struct timeval start, end;
    int result;
    // run benchmark
    gettimeofday(&start, NULL);
    result = (*f)(N);
    gettimeofday(&end, NULL);

    double benchtime = (double)(end.tv_sec + end.tv_usec/1000000.0) -
                       (double)(start.tv_sec + start.tv_usec/1000000.0);
    printf("%d (result) %.10e (seconds)\n", result, benchtime);
}

int main(int argc, char** argv) {
    for(int i=0; i<100; i++)
        benchmark(atoi(argv[1]), &foo);
}
