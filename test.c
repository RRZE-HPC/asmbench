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
    int i = 0;
    while(i<N) {
        asm( "addl %2, %0;" : "=r" (i) : "r" (i) , "i" (23) );
        asm( "subl %2, %0;" : "=r" (i) : "r" (i) , "i" (13) );
        asm( "subl %2, %0;" : "=r" (i) : "r" (i) , "i" (10) );
        i++;
    }
    return i;
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
    printf("%.10e (seconds)\n", benchtime);
    printf("%d (result)\n", result);
}

int main(int argc, char** argv) {
    benchmark(atoi(argv[1]), &foo);
}