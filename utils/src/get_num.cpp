#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(long argc, char** argv)
{
    if (argc != 3) {
        printf("usage ./get_num [num] p\n");
        return 1;
    }

    long num = atoi(argv[1]);
    long p = atoi(argv[2]);

    printf("using num: %ld, p %ld\n", num, p);

    long offset = 0;
    for (long i = 0; i < p; i++) {
        for (long j = i; j < p; j++) {
            if (offset == num) {
                printf("i: %ld, j: %ld\n", i + 1, j + 1);
            }
            offset++;
        }
    }
    printf("out of %ld\n", offset + 1);
    return 0;
}
