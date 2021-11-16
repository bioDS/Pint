#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int_fast64_t argc, char** argv)
{
    if (argc != 3) {
        printf("usage ./get_num [num] p\n");
        return 1;
    }

    int_fast64_t num = atoi(argv[1]);
    int_fast64_t p = atoi(argv[2]);

    printf("using num: %ld, p %ld\n", num, p);

    int_fast64_t offset = 0;
    for (int_fast64_t i = 0; i < p; i++) {
        for (int_fast64_t j = i; j < p; j++) {
            if (offset == num) {
                printf("i: %ld, j: %ld\n", i + 1, j + 1);
            }
            offset++;
        }
    }
    printf("out of %ld\n", offset + 1);
    return 0;
}
