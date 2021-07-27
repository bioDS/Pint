typedef struct S8bWord {
    unsigned long selector : 4;
    unsigned long values : 60;
} S8bWord;

typedef struct {
    S8bWord* compressed_indices;
    unsigned long nz;
    unsigned long nwords;
} S8bCol;

static long item_width[16] = { 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 10, 12, 15, 20, 30, 60 };
static long group_size[16] = { 240, 120, 60, 30, 20, 15, 12, 10,
    8, 7, 6, 5, 4, 3, 2, 1 };
static long masks[16] = { 0,
    0,
    (1 << 1) - 1,
    (1 << 2) - 1,
    (1 << 3) - 1,
    (1 << 4) - 1,
    (1 << 5) - 1,
    (1 << 6) - 1,
    (1 << 7) - 1,
    (1 << 8) - 1,
    (1 << 10) - 1,
    (1 << 12) - 1,
    (1 << 15) - 1,
    (1 << 20) - 1,
    (1 << 30) - 1,
    ((long)1 << 60) - 1 };

S8bWord to_s8b(long count, long* vals);
S8bCol col_to_s8b_col(long size, long* col);