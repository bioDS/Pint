#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "liblasso.h"

#define TRUE 1
#define FALSE 0


struct CL_Source {
    char* buffer;
    size_t len;
};

#define MAX_FILE_SIZE 1e6
/// Reads the entire file *filename into a new buffer and returns it.
struct CL_Source read_file(char* filename)
{
    char* big_buf = malloc(MAX_FILE_SIZE);
    char* line_buf;
    char* actual_buf;

    FILE* fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr,
            "error reading file %s, the program will probably now crash\n",
            filename);
    }

    // read entire file
    size_t pos = 0;
    size_t line_size = 0;
    int bytes_read = 0;
    while ((bytes_read = getline(&line_buf, &line_size, fp)) > 0) {
        memcpy(&big_buf[pos], line_buf, bytes_read);
        pos += bytes_read;
    }
    actual_buf = malloc(pos);
    memcpy(actual_buf, big_buf, pos);

    free(line_buf);
    free(big_buf);
    struct CL_Source src = { actual_buf, pos };
    return src;
}

Active_Set active_set_new(int max_length)
{
    struct AS_Entry* entries = malloc(sizeof *entries * max_length);
    // N.B this is what sets was_present to false for every entry
    memset(entries, 0,
        sizeof *entries * max_length); // not strictly necessary, but probably safer.
    int length = 0;
    Active_Set as = { entries, length, max_length, NULL };
    return as;
}

void active_set_free(Active_Set as)
{
    for (int i = 0; i < as.length; i++) {
        struct AS_Entry* e = &as.entries[i];
        if (NULL != e->col.compressed_indices) {
            free(e->col.compressed_indices);
        }
    }
    free(as.entries);
    if (NULL != as.permutation)
        free(as.permutation);
}

void active_set_append(Active_Set* as, int value, int* col, int len)
{
    struct AS_Entry* e = &as->entries[value];
    if (e->was_present) {
        e->present = TRUE;
    } else {
        int i = as->length;
        e->val = value;
        e->present = TRUE;
        e->was_present = TRUE;
        e->col = col_to_s8b_col(len, col);
    }
}

void active_set_remove(Active_Set* as, int index)
{
    as->entries[index].present = FALSE;
}

int active_set_get_index(Active_Set* as, int index)
{
    struct AS_Entry* e = &as->entries[index];
    if (e->present) {
        return e->val;
    } else {
        return -INT_MAX;
    }
}


/*
 * TODO: don't use thread_caches to store the result, or directly add things to as.
  * We probably need to have an allocated buffer for things that could be added to the
  * active set, and hope we don't have too many things.
 */
inline char update_working_set_cpu_old(
    // int* host_X, int* host_col_nz, int* host_col_offsets, int* host_append,
    struct XMatrixSparse Xc, char* host_append,
    float* rowsum, int* wont_update, int p, int n,
    float lambda, float* beta, int* updateable_items, int count_may_update,
    float *last_max, Active_Set *as, Thread_Cache *thread_caches) {
//char update_working_set_old(XMatrixSparse Xc, char* host_append, double *rowsum, int *wont_update,
//                        Active_Set *as, double *last_max, int p, int n,
//                        int_pair *precalc_get_num, double lambda, double *beta,
//                        Thread_Cache *thread_caches, XMatrixSparse X2c) {
    int p_int = p*(p+1)/2;
  memset(host_append, 0, p_int * sizeof(char));
  char increased_set = FALSE;
  long length_increase = 0;
#pragma omp parallel for reduction(& : increased_set) shared(last_max) schedule(static) reduction(+: length_increase)
  for (long main = 0; main < p; main++) {
    Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
    int *col_i_cache = thread_cache.col_i;
    int *col_j_cache = thread_cache.col_j;
    int main_col_len = 0;
    if (!wont_update[main]) {
      {
        int *column_entries = col_i_cache;
        long col_entry_pos = 0;
        long entry = -1;
        for (int r = 0; r < Xc.cols[main].nwords; r++) {
          S8bWord word = Xc.cols[main].compressed_indices[r];
          unsigned long values = word.values;
          for (int j = 0; j <= group_size[word.selector]; j++) {
            int diff = values & masks[word.selector];
            if (diff != 0) {
              entry += diff;
              column_entries[col_entry_pos] = entry;
              col_entry_pos++;
            }
            values >>= item_width[word.selector];
          }
        }
        main_col_len = col_entry_pos;
        // g_assert_true(main_col_len == Xc.cols[main].nz);
      }
      int read_loops = 0;
      for (long inter = main; inter < p; inter++) {
        // TODO: no need to re-read the main column when inter == main.
        // worked out by hand as being equivalent to the offset we would have
        // reached. sumb is the amount we would have reached w/o the limit -
        // the amount that was actually covered by the limit.
        int k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) -
                 (main - 1) * (main - 1) - (main - 1)) /
                    2 +
                inter;
        float sumn = 0.0;
        int col_nz = 0;
        // g_assert_true(precalc_get_num[k].i == j);
        // g_assert_true(k <= (Xc.p * (Xc.p + 1) / 2));
        if (!wont_update[inter]) {

          // we've already calculated the interaction, re-use it.
          if (as->entries[k].was_present) {
            // printf("re-using column %d\n", k);
            S8bCol s8bCol = as->entries[k].col;
            col_nz = s8bCol.nz;
            int entry = -1;
            int tmpCount = 0;
            for (int i = 0; i < s8bCol.nwords; i++) {
              S8bWord word = s8bCol.compressed_indices[i];
              unsigned long values = word.values;
              for (int j = 0; j <= group_size[word.selector]; j++) {
                tmpCount++;
                int diff = values & masks[word.selector];
                if (diff != 0) {
                  entry += diff;
                  if (entry > Xc.n) {
                    printf("entry: %d\n", entry);
                    printf("col %d col_nz: %d, tmpCount: %d\n", k, col_nz,
                           tmpCount);
                  }
                //   g_assert_true(entry < Xc.n);
                  sumn += rowsum[entry];
                }
                values >>= item_width[word.selector];
              }
            }
          } else {
            // printf("calculating new column\n");
            // this column has never been in the working set before, therefore
            // its beta value is zero and so is sumn.
            // calculate the interaction
            // and maybe store it read columns i and j simultaneously
            // int entry_i = -1;
            int i_pos = 0;
            int entry_j = -1;
            int pos = 0;
            // sum of rowsums for this column
            // int i_w = 0;
            int j_w = 0;
            // S8bWord word_i = Xc.compressed_indices[i][i_w];
            S8bWord word_j = Xc.cols[inter].compressed_indices[j_w];
            // int i_wpos = 0;
            int j_wpos = 0;
            // unsigned long values_i = word_i.values;
            unsigned long values_j = word_j.values;
            // int i_size = group_size[word_i.selector];
            int j_size = group_size[word_j.selector];
            // This whole loop is a bit awkward, but what can you do.
            // printf("interaction between %d (len %d) and %d (len %d)\n", i,
            //  Xc.col_nz[i], j, Xc.col_nz[j]);
            int entry_i = -2;
            while (i_pos < main_col_len && j_w <= Xc.cols[inter].nwords) {
            read:
              if (entry_i == entry_j) {
                // update interaction and move to next entry of each word
                sumn += rowsum[entry_i];
                col_j_cache[pos] = entry_i;
                pos++;
                // g_assert_true(pos < Xc.n);
              }
              while (entry_i <= entry_j && i_pos < main_col_len) {
                entry_i = col_i_cache[i_pos];
                i_pos++;
                if (entry_i == entry_j) {
                  goto read;
                }
              }
              if (entry_j <= entry_i) {
                // read through j until we hit the end, or reach or exceed
                // i.
                while (j_w <= Xc.cols[inter].nwords) {
                  // current word
                  while (j_wpos <= j_size) {
                    int diff = values_j & masks[word_j.selector];
                    j_wpos++;
                    values_j >>= item_width[word_j.selector];
                    if (diff != 0) {
                      entry_j += diff;
                      // we've found the next value of j
                      // if it's equal we'll handle it earlier in the loop,
                      // otherwise go to the j read loop.
                      if (entry_j >= entry_i) {
                        goto read;
                        // break;
                      }
                    }
                  }
                  // switch to the next word
                  j_w++;
                  if (j_w < Xc.cols[inter].nwords) {
                    word_j = Xc.cols[inter].compressed_indices[j_w];
                    values_j = word_j.values;
                    j_size = group_size[word_j.selector];
                    j_wpos = 0;
                  }
                }
              }
            }
            col_nz = pos;
            // g_assert_true(pos == X2c.nz[k]);
            // N.B. putting everything we use in the active set, but marking it
            // as inactive. this speeds up future checks quite significantly, at
            // the cost of increased memory use. For a gpu implenentation we
            // probably don't want this.
            //active_set_append(as, k, col_j_cache, col_nz);
            //length_increase++;
            //active_set_remove(as, k);
          }
          // if (k == 85)
          //  printf("interaction contains %d cols, sumn: %f\n", col_nz, sumn);
          // either way, we now have sumn
          sumn = fabs(sumn);
          sumn += fabs(beta[k] * col_nz);
          if (main == interesting_col && inter == interesting_col) {
              printf("lambda * n / 2 = %f\n", lambda * n / 2);
              printf("sumn: %f\n", sumn);
              printf("last_max[%ld] = %f\n", main, last_max[main]);
          }
          if (sumn > last_max[main]) {
            last_max[main] = sumn;
            if (main == interesting_col && inter == interesting_col) {
                printf("updating last_max[%ld] to %f\n", main, last_max[main]);
            }
          }
          if (sumn > lambda * n / 2) {
            // active_set_append(as, k, col_j_cache, col_nz);
            // printf("appending %d***************\n", k);
            host_append[k] = TRUE;
            increased_set = TRUE;
          } else {
            active_set_remove(as, k);
          }
          // TODO: store column for re-use even if it's not really added to
          // the active set?
        } else {
          // since we don't update any of this columns interactions, they
          // shouldn't be in the working set
          active_set_remove(as, k);
        }
      }
    }
  }
  as->length += length_increase;
  return increased_set;
}

inline char update_working_set_device(
    struct X_uncompressed Xu, char* host_append,
    float* rowsum, int* wont_update, int p, int n,
    float lambda, float* beta, int* updateable_items, int count_may_update, 
    struct OpenCL_Setup* setup, float *host_last_max)
{
    int *host_X = Xu.host_X;
    int* host_col_nz = Xu.host_col_nz;
    int* host_col_offsets = Xu.host_col_offsets;
    cl_int ret = 0;
    int p_int = p * (p + 1) / 2;
    printf("p_int=%d\n", p_int);
    cl_kernel kernel = setup->kernel;
    cl_command_queue command_queue = setup->command_queue;

    ret = clEnqueueWriteBuffer(command_queue, setup->target_wont_update, CL_TRUE, 0,
        sizeof(int) * p, wont_update, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }
    ret = clEnqueueWriteBuffer(command_queue, setup->target_rowsum, CL_TRUE, 0,
        sizeof(int) * n, rowsum, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }
    ret = clEnqueueWriteBuffer(command_queue, setup->target_beta, CL_TRUE, 0,
        sizeof(float) * p_int, beta, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device beta buffer, %d\n", ret);
    }
    ret = clEnqueueWriteBuffer(command_queue, setup->target_updateable_items, CL_TRUE, 0,
        sizeof(int) * p, updateable_items, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }

    float device_beta_test = -1.0;
    clEnqueueReadBuffer(command_queue, setup->target_beta, CL_TRUE, 0,
        sizeof(float), &device_beta_test, 0, NULL, NULL);

    //host_append = (int*)calloc(p_int, sizeof(int));
    //memset(host_append, 0, p_int * sizeof(int));

    // clear the append list
    int fill = 0;
    clEnqueueFillBuffer(command_queue, setup->target_append, &fill, sizeof(int), 0, sizeof(int) * p_int, 0, NULL, NULL);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem),
        (void*)&setup->target_rowsum);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem),
        (void*)&setup->target_wont_update);
    ret = clSetKernelArg(kernel, 2, sizeof(int), (void*)&p);
    ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&n);
    ret = clSetKernelArg(kernel, 4, sizeof(float),
        (void*)&lambda);
    ret = clSetKernelArg(kernel, 5, sizeof(cl_mem),
        (void*)&setup->target_beta);
    ret = clSetKernelArg(kernel, 6, sizeof(cl_mem),
        (void*)&setup->target_append);
    ret = clSetKernelArg(kernel, 7, sizeof(cl_mem),
        (void*)&setup->target_col_nz);
    ret = clSetKernelArg(kernel, 8, sizeof(cl_mem),
        (void*)&setup->target_X);
    ret = clSetKernelArg(kernel, 9, sizeof(cl_mem),
        (void*)&setup->target_col_offsets);
    ret = clSetKernelArg(kernel, 10, sizeof(cl_mem),
        (void*)&setup->target_updateable_items);
    ret = clSetKernelArg(kernel, 11, sizeof(int),
        (void*)&count_may_update);
    ret = clSetKernelArg(kernel, 12, sizeof(cl_mem),
        (void*)&setup->target_last_max);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to set arguments, %d\n", ret);
    }
    // local_item_size = 128; //Null is generally pretty good.
    size_t global_item_size = p;
    //if (p % local_item_size > 0)
    //    global_item_size += (local_item_size - p % local_item_size);
    // int* tmp_append = (int*)malloc(p_int * sizeof(int));
    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    ret = clEnqueueNDRangeKernel(command_queue, setup->kernel, 1,
        // NULL, &global_item_size, &local_item_size, 0,
        NULL, &global_item_size, NULL, 0,
        NULL, NULL);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "something went wrong %d\n", ret);
    }
    clEnqueueReadBuffer(command_queue, setup->target_append, CL_TRUE, 0,
        sizeof(*host_append) * p_int, host_append, 0, NULL, NULL);
    clEnqueueReadBuffer(command_queue, setup->target_last_max, CL_TRUE, 0,
        sizeof(float) * p, host_last_max, 0, NULL, NULL);
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // gpu_time = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);
}

char update_working_set(
    // int* host_X, int* host_col_nz, int* host_col_offsets, int* host_append,
    struct X_uncompressed Xu, XMatrixSparse Xc, char* host_append,
    float* rowsum, int* wont_update, int p, int n,
    float lambda, float* beta, int* updateable_items, int count_may_update, Active_Set* as,
    Thread_Cache *thread_caches, struct OpenCL_Setup *setup, float* last_max)
{
    int p_int = p * (p + 1) / 2;
    // char increased_set = TRUE;
    char increased_set = update_working_set_cpu(Xu, host_append, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, last_max);
    // char increased_set = update_working_set_cpu_old(Xc, host_append, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, last_max, as, thread_caches);
    // return increased_set;
    // update_working_set_device(Xu, host_append, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, setup, last_max);
    printf("re-calculating working set columns\n");
    struct AS_Entry *entries = as->entries;
    long length_increase = 0;
#pragma omp parallel for reduction(+ \
                                   : length_increase)
    for (int main = 0; main < p; main++) {
        int main_col_len;
        int* col_i_cache = thread_caches[omp_get_thread_num()].col_i;
        int* col_j_cache = thread_caches[omp_get_thread_num()].col_j;
        {
            int* column_entries = col_i_cache;
            long col_entry_pos = 0;
            long entry = -1;
            for (int r = 0; r < Xc.cols[main].nwords; r++) {
                S8bWord word = Xc.cols[main].compressed_indices[r];
                unsigned long values = word.values;
                for (int j = 0; j <= group_size[word.selector]; j++) {
                    int diff = values & masks[word.selector];
                    if (diff != 0) {
                        entry += diff;
                        column_entries[col_entry_pos] = entry;
                        col_entry_pos++;
                    }
                    values >>= item_width[word.selector];
                }
            }
            main_col_len = col_entry_pos;
        }
        for (int inter = main; inter < p; inter++) {
            int k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
            if (host_append[k]) {
            // if (TRUE) {
                if (!entries[k].was_present) {
                    int col_nz = 0;
                    int i_pos = 0;
                    int entry_j = -1;
                    int pos = 0;
                    // sum of rowsums for this column
                    int j_w = 0;
                    S8bWord word_j = Xc.cols[inter].compressed_indices[j_w];
                    int j_wpos = 0;
                    unsigned long values_j = word_j.values;
                    int j_size = group_size[word_j.selector];
                    // This whole loop is a bit awkward, but what can you do.
                    int entry_i = -2;
                    while (i_pos < main_col_len && j_w <= Xc.cols[inter].nwords) {
                    read2:
                        if (entry_i == entry_j) {
                            // update interaction and move to next entry of each word
                            col_j_cache[pos] = entry_i;
                            pos++;
                        }
                        while (entry_i <= entry_j && i_pos < main_col_len) {
                            entry_i = col_i_cache[i_pos];
                            i_pos++;
                            if (entry_i == entry_j) {
                                goto read2;
                            }
                        }
                        if (entry_j <= entry_i) {
                            // read through j until we hit the end, or reach or exceed
                            // i.
                            while (j_w <= Xc.cols[inter].nwords) {
                                // current word
                                while (j_wpos <= j_size) {
                                    int diff = values_j & masks[word_j.selector];
                                    j_wpos++;
                                    values_j >>= item_width[word_j.selector];
                                    if (diff != 0) {
                                        entry_j += diff;
                                        // we've found the next value of j
                                        // if it's equal we'll handle it earlier in the loop,
                                        // otherwise go to the j read loop.
                                        if (entry_j >= entry_i) {
                                            goto read2;
                                            // break;
                                        }
                                    }
                                }
                                // switch to the next word
                                j_w++;
                                if (j_w < Xc.cols[inter].nwords) {
                                    word_j = Xc.cols[inter].compressed_indices[j_w];
                                    values_j = word_j.values;
                                    j_size = group_size[word_j.selector];
                                    j_wpos = 0;
                                }
                            }
                        }
                    }
                    col_nz = pos;
                    // g_assert_true(k < p_int);
                    // printf("appending %d\n", k);
                    active_set_append(as, k, col_j_cache, col_nz);
                } else {
                    active_set_append(as, k, NULL, 0);
                }
                if (!as->entries[k].was_present) {
                    length_increase++;
                }
            //} else if (remove[k]) {
            //    active_set_remove(as, k);
            }
        }
    }
    return increased_set;
}

char update_working_set_cpu(
    // int* host_X, int* host_col_nz, int* host_col_offsets, int* host_append,
    struct X_uncompressed Xu, char* host_append,
    float* rowsum, int* wont_update, int p, int n,
    float lambda, float* beta, int* updateable_items, int count_may_update,
    float *last_max)
{
    int* host_X = Xu.host_X;
    int* host_col_nz = Xu.host_col_nz;
    int* host_col_offsets = Xu.host_col_offsets;
    char increased_set = FALSE;
    long length_increase = 0;
    int total = 0, skipped = 0;
    int p_int = p * (p + 1) / 2;
    char* append = host_append;
    // int* remove = (int*)calloc(p_int, sizeof(int));
    memset(append, 0, p_int * sizeof(char));
    // memset(remove, 0, p_int * sizeof *remove);
    // char *done = calloc(p_int, sizeof *done);

    int correct_k = 0;
#pragma omp parallel for
    for (long main_i = 0; main_i < count_may_update; main_i++) {
        long main = updateable_items[main_i];
        int inter_cols = 0;
        //TODO: really we want hash sets here, this is quite a waste.
        float *sum_with_col = calloc(p, sizeof *sum_with_col);
        long *inters_found = calloc(p, sizeof *inters_found);
        char *found_inter = calloc(p_int, sizeof *found_inter);
        // iterate through rows with an entry in main, check inverted list for interactions in this row.
        for (long row_main_i = 0; row_main_i < Xu.host_col_nz[main]; row_main_i++) {
            long row_main = host_X[host_col_offsets[main] + row_main_i];
            // check inverted list for interactions along row_main
            for (long inter_i = 0; inter_i < Xu.host_row_nz[row_main]; inter_i++) {
                long inter = Xu.host_X_row[Xu.host_row_offsets[row_main] + inter_i];
                int k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
                sum_with_col[inter] += rowsum[row_main];
                if (!found_inter[k]) {
                    inters_found[inter_cols] = inter;
                    inter_cols++;
                    found_inter[k] = TRUE;
                }
            }
        }
        for (int i = 0; i < inter_cols; i++) {
            long inter = inters_found[i];
            if (sum_with_col[inter] > 0 && sum_with_col[inter] > lambda * n / 2) {
                int k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) - (main - 1) * (main - 1) - (main - 1)) / 2 + inter;
                append[k] = TRUE;
                increased_set = TRUE;
                total++;
            }
        }
        free(sum_with_col);
        free(inters_found);
        free(found_inter);
    }
    printf("total: %d, skipped %d\n", total, skipped);
    // free(remove);
    return increased_set;
}


struct OpenCL_Setup setup_working_set_kernel(
    struct X_uncompressed Xu, int n, int p)
{
    int *host_X = Xu.host_X;
    int *host_col_nz = Xu.host_col_nz;
    int *host_col_offsets = Xu.host_col_offsets;
    // Create a program from the kernel source file
    // Creates the program
    cl_int ret = 0;
    int p_int = p * (p + 1) / 2;
    //TODO: don't rely on file location like this.
    struct CL_Source source = read_file("../src/update_working_set_kernel.cl");

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to get OpenCL platform, err %d\n", ret);
    }
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
        &ret_num_devices);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to get OpenCL device, err %d\n", ret);
    }
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create OpenCL context, err %d\n", ret);
    }
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_program program = clCreateProgramWithSource(context, 1, &source.buffer, &source.len, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to build program, err %d\n", ret);
    }

    //const char *cl_build_options = "-cl-opt-disable";
    const char* cl_build_options = "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable";
    // Build the program
    ret = clBuildProgram(program, 1, &device_id, cl_build_options, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to build program, err %d\n", ret);
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "update_working_set", &ret);

    // setup target memory
    cl_mem target_X = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(int) * n * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_col_nz = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_col_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_wont_update = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_rowsum = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_beta = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * p_int, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device beta buffer, %d\n", ret);
    }
    cl_mem target_updateable_items = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_append = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(char) * p_int, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }
    cl_mem target_last_max = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * p, NULL, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to create device buffer, %d\n", ret);
    }

    // zero everything in target memory
    int fill = 0;
    clEnqueueFillBuffer(command_queue, target_append, &fill, sizeof(char), 0, sizeof(char) * p_int, 0, NULL, NULL);
    clEnqueueFillBuffer(command_queue, target_last_max, &fill, sizeof(int), 0, sizeof(float) * p, 0, NULL, NULL);

    ret = clEnqueueWriteBuffer(command_queue, target_X, CL_TRUE, 0,
        sizeof(int) * Xu.total_size, host_X, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }
    ret = clEnqueueWriteBuffer(command_queue, target_col_nz, CL_TRUE, 0,
        sizeof(int) * p, host_col_nz, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }
    ret = clEnqueueWriteBuffer(command_queue, target_col_offsets, CL_TRUE, 0,
        sizeof(int) * p, host_col_offsets, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "failed to write device buffers, %d\n", ret);
    }

    struct OpenCL_Setup setup;
    setup.context = context;
    setup.kernel = kernel;
    setup.command_queue = command_queue;
    setup.program = program;
    setup.target_X = target_X;
    setup.target_col_nz = target_col_nz;
    setup.target_col_offsets = target_col_offsets;
    setup.target_wont_update = target_wont_update;
    setup.target_rowsum = target_rowsum;
    setup.target_beta = target_beta;
    setup.target_updateable_items = target_updateable_items;
    setup.target_append = target_append;
    setup.target_last_max = target_last_max;
    return setup;
}

void opencl_cleanup(struct OpenCL_Setup setup)
{
    // clean up
    cl_int ret = 0;
    ret = clFlush(setup.command_queue);
    ret = clFinish(setup.command_queue);
    ret = clReleaseKernel(setup.kernel);
    ret = clReleaseProgram(setup.program);
    ret = clReleaseMemObject(setup.target_X);
    ret = clReleaseMemObject(setup.target_col_nz);
    ret = clReleaseMemObject(setup.target_col_offsets);
    ret = clReleaseMemObject(setup.target_wont_update);
    ret = clReleaseMemObject(setup.target_rowsum);
    ret = clReleaseMemObject(setup.target_beta);
    ret = clReleaseMemObject(setup.target_updateable_items);
    ret = clReleaseMemObject(setup.target_append);
    ret = clReleaseCommandQueue(setup.command_queue);
    ret = clReleaseContext(setup.context);
}

static struct timespec start, end;
static float gpu_time = 0.0;


//int working_set_main()
//{
//
//    int* host_X;
//    int* host_col_nz;
//    int* host_col_offsets;
//    int* host_append;
//    // working set test;
//    int n = 1000;
//    int p = 10000;
//    printf("using p = %d\n", p);
//
//    int *col_offsets, *X, *col_nz;
//    int* wont_update;
//    float *rowsum, *beta;
//
//    int p_int = p * (p + 1) / 2;
//
//    X = (int*)calloc(n * p, sizeof(int));
//    col_nz = (int*)calloc(p, sizeof(int));
//    col_offsets = (int*)calloc(p, sizeof(int));
//    wont_update = (int*)calloc(p, sizeof(int));
//    rowsum = (float*)calloc(n, sizeof(float));
//    beta = (float*)calloc(p_int, sizeof(int));
//    memset(col_nz, 0, p * sizeof(int));
//    int offset = 0;
//    for (int i = 0; i < n; i++) {
//        rowsum[i] = 0;
//    }
//    for (int i = 0; i < p; i++) {
//        wont_update[i] = ((i + 173) % 4) == 0;
//    }
//    for (int i = 0; i < p_int; i++) {
//        beta[i] = 1.273 * (i % 7) + (i % 3);
//    }
//    for (int j = 0; j < p; j++) {
//        int size = (j + 273) % n;
//        col_nz[j] = size;
//        col_offsets[j] = offset;
//        for (int i = 0; i < size; i++) {
//            X[offset] = i * (n / size) % n;
//            rowsum[X[offset]] += beta[j];
//            offset++;
//        }
//    }
//    beta[0] = 1.2;
//
//    // create a list of only the columns that might update.
//    Queue* q = queue_new();
//    int count_may_update = 0;
//    for (int i = 0; i < p; i++) {
//        if (!wont_update[i]) {
//            count_may_update++;
//            queue_push_tail(q, (void*)(long)i);
//            //printf("%d may update\n", i);
//        }
//    }
//    printf("there were %d updateable items\n", count_may_update);
//    //int* updateable_items = malloc(sizeof *updateable_items * count_may_update);
//    int* updateable_items = malloc(sizeof *updateable_items * p);
//    for (int i = 0; i < count_may_update; i++) {
//        updateable_items[i] = (long)queue_pop_head(q);
//    }
//    queue_free(q);
//
//    host_X = X;
//    host_col_nz = col_nz;
//    host_col_offsets = col_offsets;
//
//    host_append = (int*)calloc(p_int, sizeof(int));
//    memset(host_append, 0, p_int * sizeof(int));
//
//    float lambda = 0.96;
//
//    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
//    update_working_set_cpu(host_X, host_col_nz, host_col_offsets, host_append,
//        rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update);
//    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
//    float cpu_time = ((float)(end.tv_nsec - start.tv_nsec)) / 1e9 + (end.tv_sec - start.tv_sec);
//
//    int total_appended = 0;
//    for (int i = 0; i < p_int; i++) {
//        if (host_append[i]) {
//            total_appended++;
//        }
//    }
//    printf("\n");
//    // TODO: record cpu/gpu running times.
//    printf("host appended   %d in total \n", total_appended);
//
//    struct OpenCL_Setup ocl_working_set = setup_working_set_kernel(host_X, host_col_nz, host_col_offsets, host_append, n, p);
//
//    int* tmp_append = (int*)calloc(p_int, sizeof(int));
//    update_working_set_device(host_X, host_col_nz, host_col_offsets, host_append,
//        rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, ocl_working_set, tmp_append);
//
//    total_appended = 0;
//    int total_diff = 0;
//    for (int i = 0; i < p_int; i++) {
//        if (host_append[i] && !tmp_append[i]) {
//           //printf("host appended %d but device didn't\n", i);
//           total_diff++;
//        } else if (!host_append[i] && tmp_append[i]) {
//           //printf("device appended %d but host didn't\n", i);
//           total_diff++;
//        }
//        if (tmp_append[i]) {
//            total_appended++;
//        }
//    }
//    printf("device appended %d in total \n", total_appended);
//    printf("total disagreements: %d\n", total_diff);
//
//    printf("cpu time: %.4f, gpu time %.4f\n", cpu_time, gpu_time);
//    printf("relative speedup: %.4f x\n", cpu_time / gpu_time);
//
//    free(host_append);
//    free(host_X);
//    free(host_col_nz);
//    free(host_col_offsets);
//    free(beta);
//
//    return 0;
//
//    // clean up
//    opencl_cleanup(ocl_working_set);
//}