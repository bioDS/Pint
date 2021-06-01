#include "robin_hood.h"
#include <glib-2.0/glib.h>
#include <omp.h>
#include <stdlib.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "flat_hash_map.hpp"
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

Active_Set active_set_new(int max_length, int p)
{
    Active_Set as;
    as.length = 0;
    as.max_length = max_length;
    as.permutation = NULL;
    as.p = p;
    return as;
}

void active_set_free(Active_Set as)
{
    for (auto c = as.entries1.begin(); c != as.entries1.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    for (auto c = as.entries2.begin(); c != as.entries2.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    for (auto c = as.entries3.begin(); c != as.entries3.end(); c++) {
        struct AS_Entry e = c->second;
        if (NULL != e.col.compressed_indices) {
            free(e.col.compressed_indices);
        }
    }
    as.entries1.clear();
    as.entries2.clear();
    as.entries3.clear();
    if (NULL != as.permutation)
        gsl_permutation_free(as.permutation);
}

bool active_set_present(Active_Set* as, long value)
{
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    int p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }

    return (entries->contains(value) && entries->at(value).present);
}

void active_set_append(Active_Set* as, long value, int* col, int len)
{
    //if (value == pair_to_val(std::make_tuple(interesting_col, interesting_col), 100)) {
    //  printf("appending interesting col %d to as\n", value);
    //}
    // printf("as, adding val %ld as ", value);
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    int p = as->p;
    if (p % 5 != 0) {
        // printf("\np = %d\n", p);
        g_assert_true(p % 5 == 0);
    }
    if (value < p) {
        // printf("[%ld < %d]: main\n", value, p);
        entries = &as->entries1;
    } else if (value < p * p) {
        // printf("[%ld < %d]: pair\n", value, p*p);
        entries = &as->entries2;
    } else {
        // printf("triple\n");
        entries = &as->entries3;
    }
    if (entries->contains(value)) {
        struct AS_Entry e = entries->at(value);
        if (e.present)
            return;
        if (e.was_present) {
            e.present = TRUE;
        }
        entries->insert_or_assign(value, e);
    } else {
        struct AS_Entry e;
        e.val = value;
        e.present = TRUE;
        e.was_present = TRUE;
        e.col = col_to_s8b_col(len, col);
        int i = as->length;
        entries->insert_or_assign(value, e);
    }
    as->length++;
}

void active_set_remove(Active_Set* as, long value)
{
    robin_hood::unordered_flat_map<long, AS_Entry>* entries;
    int p = as->p;
    if (value < p) {
        entries = &as->entries1;
    } else if (value < p * p) {
        entries = &as->entries2;
    } else {
        entries = &as->entries3;
    }
    entries->at(value).present = FALSE; //TODO does this work?
    as->length--;
}

//int active_set_get_index(Active_Set* as, int index)
//{
//    struct AS_Entry* e = &as->entries[index];
//    if (e->present) {
//        return e->val;
//    } else {
//        return -INT_MAX;
//    }
//}

char update_working_set_cpu(
    struct XMatrixSparse Xc, struct row_set relevant_row_set, Thread_Cache* thread_caches, Active_Set* as,
    struct X_uncompressed Xu, float* rowsum, bool* wont_update, int p, int n,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, int* updateable_items, int count_may_update,
    float* last_max)
{
    int* host_X = Xu.host_X;
    int* host_col_nz = Xu.host_col_nz;
    int* host_col_offsets = Xu.host_col_offsets;
    char increased_set = FALSE;
    long length_increase = 0;
    int total = 0, skipped = 0;
    int p_int = p * (p + 1) / 2;

    long total_inter_cols = 0;
    int correct_k = 0;
#pragma omp parallel for reduction(+ \
                                   : total_inter_cols, total, skipped) schedule(static)
    for (long main_i = 0; main_i < count_may_update; main_i++) {
        // use Xc to read main effect
        Thread_Cache thread_cache = thread_caches[omp_get_thread_num()];
        int* col_i_cache = thread_cache.col_i;
        int* col_j_cache = thread_cache.col_j;
        long main = updateable_items[main_i];
        float max_inter_val = 0;
        int inter_cols = 0;
        // ska::flat_hash_set<long> inters_found;
        robin_hood::unordered_flat_map<long, float> sum_with_col;
        // robin_hood::unordered_flat_map<long, float> sum_with_col = thread_cache.lf_map;
        //robin_hood::unordered_flat_map<std::pair<long, long>, float> sum_with_col2;
        int main_col_len = 0;

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

                    int row_main = entry;
                    float rowsum_diff = rowsum[row_main];

                    // Do thing per entry here:

                    // Iterate through matrix of remaining pairs, checking three-way interactions.
                    // printf("checking main: %ld\n", main); //TOOD: maintain separate lists so we can solve them in order
                    sum_with_col[main] += rowsum_diff;
                    for (int ri = 0; ri < relevant_row_set.row_lengths[row_main]; ri++) {
                        int inter = relevant_row_set.rows[row_main][ri];
                        if (inter > main) {
                            // printf("checking pairwise %ld,%d\n", main, inter); //TOOD: maintain separate lists so we can solve them in order
                            sum_with_col[inter] += rowsum_diff;
                            for (int ri2 = ri + 1; ri2 < relevant_row_set.row_lengths[row_main]; ri2++) {
                                int inter2 = relevant_row_set.rows[row_main][ri2];
                                long inter_ind = pair_to_val(std::make_tuple(inter, inter2), p);
                                // printf("checking triple %ld,%d,%d: diff %f\n", main, inter, inter2, rowsum_diff);
                                if (row_main == 0 && inter == 1 && inter2 == 2) {
                                    // printf("interesting col ind == %ld", inter_ind);
                                }
                                sum_with_col[inter_ind] += rowsum_diff;
                                if (main == interesting_col && inter == interesting_col) {
                                    // printf("appending %f to interesting col (%d,%d)\n", rowsum_diff, main, inter);
                                }
                            }
                        }
                    }

                    column_entries[col_entry_pos] = entry;
                    col_entry_pos++;
                }
                values >>= item_width[word.selector];
            }
        }
        main_col_len = col_entry_pos;

        if (VERBOSE && main == interesting_col) {
            printf("interesting column sum %ld: %f\n", main, sum_with_col[main]);
        }
        inter_cols = sum_with_col.size();
        total_inter_cols += inter_cols;
        auto curr_inter = sum_with_col.cbegin();
        auto last_inter = sum_with_col.cend();
        while (curr_inter != last_inter) {
            long tuple_val = curr_inter->first;
            float sum = std::abs(curr_inter->second);
            if (tuple_val == main) {
                // printf("%ld,sum: %f > %f (lambda)?\n", main, sum, lambda);
            } else if (tuple_val < p) {
                // printf("%ld,%ld, sum: %f > %f (lambda)?\n", main, tuple_val, sum, lambda);
            } else {
                g_assert_true(tuple_val < p * p);
                // std::tuple<long,long> inter_pair_tmp = val_to_pair(tuple_val, p);
                // printf("%ld,%d,%d sum: %f > %f (lambda)?\n", main, std::get<0>(inter_pair_tmp), std::get<1>(inter_pair_tmp), sum, lambda);
            }
            max_inter_val = std::max(max_inter_val, sum);
            // printf("testing inter %d, sum is %d\n", inter, sum_with_col[inter]);
            if (sum > lambda) {
                int a, b, c;
                long k;
                std::tuple<long, long> inter_pair = val_to_pair(tuple_val, p);
                if (tuple_val == main) {
                    a = main;
                    b = main; //TODO: unnecessary
                    c = main;
                    // k = pair_to_val(std::make_tuple(a, b), p);
                    k = main;
                } else if (tuple_val < p) {
                    a = main;
                    b = tuple_val;
                    c = main; //TODO: unnecessary
                    k = pair_to_val(std::make_tuple(a, b), p);
                    if (k < p) {
                        printf("(%d,%d|%d): k = %ld\n", a, b, p, k);
                    }
                    g_assert_true(k >= p || k < p * p);
                } else {
                    g_assert_true(tuple_val <= p * p);
                    a = main;
                    b = std::get<0>(inter_pair);
                    c = std::get<1>(inter_pair);
                    k = triplet_to_val(std::make_tuple(a, b, c), p);
                }
                long inter = std::get<0>(inter_pair);

                int* colA = &Xu.host_X[Xu.host_col_offsets[a]];
                int* colB = &Xu.host_X[Xu.host_col_offsets[b]];
                int* colC = &Xu.host_X[Xu.host_col_offsets[c]];
                int ib = 0, ic = 0;
                long inter_len = 0;
                for (int ia = 0; ia < Xu.host_col_nz[a]; ia++) {
                    int cur_row = colA[ia];
                    //if (a == b && a == c) {
                    //  printf("%d: %d ", ia, cur_row);
                    //}
                    while (colB[ib] < cur_row && ib < Xu.host_col_nz[b] - 1)
                        ib++;
                    while (colC[ic] < cur_row && ic < Xu.host_col_nz[c] - 1)
                        ic++;
                    if (cur_row == colB[ib] && cur_row == colC[ic]) {
                        //if (a == b && a == c) {
                        //  printf("\n%d,%d,%d\n", ia, ib, ic);
                        //}
                        col_j_cache[inter_len] = cur_row;
                        inter_len++;
                    }
                }
                // if (main == interesting_col) {
                // printf("appending interesting col %d (%ld)\n", k, main);
                // }
                active_set_append(as, k, col_j_cache, inter_len);
                increased_set = TRUE;
                total++;
            }
            curr_inter++;
        }
        if (main == interesting_col)
            printf("largest inter found for effect %ld was %f\n", main, max_inter_val);
        last_max[main] = max_inter_val;
        sum_with_col.clear();
    }

    // printf("total: %d, skipped %d, inter_cols %d\n", total, skipped, total_inter_cols);
    return increased_set;
}

char update_working_set(
    struct X_uncompressed Xu, XMatrixSparse Xc, float* rowsum, bool* wont_update, int p, int n,
    float lambda, robin_hood::unordered_flat_map<long, float>* beta, int* updateable_items, int count_may_update, Active_Set* as,
    Thread_Cache* thread_caches, struct OpenCL_Setup* setup, float* last_max)
{
    int p_int = p * (p + 1) / 2;

    // construct small Xc containing only the relevant columns.
    // in particular, we want the row index with no columns outside the updateable_items set.

    // printf("wont_update:\n");
    // for (int i = 0; i < p; i++) {
    //   if (wont_update[i])
    //     printf("%d ", i);
    // }
    // printf("\n");
    struct row_set new_row_set = row_list_without_columns(Xc, Xu, wont_update, thread_caches);
    // quick test:
    //for (int row = 0; row < n; row++) {
    //  for (long inter_i = 0; inter_i < Xu.host_row_nz[row]; inter_i++) {
    //      long col = Xu.host_X_row[Xu.host_row_offsets[row] + inter_i];
    //      if (new_row_set.rows[row][inter_i] != col) {
    //        printf("%d,%ld : %d != %ld\n", row, inter_i, new_row_set.rows[row][inter_i], col);
    //      }
    //      g_assert_true(new_row_set.rows[row][inter_i] == col);
    //  }
    //}
    char increased_set = update_working_set_cpu(Xc, new_row_set, thread_caches, as, Xu, rowsum, wont_update, p, n, lambda, beta, updateable_items, count_may_update, last_max);
    for (int i = 0; i < n; i++) {
        free(new_row_set.rows[i]);
    }
    free(new_row_set.rows);
    free(new_row_set.row_lengths);

    return increased_set;
}

//struct OpenCL_Setup setup_working_set_kernel(
//    struct X_uncompressed Xu, int n, int p)
//{
//    host_append_sets = new ska::flat_hash_set<long>[omp_get_max_threads()];
//    int *host_X = Xu.host_X;
//    int *host_col_nz = Xu.host_col_nz;
//    int *host_col_offsets = Xu.host_col_offsets;
//    // Create a program from the kernel source file
//    // Creates the program
//    cl_int ret = 0;
//    int p_int = p * (p + 1) / 2;
//    //TODO: don't rely on file location like this.
//    struct CL_Source source = read_file("../src/update_working_set_kernel.cl");
//
//    cl_platform_id platform_id = NULL;
//    cl_device_id device_id = NULL;
//    cl_uint ret_num_devices;
//    cl_uint ret_num_platforms;
//    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to get OpenCL platform, err %d\n", ret);
//    }
//    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
//        &ret_num_devices);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to get OpenCL device, err %d\n", ret);
//    }
//    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create OpenCL context, err %d\n", ret);
//    }
//    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
//
//    cl_program program = clCreateProgramWithSource(context, 1, &source.buffer, &source.len, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to build program, err %d\n", ret);
//    }
//
//    //const char *cl_build_options = "-cl-opt-disable";
//    const char* cl_build_options = "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable";
//    // Build the program
//    ret = clBuildProgram(program, 1, &device_id, cl_build_options, NULL, NULL);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to build program, err %d\n", ret);
//    }
//
//    // Create the OpenCL kernel
//    cl_kernel kernel = clCreateKernel(program, "update_working_set", &ret);
//
//    // setup target memory
//    cl_mem target_X = clCreateBuffer(context, CL_MEM_READ_ONLY,
//        sizeof(int) * n * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_col_nz = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_col_offsets = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_wont_update = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_rowsum = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * n, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_beta = clCreateBuffer(context, CL_MEM_READ_ONLY,
//        sizeof(float) * p_int, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device beta buffer, %d\n", ret);
//    }
//    cl_mem target_updateable_items = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_append = clCreateBuffer(context, CL_MEM_READ_WRITE,
//        sizeof(char) * p_int, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//    cl_mem target_last_max = clCreateBuffer(context, CL_MEM_READ_WRITE,
//        sizeof(float) * p, NULL, &ret);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to create device buffer, %d\n", ret);
//    }
//
//    // zero everything in target memory
//    int fill = 0;
//    clEnqueueFillBuffer(command_queue, target_append, &fill, sizeof(char), 0, sizeof(char) * p_int, 0, NULL, NULL);
//    clEnqueueFillBuffer(command_queue, target_last_max, &fill, sizeof(int), 0, sizeof(float) * p, 0, NULL, NULL);
//
//    ret = clEnqueueWriteBuffer(command_queue, target_X, CL_TRUE, 0,
//        sizeof(int) * Xu.total_size, host_X, 0, NULL, NULL);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to write device buffers, %d\n", ret);
//    }
//    ret = clEnqueueWriteBuffer(command_queue, target_col_nz, CL_TRUE, 0,
//        sizeof(int) * p, host_col_nz, 0, NULL, NULL);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to write device buffers, %d\n", ret);
//    }
//    ret = clEnqueueWriteBuffer(command_queue, target_col_offsets, CL_TRUE, 0,
//        sizeof(int) * p, host_col_offsets, 0, NULL, NULL);
//    if (ret != CL_SUCCESS) {
//        fprintf(stderr, "failed to write device buffers, %d\n", ret);
//    }
//
//    struct OpenCL_Setup setup;
//    setup.context = context;
//    setup.kernel = kernel;
//    setup.command_queue = command_queue;
//    setup.program = program;
//    setup.target_X = target_X;
//    setup.target_col_nz = target_col_nz;
//    setup.target_col_offsets = target_col_offsets;
//    setup.target_wont_update = target_wont_update;
//    setup.target_rowsum = target_rowsum;
//    setup.target_beta = target_beta;
//    setup.target_updateable_items = target_updateable_items;
//    setup.target_append = target_append;
//    setup.target_last_max = target_last_max;
//    return setup;
//}

//void opencl_cleanup(struct OpenCL_Setup setup)
//{
//    // clean up
//    cl_int ret = 0;
//    ret = clFlush(setup.command_queue);
//    ret = clFinish(setup.command_queue);
//    ret = clReleaseKernel(setup.kernel);
//    ret = clReleaseProgram(setup.program);
//    ret = clReleaseMemObject(setup.target_X);
//    ret = clReleaseMemObject(setup.target_col_nz);
//    ret = clReleaseMemObject(setup.target_col_offsets);
//    ret = clReleaseMemObject(setup.target_wont_update);
//    ret = clReleaseMemObject(setup.target_rowsum);
//    ret = clReleaseMemObject(setup.target_beta);
//    ret = clReleaseMemObject(setup.target_updateable_items);
//    ret = clReleaseMemObject(setup.target_append);
//    ret = clReleaseCommandQueue(setup.command_queue);
//    ret = clReleaseContext(setup.context);
//}

static struct timespec start, end;
static float gpu_time = 0.0;