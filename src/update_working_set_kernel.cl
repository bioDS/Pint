#define TRUE 1
#define FALSE 0

__kernel void vector_add(__global const int *A, __global const int *B,
                         __global int *C) {
  int i = get_global_id(0);
  if (i == 0)
    printf("running kernel\n");

  C[i] = A[i] + B[i];
}

__kernel void
update_working_set(__global float *rowsum, __global int *wont_update, int p,
                   int n, float lambda, __global ska::flat_hash_map<long, float> beta,
                   __global char *append, __global int *target_col_nz,
                   __global int *target_X, __global int *target_col_offsets,
                   __global int *updateable_items, int count_may_update, 
                   __global float *last_max) {
  char increased_set = FALSE;
  long length_increase = 0;
  int total = 0, skipped = 0;
  int p_int = p * (p + 1) / 2;

  int global_size = get_global_size(0);
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_size = get_local_size(0);
  int group_id = get_group_id(0);
  int num_groups = get_num_groups(0);

  //if (get_global_id(0) == 5) {
  //    printf("global_size = %d\n", global_size);
  //    printf("global_id = %d\n", global_id);
  //}
  // printf("test, global_id = %d\n", global_id);

  //if (global_id < 500) {
  //  printf("count_may_update = %d\n", count_may_update);
  //}

  //for (int i = global_id; i < p_int; i += global_size) {
  //  append[i] = 0;
  //  // remove[i] = 0;
  //}

  int iter_count = 0;
  //for (long main_i = global_id; main_i < count_may_update; main_i += global_size) {
  for (long main_i = group_id; main_i < count_may_update; main_i += num_groups) {
    long main = updateable_items[main_i];
    //for (long inter_i = main_i; inter_i < count_may_update; inter_i += 1) {
    for (long inter_i = main_i + local_id; inter_i < count_may_update; inter_i += group_size) {
      long inter = updateable_items[inter_i];
      int k = (2 * (p - 1) + 2 * (p - 1) * (main - 1) -
               (main - 1) * (main - 1) - (main - 1)) /
                  2 +
              inter;
      //printf("group %d, local %d (iter %d): checking %d (%ld,%ld)\n", group_id, local_id, iter_count, k, main, inter);
      iter_count++;
      float sumn = 0.0;
      int col_nz = 0;
      int j_pos = 0;
      for (int i_pos = 0; i_pos < target_col_nz[main]; i_pos++) {
        int entry_i = target_X[target_col_offsets[main] + i_pos];
        int entry_j = target_X[target_col_offsets[inter] + j_pos];
        if (entry_i == entry_j) {
          sumn += rowsum[entry_i];
          col_nz++;
        }
        // find correct position of j_pos;
        while (target_X[target_col_offsets[inter] + j_pos] < entry_i &&
               j_pos < target_col_nz[inter]) {
          j_pos++;
        }
      }
      sumn = fabs(sumn);
      sumn += fabs( beta[k] * col_nz);
      last_max[main] = fmax(last_max[main], sumn);
      last_max[inter] = fmax(last_max[inter], sumn);
      if (sumn > lambda * n / 2) {
        total++;
        append[k] = TRUE;
      } else {
        skipped++;
        // remove[k] = TRUE;
      }
    }
  }
  // printf("total: %d, skipped %d\n", total, skipped);
  // free(remove);
}