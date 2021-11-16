FILE* init_log(const char* filename, int_fast64_t n, int_fast64_t p, int_fast64_t num_betas, const char** job_args,
    int_fast64_t job_args_num);
void save_log(int_fast64_t iter, float lambda_value, int_fast64_t lambda_count, Beta_Value_Sets* beta_sets,
    FILE* log_file);
int_fast64_t check_can_restore_from_log(const char* filename, int_fast64_t n, int_fast64_t p, int_fast64_t num_betas,
    const char** job_args, int_fast64_t job_args_num);
void close_log(FILE* log_file);
FILE* restore_from_log(const char* filename, bool check_args, int_fast64_t n, int_fast64_t p,
    const char** job_args, int_fast64_t job_args_num, int_fast64_t* actual_iter,
    int_fast64_t* actual_lambda_count, float* actual_lambda_value,
    Beta_Value_Sets *actual_beta_sets);