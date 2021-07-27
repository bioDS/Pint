FILE* init_log(char* filename, long n, long p, long num_betas, char** job_args,
    long job_args_num);
void save_log(long iter, float lambda_value, long lambda_count, Beta_Value_Sets* beta_sets,
    FILE* log_file);
long check_can_restore_from_log(char* filename, long n, long p, long num_betas,
    char** job_args, long job_args_num);
void close_log(FILE* log_file);
FILE* restore_from_log(char* filename, bool check_args, long n, long p,
    char** job_args, long job_args_num, long* actual_iter,
    long* actual_lambda_count, float* actual_lambda_value,
    Beta_Value_Sets *actual_beta_sets);