FILE* init_log(char* filename, int n, int p, int num_betas, char** job_args,
    int job_args_num);
void save_log(int iter, float lambda_value, int lambda_count, Beta_Value_Sets* beta_sets,
    FILE* log_file);
int check_can_restore_from_log(char* filename, int n, int p, int num_betas,
    char** job_args, int job_args_num);
void close_log(FILE* log_file);
FILE* restore_from_log(char* filename, int n, int p,
    char** job_args, int job_args_num, int* actual_iter,
    int* actual_lambda_count, float* actual_lambda_value,
    Beta_Value_Sets *actual_beta_sets);