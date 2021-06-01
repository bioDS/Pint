FILE* init_log(char* filename, int n, int p, int num_betas, char** job_args,
    int job_args_num);
void save_log(int iter, float lambda_value, int lambda_count, robin_hood::unordered_flat_map<long, float> beta,
    int n_betas, FILE* log_file);
int check_can_restore_from_log(char* filename, int n, int p, int num_betas,
    char** job_args, int job_args_num);
void close_log(FILE* log_file);
FILE* restore_from_log(char* filename, int n, int p, int num_betas,
    char** job_args, int job_args_num, int* actual_iter,
    int* actual_lambda_count, float* actual_lambda_value,
    robin_hood::unordered_flat_map<long, float> actual_beta);