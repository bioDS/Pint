use std::fs::File;
use clap::{Arg, App, SubCommand};

//TODO: use bitpacking
//TODO: multithreading

struct XMatrix {
    rows: Vec<Vec<bool>>
}

struct XMatrix_Cols {
    cols: Vec<Vec<bool>>
}

struct Sparse_Xmatrix {
    n: usize,
    p: usize,
    column_indices: Vec<Vec<usize>>,
    column_nonzeros: Vec<usize>
}

trait matrix_functions {
    fn next(&self);
}

impl matrix_functions for Sparse_Xmatrix {
    fn next(&self) {
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_x2() {
        let X_row = read_x_csv("testX.csv");
        let X_col = row_to_col(X_row);
        let Y = read_y_csv("testY.csv");
        let X2 = x_to_x2_sparse_col(&X_col);

        let testX2 = read_x_csv("./testX2.csv");
        for col_ind in 0..X2.column_indices.len() {
            let col = &X2.column_indices[col_ind];
            for entry in col {
                if !testX2.rows[*entry][col_ind] {
                    println!("col {} row {} not present in testX2.csv", col_ind, entry);
                }
                assert!(testX2.rows[*entry][col_ind]);
            }
        }
    }
}

fn main() {
    let matches = App::new("cli lasso")
                    .version("0.1")
                    .arg(Arg::with_name("xmatrix")
                         .short("x")
                         .long("xmatrix")
                         .value_name("FILE")
                         .help(".csv file for X matrix")
                         .takes_value(true))
                    .arg(Arg::with_name("ymatrix")
                         .short("y")
                         .long("ymatrix")
                         .value_name("FILE")
                         .help(".csv file for Y matrix")
                         .takes_value(true))
                    .get_matches();

    let x_filename = matches.value_of("xmatrix").unwrap_or("testX.csv");
    let y_filename = matches.value_of("ymatrix").unwrap_or("testY.csv");

    let X_row = read_x_csv(x_filename);
    let X_col = row_to_col(X_row);
    let Y = read_y_csv(y_filename);

    let X2 = x_to_x2_sparse_col(&X_col);

    let beta = simple_coordinate_descent_lasso(&X2, Y);

    // find large looking beta values
    for b_ind in 0..beta.len() {
        let b = &beta[b_ind];
        if b.abs() > 500.0 {
            let (i1, i2) = match (get_num(b_ind, X_col.cols.len())) {
                Ok(x) => x,
                Err(e) => panic!(e),
            };
            println!("{} ({},{}): {}", b_ind, i1, i2, b);
        }
    }
}

fn x_to_x2_sparse_col(X: &XMatrix_Cols) -> Sparse_Xmatrix {
    let p = X.cols.len();
    let p_int = (X.cols.len()*(X.cols.len()+1))/2;
    let n = X.cols[0].len();
    println!("building X2. n = {}, p = {}", n, p);
    let mut column_indices: Vec<Vec<usize>> = vec![Vec::new(); p_int];
    let mut column_nonzeros = vec![0; p_int];
    let mut col_ind = 0;
    for col1_ind in 0..p {
        for col2_ind in col1_ind..p {
            let col1 = &X.cols[col1_ind];
            let col2 = &X.cols[col2_ind];
            //let size: usize = col.iter().map(|entry| if *entry==true {1} else {0}).sum();
            let mut size = 0;
            for k in 0..n {
                if col1[k] == true && col2[k] == true{
                    size += 1;
                    column_indices[col_ind].push(k);
                }
            }
            column_nonzeros[col_ind] = size;
            col_ind += 1;
        }
    }

    Sparse_Xmatrix { n, p: p_int, column_indices, column_nonzeros }
}

fn row_to_col(row_X: XMatrix) -> XMatrix_Cols {
    let rows = row_X.rows;
    let n = rows.len();
    let p = rows[0].len();

    let mut cols: Vec<Vec<bool>> = (0..p).map(|_x| vec![false; n]).collect();

    for row_ind in 0..n {
        for col_ind in 0..p {
            //println!("col: {}, row: {}", col_ind, row_ind);
            cols[col_ind][row_ind] = rows[row_ind][col_ind];
        }
    }

    XMatrix_Cols {cols}
}

fn print_x2_to_stdout(X2: &Vec<Vec<bool>>) {
    let mut row_count = 0;
    for row in X2 {
        print!("\"{}\"", row_count);
        for col in row {
            if *col == true {
                print!(",{}", 1);
            } else {
                print!(",{}", 0);
            }
        }
        row_count += 1;
        println!();
    }
}

fn X2_from_X(X: XMatrix) -> Vec<Vec<bool>> {
    let mut X2 = Vec::new();
    for row in &X.rows {
        let mut new_row: Vec<bool> = Vec::with_capacity(row.len()*row.len());
        for i in 0..row.len() {
            for j in i..row.len() {
                new_row.push(row[i] && row[j]);
            }
        }
        X2.push(new_row);
    }

    X2
}

// X should be column major
fn simple_coordinate_descent_lasso(X: &Sparse_Xmatrix, Y: Vec<f64>)  -> Vec<f64> {
    let p = X.p;
    let n = X.n;
    println!("p: {}, n: {}", p, n);
    let mut beta = vec![0.0; p];
    let mut rowsums = vec![0.0; n];
    let mut colsums = vec![0; p];
    let mut lambda = 100.0; //TODO: find max lambda;
    let halt_beta_diff = 1.0001;
    let mut error = calculate_error(&rowsums, &Y);
    println!("for testing purposes, initial e is {:.2}", error as f64);
    for lambda_seq in 0..50 {
        println!("lambda {}: {}", lambda_seq+1, lambda);
        for iter in 0..100 {
            let mut iter_max_change = 0.0;
            for k in 0..p {
                let col_k_max_change = update_beta_cyclic(X, &Y, &mut beta, n, p, k, &mut rowsums, lambda);
                if col_k_max_change > iter_max_change {
                    iter_max_change = col_k_max_change;
                }
            }
            let prev_error = error;
            error = calculate_error(&rowsums, &Y);
            //println!("error: {}", error);
            if prev_error/error < halt_beta_diff {
                println!("Change was {:.2}. Halting after {} iterations. e: {:.2}", (prev_error/error), iter, error as f64);
                break;
            }
            if iter == 99 {
                println!("reached 100th iteration, halting. e: {:.2}", error as f64);
            }
        }
        lambda *= 0.90;
    }
    println!("done coordinate descent, final mean squared error is {:.2}", error as f64);
    beta
}

fn get_num(n: usize, p: usize) -> Result<(usize, usize), &'static str> {
    let mut offset = 0;
    for i in 0..p {
        for j in i..p {
            if offset == n {
                return Ok((i+1,j+1));
            }
            offset += 1;
        }
    }
    Err("Could not find value")
}

fn calculate_error(rowsums: &Vec<f64>, Y: &Vec<f64>) -> f64 {
    let mut error = 0.0;
    for ind in 0..Y.len() {
        error += (Y[ind] - rowsums[ind] as f64).powi(2);
    }
    error
}

fn update_beta_cyclic(X: &Sparse_Xmatrix, Y: &Vec<f64>, beta: &mut Vec<f64>, n: usize, p: usize, k: usize, rowsum: &mut Vec<f64>, lambda: f64) -> f64 {
    let sumk = X.column_nonzeros[k] as f64;
    let mut sumn = X.column_nonzeros[k] as f64 * beta[k];
    for i in &X.column_indices[k] {
        sumn += Y[*i] - rowsum[*i] as f64;
    }
    let old_beta_k = beta[k];
    if sumk == 0.0 {
        beta[k] = 0.0;
    } else {
        beta[k] = soft_threshold(sumn, lambda*(n as f64)/2.0)/sumk;
    }
    let beta_k_diff = beta[k] - old_beta_k;
    // update rowsums if we have to
    if beta_k_diff != 0.0 {
        for i in &X.column_indices[k] {
            rowsum[*i] += beta_k_diff;
        }
    }
    let squared_beta_k_diff = beta_k_diff*beta_k_diff;

    return squared_beta_k_diff;
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    let abs = z.abs();
    if abs < gamma {
        0.0
    } else {
        let val = abs - gamma;
        if z < 0.0 {
            -val
        } else {
            val
        }
    }
}

// row-major.
fn read_x_csv(file_path: &str) -> XMatrix {
    let mut rows: Vec<Vec<bool>> = Vec::new();
    let file = match File::open(file_path) {
        Ok(x) => x,
        Err(e) => panic!("couldn't open file")
    };
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut row = 0;
    for record in reader.records() {
        let mut new_row: Vec<bool> = Vec::new();
        let actual_record = match record {
            Ok(r) => r,
            Err(_e) => panic!("failed to parse record")
        };
        for value in actual_record.iter().skip(1) {
            let val = value.parse::<u64>().unwrap();
            new_row.push(
                if val > 0 { true }
                else if val == 0 { false }
                else { panic!("read invalid value"); })
        }
        rows.push(new_row);
    }

    return XMatrix { rows };
}

fn read_y_csv(file_path: &str) -> Vec<f64> {
    let mut rows: Vec<f64> = Vec::new();
    let file = match File::open(file_path) {
        Ok(x) => x,
        Err(e) => panic!("couldn't open file")
    };
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_reader(file);

    let mut row = 0;
    for record in reader.records() {
        let actual_record = match record {
            Ok(r) => r,
            Err(_e) => panic!("failed to parse record")
        };
        if actual_record.len() != 2 {
            panic!("Rows in Y should only contain one entry and a label");
        }
        for value in actual_record.iter().skip(1) {
            let val = value.parse::<f64>().unwrap();
            rows.push(val);
        }
    }

    return rows;
}
