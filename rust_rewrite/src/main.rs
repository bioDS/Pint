use std::fs::File;
use clap::{Arg, App};
use bitpacking::{BitPacker4x, BitPacker};
use std::cmp::min;
use rayon::prelude::*;
use crossbeam_utils::atomic::AtomicCell;
use std::sync::Arc;

//TODO: replace actual_entries with final block
//TODO: shuffle column order
//TODO: limit threads

struct XMatrix {
    rows: Vec<Vec<bool>>
}

struct CompressedColumn {
    bytes: Vec<u8>,
    block_inf: Vec<(usize, u8, usize)>, //compressed_len, num_bits, actual_entries
    size: usize,
}

struct ColumnIterator<'a> {
    column: &'a CompressedColumn,
    index: usize,
    bitpacker: BitPacker4x,
    last_value: u32,
    prev_pos: usize,
}

impl Iterator for ColumnIterator<'_> {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Vec<u32>> {
        if self.index == self.column.block_inf.len() {
            return None;
        }
        let bitpacker = self.bitpacker;
        assert!(self.column.block_inf.len() > 0);
        let (compressed_len,num_bits, actual_entries) = self.column.block_inf[self.index];
        let mut decompressed = vec![0_u32; BitPacker4x::BLOCK_LEN];
        bitpacker.decompress_sorted(self.last_value, &self.column.bytes[self.prev_pos..self.prev_pos+compressed_len], &mut decompressed[..], num_bits);
        self.prev_pos += compressed_len;
        self.index += 1;
        self.last_value = decompressed[actual_entries - 1];
        decompressed.resize(actual_entries, 0);
        Some(decompressed)
    }
}

impl<'a> IntoIterator for &'a CompressedColumn {
    type Item = Vec<u32>;
    type IntoIter = ColumnIterator<'a>;

    fn into_iter(self) -> ColumnIterator<'a> {
        ColumnIterator {
            column: self,
            index: 0,
            bitpacker: BitPacker4x::new(),
            last_value: 0,
            prev_pos: 0
        }
    }
}

struct XMatrixCols {
    cols: Vec<Vec<bool>>
}

struct SparseXmatrix {
    n: usize,
    p: usize,
    compressed_columns: Vec<CompressedColumn>,
}

trait MatrixFunctions {
    fn next(&self);
}

impl MatrixFunctions for SparseXmatrix {
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
        let mut actual_count: usize = 0;
        for row in &testX2.rows {
            let size: usize = row.iter().map(|entry| if *entry==true {1} else {0}).sum();
            actual_count += size;
        }
        let bitpacker = X2.bitpacker;
        let mut col_ind = 0;
        let mut count = 0;
        for column in X2.compressed_columns {
            for block in &column {
                for entry in block {
                    if !testX2.rows[entry as usize][col_ind] {
                        println!("col {} row {} not present in testX2.csv (count {})", col_ind, entry, count);
                    }
                    assert!(testX2.rows[entry as usize][col_ind]);
                    count += 1;
                }
            }

            col_ind += 1;
        }
        assert_eq!(count, actual_count);
        assert_eq!(col_ind, (X_col.cols.len()*(X_col.cols.len()+1))/2);
    }

    #[test]
    fn test_bitpacking() {
        let mut rng = rand::thread_rng();
        let mut my_data: Vec<u32> = Vec::new();
        let n = 513;
        for _i in 0..n {
            my_data.push(my_data.last().unwrap_or(&0_u32) + rng.gen_range(0,100));
        }
        let bitpacker = BitPacker4x::new();
        let block_len = BitPacker4x::BLOCK_LEN;
        let mut compressed_col: Vec<u8> = Vec::new();
        let mut compressed_lens = Vec::new();

        println!("bing!");
        // add elements until we have less than one chunk left
        if n % block_len != 0 {
            my_data.resize((n as usize/block_len + 1)*block_len, *my_data.last().unwrap_or(&0));
        }
        println!("new len: {}", my_data.len());
        let mut last_value = 0;
        for ind in 0..(my_data.len() as usize/block_len) {
            let actual_entries = min(n - ind*block_len, block_len);
            println!("actual_entries: {}", actual_entries);
            let num_bits: u8 = bitpacker.num_bits(&my_data[ind*block_len..(ind+1)*block_len]);
            let mut compressed = vec![0_u8; 8*BitPacker4x::BLOCK_LEN];
            let compressed_len = bitpacker.compress_sorted(last_value, &my_data[ind*block_len..(ind+1)*block_len], &mut compressed[..], num_bits);
            last_value = my_data[ind*block_len + actual_entries - 1];
            println!("new last value: {}", last_value);
            println!("adding ({},{})", compressed_len, num_bits);
            compressed_lens.push((compressed_len, num_bits, actual_entries));
            compressed_col.extend_from_slice(&mut compressed[0..compressed_len]);
        }


        let mut prev_pos = 0;
        let mut count = 0;
        let mut found = 0;
        let mut last_value = 0;
        for (compressed_len, num_bits, actual_entries) in &compressed_lens {
            let mut decompressed = vec![0_u32; BitPacker4x::BLOCK_LEN];
            bitpacker.decompress_sorted(last_value, &compressed_col[prev_pos..prev_pos+compressed_len], &mut decompressed[..], *num_bits);
            last_value = decompressed[actual_entries - 1];
            prev_pos += compressed_len;

            assert_eq!(&my_data[count..count+*actual_entries], &decompressed[..*actual_entries]);
            found += actual_entries;
            count += block_len;
        }
        count = 0;
        let cc = CompressedColumn { size: n, bytes: compressed_col, block_inf: compressed_lens };
        for block in &cc {
            for entry in block {
                assert_eq!(entry, my_data[count]);
                count += 1;
            }
        }
        assert_eq!(found, n);
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
    print!("X is {}x{}, ", X_row.rows.len(), X_row.rows[0].len());
    let X_col = row_to_col(X_row);
    let Y = read_y_csv(y_filename);
    println!("Y is {}", Y.len());

    let X2 = x_to_x2_sparse_col(&X_col);

    let beta = simple_coordinate_descent_lasso(X2, Y);

    // find large looking beta values
    for b_ind in 0..beta.len() {
        let b = f64::from_bits(beta[b_ind].load());
        if b.abs() > 500.0 {
            let (i1, i2) = match (get_num(b_ind, X_col.cols.len())) {
                Ok(x) => x,
                Err(e) => panic!(e),
            };
            println!("{} ({},{}): {}", b_ind, i1, i2, b);
        }
    }
}

fn x_to_x2_sparse_col(X: &XMatrixCols) -> SparseXmatrix {
    let p = X.cols.len();
    let p_int = (X.cols.len()*(X.cols.len()+1))/2;
    let n = X.cols[0].len();
    println!("building X2. n = {}, p = {}", n, p);

    let bitpacker = BitPacker4x::new();
    let block_len = BitPacker4x::BLOCK_LEN;
    let mut compressed_columns: Vec<CompressedColumn> = Vec::new();

    for col1_ind in 0..p {
        compressed_columns.par_extend((col1_ind..p).into_par_iter().map(|col2_ind| {
            let mut current_col_indices: Vec<u32> = Vec::new();
            let col1 = &X.cols[col1_ind];
            let col2 = &X.cols[col2_ind];
            let mut size = 0;
            for k in 0..n {
                if col1[k] == true && col2[k] == true{
                    size += 1;
                    current_col_indices.push(k as u32);
                }
            }

            let mut compressed_col: Vec<u8> = Vec::new();
            let mut compressed_lens: Vec<(usize, u8, usize)> = Vec::new();
            let mut last_value = 0;
            if current_col_indices.len() % BitPacker4x::BLOCK_LEN != 0 {
                current_col_indices.resize((current_col_indices.len()/BitPacker4x::BLOCK_LEN +1)*BitPacker4x::BLOCK_LEN, *current_col_indices.last().unwrap_or(&0));
            }

            for ind in 0..(current_col_indices.len() as usize/block_len) {
                let actual_entries = min(size - ind*block_len, block_len);
                let num_bits: u8 = bitpacker.num_bits(&current_col_indices[ind*block_len..(ind+1)*block_len]);
                let mut compressed = vec![0_u8; 8*BitPacker4x::BLOCK_LEN];
                let compressed_len = bitpacker.compress_sorted(last_value, &current_col_indices[ind*block_len..(ind+1)*block_len], &mut compressed[..], num_bits);
                last_value = current_col_indices[ind*block_len + actual_entries - 1];

                compressed_lens.push((compressed_len, num_bits, actual_entries));
                compressed_col.extend_from_slice(&mut compressed[0..compressed_len]);
            }

            CompressedColumn { bytes: compressed_col, block_inf: compressed_lens, size}
        }));
    }

    SparseXmatrix { n, p: p_int, compressed_columns }
}

fn row_to_col(row_X: XMatrix) -> XMatrixCols {
    let rows = row_X.rows;
    let n = rows.len();
    let p = rows[0].len();

    let mut cols: Vec<Vec<bool>> = (0..p).map(|_x| vec![false; n]).collect();

    for row_ind in 0..n {
        for col_ind in 0..p {
            cols[col_ind][row_ind] = rows[row_ind][col_ind];
        }
    }

    XMatrixCols {cols}
}

// X should be column major
fn simple_coordinate_descent_lasso(X: SparseXmatrix, Y: Vec<f64>)  -> Vec<Arc<AtomicCell<u64>>> {
    let p = X.p;
    let n = X.n;
    println!("p: {}, n: {}", p, n);
    //let mut beta = vec![0.0; p];
    let mut beta = Vec::with_capacity(p);
    //let mut rowsum = vec![0.0; n];
    let mut rowsum = Vec::with_capacity(n);//vec![AtomicCell::new(0.0); n];
    for _ in 0..n {
        rowsum.push(Arc::new(AtomicCell::new(0)));
    }
    for _ in 0..p {
        beta.push(Arc::new(AtomicCell::new(0)));
    }
    let mut lambda = 100.0; //TODO: find max lambda;
    let halt_beta_diff = 1.0001;
    let mut error = calculate_error(&rowsum, &Y);
    println!("for testing purposes, initial e is {:.2}", error as f64);
    let mut column_iter: Vec<ColumnIterator> = Vec::new();


    for lambda_seq in 0..50 {
        println!("lambda {}: {}", lambda_seq+1, lambda);
        for iter in 0..100 {
            let mut iter_max_change = 0.0;
            X.compressed_columns.par_iter().enumerate().for_each(|(k, column)|{
                update_beta_cyclic(&column, &Y, &beta, n, p, &rowsum, lambda, k);
            });

            let prev_error = error;
            error = calculate_error(&rowsum, &Y);
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

fn calculate_error(rowsums: &Vec<Arc<AtomicCell<u64>>>, Y: &Vec<f64>) -> f64 {
    let mut error = 0.0;
    for ind in 0..Y.len() {
        error += (Y[ind] - f64::from_bits(rowsums[ind].load()) as f64).powi(2);
    }
    error
}

fn beta_change(column: &CompressedColumn, Y: &Vec<f64>, beta: &Vec<f64>, n: usize, p: usize, rowsum: &Vec<f64>, lambda: f64, k: usize) -> f64 {
    let sumk = column.size as f64;
    let mut sumn = sumk * beta[k];
    let old_beta_k = beta[k];
    for block in column.into_iter() {
        for entry in block {
            sumn += Y[entry as usize] - rowsum[entry as usize] as f64;
        }
    }
    if sumk == 0.0 {
        0.0
    } else {
        soft_threshold(sumn, lambda*(n as f64)/2.0)/sumk - old_beta_k
    }
}
fn update_beta_cyclic(column: &CompressedColumn, Y: &Vec<f64>, beta: &Vec<Arc<AtomicCell<u64>>>, n: usize, p: usize, rowsum: &Vec<Arc<AtomicCell<u64>>>, lambda: f64, k: usize) {
    let sumk = column.size as f64;
    let mut sumn = sumk * f64::from_bits(beta[k].load());
    let mut complete_row: Vec<usize> = Vec::with_capacity(column.size);
    let old_beta_k = f64::from_bits(beta[k].load());
    for block in column.into_iter() {
        for entry in block {
            sumn += Y[entry as usize] - f64::from_bits(rowsum[entry as usize].load());
            complete_row.push(entry as usize);
        }
    }
    if sumk == 0.0 {
        beta[k].store(0);
    } else {
        let new_beta_k = soft_threshold(sumn, lambda*(n as f64)/2.0)/sumk;
        beta[k].store(new_beta_k.to_bits());
    let beta_k_diff = new_beta_k - old_beta_k;
    if beta_k_diff != 0.0 {
        // update rowsums if we have to
            for i in complete_row {
                let mut current_rowsum = rowsum[i].load();
                let mut new_rowsum = current_rowsum + 1;
                while new_rowsum != current_rowsum {
                    current_rowsum = rowsum[i].load();
                    new_rowsum = rowsum[i].compare_and_swap(current_rowsum, (f64::from_bits(current_rowsum) + beta_k_diff).to_bits()); // Will waiting for this eventually be a bottleneck with enough cores?
                }
            }
        }
    }
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
        // ensure all rows are the same size
        assert_eq!(new_row.len(), rows.last().unwrap_or(&new_row).len());
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
