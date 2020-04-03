use std::fs::File;
use bitpacking::{BitPacker4x, BitPacker};
use std::cmp::min;
use rayon::prelude::*;
use crossbeam_utils::atomic::AtomicCell;
use std::sync::Arc;
use streaming_iterator::StreamingIterator;
use std::cell::{RefCell, RefMut};
use rand::{thread_rng, Rng};
use rand::seq::SliceRandom;

mod turbopfor;
use crate::turbopfor::*;

//TODO: limit threads

pub struct XMatrix {
    pub rows: Vec<Vec<bool>>
}

pub struct CompressedColumn {
    bytes: Vec<u8>,
    block_inf: Vec<(usize, u8, usize)>, //compressed_len, num_bits, actual_entries
    size: usize,
    index: usize,
}

pub struct TurboPFor_Compressed_Column {
    bytes: Vec<u8>,
    size: usize,
    index: usize
}

pub struct ColumnIterator<'a> {
    decompressed_block: Vec<u32>,
    column: &'a CompressedColumn,
    index: usize,
    bitpacker: BitPacker4x,
    last_value: u32,
    prev_pos: usize,
    read_entries: usize,
}

trait ReusableStreamingIterator {
    fn clean(&mut self);
}

impl<'a> ReusableStreamingIterator for ColumnIterator<'a> {
    fn clean(&mut self) {
        self.index = 0;
        self.last_value = 0;
        self.prev_pos = 0;
        self.read_entries = 0;
    }
}

trait IntoStreamingIterator<'a> {
    fn into_streaming_iter(self) -> ColumnIterator<'a>;
}

impl<'a> StreamingIterator for ColumnIterator<'a> {
    type Item = [u32];

    fn advance (&mut self) {
        if self.index == self.column.block_inf.len() {
            self.index += 1;
            return;
        }
        let bitpacker = self.bitpacker;
        assert!(self.column.block_inf.len() > 0);
        let (compressed_len,num_bits, actual_entries) = self.column.block_inf[self.index];
        //let mut decompressed = vec![0_u32; BitPacker4x::BLOCK_LEN];
        //let decompressed = &mut self.decompressed_block;
        bitpacker.decompress_sorted(self.last_value, &self.column.bytes[self.prev_pos..self.prev_pos+compressed_len], &mut self.decompressed_block[..], num_bits);
        self.prev_pos += compressed_len;
        self.index += 1;
        self.last_value = self.decompressed_block[actual_entries - 1];
        self.read_entries = actual_entries;
        //decompressed.resize(actual_entries, 0);
        //Some(self.decompressed_block)
    }

    fn get(&self) -> Option<&[u32]> {
        if self.index > self.column.block_inf.len() {
            None
        } else {
            Some(&self.decompressed_block[..self.read_entries])
        }
    }
}

impl<'a> IntoStreamingIterator<'a> for &'a CompressedColumn {
    //type Item = &'a Vec<u32>;
    //type IntoIter = ColumnIterator<'a>;

    fn into_streaming_iter(self) -> ColumnIterator<'a> {
        ColumnIterator {
            decompressed_block: vec![0_u32; BitPacker4x::BLOCK_LEN],
            column: self,
            index: 0,
            bitpacker: BitPacker4x::new(),
            last_value: 0,
            prev_pos: 0,
            read_entries: 0,
        }
    }
}

pub struct XMatrixCols {
    pub cols: Vec<Vec<bool>>
}

pub struct SparseXmatrix {
    n: usize,
    p: usize,
    compressed_columns: Vec<CompressedColumn>,
}

pub struct TurboPFor_Sparse_Xmatrix {
    n: usize,
    p: usize,
    compressed_columns: Vec<TurboPFor_Compressed_Column>,
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

    //#[test]
    //fn test_sparse_x2() {
    //    let X_row = read_x_csv("testX.csv");
    //    let X_col = row_to_col(X_row);
    //    let Y = read_y_csv("testY.csv");
    //    let X2 = x_to_x2_sparse_col(&X_col);


    //    let testX2 = read_x_csv("./testX2.csv");
    //    let mut actual_count: usize = 0;
    //    for row in &testX2.rows {
    //        let size: usize = row.iter().map(|entry| if *entry==true {1} else {0}).sum();
    //        actual_count += size;
    //    }
    //    let bitpacker = BitPacker4x;
    //    let mut col_ind = 0;
    //    let mut count = 0;
    //    for column in X2.compressed_columns {
    //        let mut column_iter = column.into_streaming_iter();
    //        let old_count = count;
    //        while let Some(block) = column_iter.next() {
    //        //for block in &column {
    //            for entry in block {
    //                if !testX2.rows[*entry as usize][col_ind] {
    //                    println!("col {} row {} not present in testX2.csv (count {})", col_ind, entry, count);
    //                }
    //                assert!(testX2.rows[*entry as usize][col_ind]);
    //                count += 1;
    //            }
    //        }
    //        assert_eq!(count - old_count, column.size);

    //        col_ind += 1;
    //    }
    //    assert_eq!(count, actual_count);
    //    assert_eq!(col_ind, (X_col.cols.len()*(X_col.cols.len()+1))/2);
    //}

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
        //let cc = CompressedColumn { size: n, bytes: compressed_col, block_inf: compressed_lens };
        //let mut cc_iter = cc.into_streaming_iter();
        //while let Some(block) = cc_iter.next() {
        //    for entry in block {
        //        assert_eq!(*entry, my_data[count]);
        //        count += 1;
        //    }
        //}
        //assert_eq!(found, n);
    }

    #[test]
    fn test_turbopfor_matrix() {
        let X_row = read_x_csv("testX.csv");
        let X_col = row_to_col(X_row);
        let Y = read_y_csv("testY.csv");
        let mut X2 = x_to_x2_sparse_col_turbopfor(&X_col);

        let testX2 = read_x_csv("./testX2.csv");
        let mut actual_count: usize = 0;
        for row in &testX2.rows {
            let size: usize = row.iter().map(|entry| if *entry==true {1} else {0}).sum();
            actual_count += size;
        }
        let mut col_ind = 0;
        let mut count = 0;
        let mut decompressed_column_buffer = vec![0_u32; X2.n];
        for compressed_column in X2.compressed_columns {
            decompress_delta(&compressed_column.bytes[..], compressed_column.size, &mut decompressed_column_buffer);
            for entry in &decompressed_column_buffer {
                println!("checking col {}, entry {}", col_ind, entry);
                if !testX2.rows[*entry as usize][col_ind] {
                    println!("col {} row {} not present in testX2.csv (count {})", col_ind, entry, count);
                }
                assert!(testX2.rows[*entry as usize][col_ind]);
                count += 1;
            }

            col_ind += 1;
        }
        assert_eq!(count, actual_count);
        assert_eq!(col_ind, (X_col.cols.len()*(X_col.cols.len()+1))/2);
    }
}


pub fn x_to_x2_sparse_col_turbopfor(X: &XMatrixCols) -> TurboPFor_Sparse_Xmatrix {
    let p = X.cols.len();
    let p_int = (X.cols.len()*(X.cols.len()+1))/2;
    let n = X.cols[0].len();
    println!("building X2. n = {}, p = {}", n, p);
    let mut col_ind = 0;

    //let mut compressed_columns: Vec<TurboPFor_Compressed_Column> = Vec::new();
    //let mut compressed_col_buffer = vec![0_u8; n*4];

    let compressed_columns = ((0..p).into_par_iter().flat_map(|col1_ind| {
        let compressed_combined_col: Vec<TurboPFor_Compressed_Column> = (col1_ind..p).into_par_iter().map(|col2_ind| {
            let mut current_col_indices: Vec<u32> = Vec::new();
            //compressed_col_buffer.clear(); // shouldn't actually be necessary
            let col1 = &X.cols[col1_ind];
            let col2 = &X.cols[col2_ind];
            let mut size = 0;
            for k in 0..n {
                if col1[k] == true && col2[k] == true{
                    size += 1;
                    current_col_indices.push(k as u32);
                }
            }

            let compressed_bytes = compress_delta(&mut current_col_indices[..]);

            assert_eq!(size, current_col_indices.len());
            //println!("pushing size {}, lens {}", size, compressed_lens.len());
            let current_col_ind = (2*(p as isize -1) + 2*(p as isize -1)*(col1_ind as isize -1) - (col1_ind as isize -1)*(col1_ind as isize -1) - (col1_ind as isize -1))/2 + col2_ind as isize;
            TurboPFor_Compressed_Column { bytes: compressed_bytes, size, index: current_col_ind as usize}
        }).collect();
        compressed_combined_col
    })).collect();
    //let mut current_col_indices: Vec<u32> = Vec::new();
    //for col1_ind in 0..p {
    //    for col2_ind in col1_ind..p {
    //        current_col_indices.clear();
    //        compressed_col_buffer.clear(); // shouldn't actually be necessary
    //        let col1 = &X.cols[col1_ind];
    //        let col2 = &X.cols[col2_ind];
    //        let mut size = 0;
    //        for k in 0..n {
    //            if col1[k] == true && col2[k] == true{
    //                size += 1;
    //                current_col_indices.push(k as u32);
    //            }
    //        }

    //        let compressed_bytes = compress_delta(&mut current_col_indices[..]);

    //        assert_eq!(size, current_col_indices.len());
    //        //println!("pushing size {}, lens {}", size, compressed_lens.len());
    //        compressed_columns.push(TurboPFor_Compressed_Column { bytes: compressed_bytes, size, index: col_ind});

    //        col_ind += 1;
    //    }
    //}

    TurboPFor_Sparse_Xmatrix { n, p: p_int, compressed_columns}
}

pub fn x_to_x2_sparse_col(X: &XMatrixCols) -> SparseXmatrix {
    let p = X.cols.len();
    let p_int = (X.cols.len()*(X.cols.len()+1))/2;
    let n = X.cols[0].len();
    println!("building X2. n = {}, p = {}", n, p);

    let bitpacker = BitPacker4x::new();
    let block_len = BitPacker4x::BLOCK_LEN;

    //compressed_columns.
    let compressed_columns = ((0..p).into_par_iter().flat_map(|col1_ind| {
        let compressed_combined_col: Vec<CompressedColumn> = (col1_ind..p).into_par_iter().map(|col2_ind| {
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

            let current_col_ind = (2*(p as isize -1) + 2*(p as isize -1)*(col1_ind as isize -1) - (col1_ind as isize -1)*(col1_ind as isize -1) - (col1_ind as isize -1))/2 + col2_ind as isize;
            CompressedColumn { bytes: compressed_col, block_inf: compressed_lens, size, index: current_col_ind as usize}
        }).collect();
        compressed_combined_col
    })).collect();

    SparseXmatrix { n, p: p_int, compressed_columns }
}

pub fn row_to_col(row_X: XMatrix) -> XMatrixCols {
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

thread_local! {
    static COLUMN_CACHE: RefCell<Vec<u32>> = RefCell::new(Vec::new());
}

// X should be column major
pub fn simple_coordinate_descent_lasso(mut X: TurboPFor_Sparse_Xmatrix, Y: Vec<f64>)  -> Vec<Arc<AtomicCell<u64>>> {
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


    for lambda_seq in 0..50 {
        println!("lambda {}: {}", lambda_seq+1, lambda);
        for iter in 0..100 {
            let mut iter_max_change = 0.0;
            //X.compressed_columns.shuffle(&mut thread_rng());
            //X.compressed_columns.par_iter().enumerate().for_each(|(k, column)|{
            X.compressed_columns.shuffle(&mut thread_rng());
            X.compressed_columns.par_iter_mut().enumerate().for_each(|(k, mut column_iter)|{
            //X.compressed_columns.iter_mut().enumerate().for_each(|(k, mut column_iter)|{
                COLUMN_CACHE.with(|mut cache| {
                    update_beta_cyclic(&mut column_iter, &Y, &beta, n, p, &rowsum, lambda, cache.borrow_mut());
                });
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

pub fn get_num(n: usize, p: usize) -> Result<(usize, usize), &'static str> {
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

fn update_beta_cyclic(column: &mut TurboPFor_Compressed_Column, Y: &Vec<f64>, beta: &Vec<Arc<AtomicCell<u64>>>, n: usize, p: usize, rowsum: &Vec<Arc<AtomicCell<u64>>>, lambda: f64, mut complete_row: RefMut<Vec<u32>>) {
    let k = column.index;
    let sumk = column.size as f64;
    let mut sumn = sumk * f64::from_bits(beta[k].load());
    //let mut complete_row: Vec<usize> = Vec::with_capacity(column.size);
    complete_row.clear();
    let old_beta_k = f64::from_bits(beta[k].load());
    //for block in column.into_iter() {

    // use the function for debugging/profiling
    read_iter_loop(column, rowsum, &mut complete_row, &mut sumn, Y);
    //let mut column_iter = column.into_streaming_iter();
    //while let Some(block) = column_iter.next() {
    //    for entry in block {
    //        sumn += Y[*entry as usize] - f64::from_bits(rowsum[*entry as usize].load());
    //        complete_row.push(*entry as usize);
    //    }
    //}

    if sumk == 0.0 {
        beta[k].store(0);
    } else {
        let new_beta_k = soft_threshold(sumn, lambda*(n as f64)/2.0)/sumk;
        beta[k].store(new_beta_k.to_bits());
    let beta_k_diff = new_beta_k - old_beta_k;
    if beta_k_diff != 0.0 {
        // update rowsums if we have to
            for i in complete_row.iter() {
                atomic_inc(&rowsum[*i as usize], beta_k_diff);
                //let mut current_rowsum = rowsum[i].load();
                //let mut new_rowsum = current_rowsum + 1;
                //while new_rowsum != current_rowsum {
                //    current_rowsum = rowsum[i].load();
                //    new_rowsum = rowsum[i].compare_and_swap(current_rowsum, (f64::from_bits(current_rowsum) + beta_k_diff).to_bits()); // Will waiting for this eventually be a bottleneck with enough cores?
                //}
            }
        }
    }
}

fn read_iter_loop(column: &mut TurboPFor_Compressed_Column, rowsum: &Vec<Arc<AtomicCell<u64>>>, complete_row: &mut RefMut<Vec<u32>>, sumn: &mut f64, Y: &[f64]) {
    decompress_delta(&column.bytes, column.size, complete_row.as_mut());
    for entry in complete_row.as_slice() {
        *sumn += Y[*entry as usize] - f64::from_bits(rowsum[*entry as usize].load());
    }
}

fn atomic_inc(cell: &Arc<AtomicCell<u64>>, inc_value: f64) {
                let mut current_value = cell.load();
                let mut new_value = current_value + 1;
                while new_value != current_value {
                    current_value = cell.load();
                    new_value = cell.compare_and_swap(current_value, (f64::from_bits(current_value) + inc_value).to_bits()); // Will waiting for this eventually be a bottleneck with enough cores?
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
pub fn read_x_csv(file_path: &str) -> XMatrix {
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

pub fn read_y_csv(file_path: &str) -> Vec<f64> {
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
