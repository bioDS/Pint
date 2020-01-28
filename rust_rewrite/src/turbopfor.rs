use libc::size_t;

//#[link(name = "vp4c", kind="static")]
#[link(name = "ic", kind="static")]
extern {
    pub fn p4ndenc32(in_: *mut u32, n: usize, out: *mut ::std::os::raw::c_uchar) -> usize;
    pub fn p4nddec32(in_: *mut ::std::os::raw::c_uchar, n: usize, out: *mut u32) -> usize;
}
//#[link(name = "vp4d", kind="static")]
//#[link(name = "bitunpack", kind="static")]
//#[link(name = "ic", kind="static")]
//extern {
//}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pfor() {
        println!("Hello, world!");

        let array_size = 100;
        let mut seq = [2_u32; 100];
        //let mut seq2 = Box::new((1..100).collect());
        let mut seq2: Vec<u32> = (0..100).collect();
        //let seq = vec![1_u32..100_u32];
        //let test_seq = Box::new(seq.clone());
        //let test_seq = [1; 100];
        let mut compressed_size = 0;
        let mut compressed_size2 = 0;
        unsafe {
            let arr_ptr = libc::malloc(std::mem::size_of::<i32>() * array_size * 1);
            let arr = arr_ptr as *mut u8;
            let arr2_ptr = libc::malloc(std::mem::size_of::<i32>() * array_size * 1);
            let arr2 = arr2_ptr as *mut u8;

            let mut test_arr = Box::new(vec![0_u32, array_size as u32]);

            let compressed_size_test = compress_delta(&mut seq[..], array_size, arr2);
            compressed_size = compress_delta(&mut seq[..], array_size, arr);
            compressed_size2 = compress_delta(&mut seq2, array_size, arr2);

            println!("Compressed size: {}", compressed_size);
            println!("Compressed size 2: {}", compressed_size2);

            let mut decompressed_seq = vec![0_u32; array_size * 2];
            let decomrpessed_size = decompress_delta(arr2, array_size, &mut decompressed_seq);

            for element in &decompressed_seq[..array_size] {
                print!("{} ", element);
            }
            println!();


            libc::free(arr_ptr);
            libc::free(arr2_ptr);

            println!("freed stuff from rust");
        }
    }
}

fn compress_delta(seq: &mut [u32], size: usize, arr: *mut ::std::os::raw::c_uchar) -> usize {
    unsafe {
        p4ndenc32(seq.as_mut_ptr(), size, arr)
    }
}
fn decompress_delta(arr: *mut ::std::os::raw::c_uchar, size: usize, seq: &mut [u32]) -> usize {
    unsafe {
        p4nddec32(arr, size, seq.as_mut_ptr())
    }
}
