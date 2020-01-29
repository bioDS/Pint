use libc::size_t;

//#[link(name = "vp4c", kind="static")]
#[link(name = "ic", kind="static")]
extern {
    pub fn p4ndenc32(in_: *const u32, n: usize, out: *mut ::std::os::raw::c_uchar) -> usize;
    pub fn p4nddec32(in_: *const::std::os::raw::c_uchar, n: usize, out: *mut u32) -> usize;
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
        let mut seq2: Vec<u32> = (0..100).collect();
        let mut compressed_size = 0;
        let mut compressed_size2 = 0;
        unsafe {
            let mut test_arr = Box::new(vec![0_u32, array_size as u32]);

            let compressed1: Vec<u8> = compress_delta(&mut seq[..]);
            compressed_size = compressed1.len();
            let mut compressed_seq = compress_delta(&mut seq2);
            compressed_size2 = compressed_seq.len();

            println!("Compressed size: {}", compressed_size);
            println!("Compressed size 2: {}", compressed_size2);

            let mut decompressed_seq = vec![0_u32; array_size*2];
            let decomrpessed_size = decompress_delta(&mut compressed_seq, array_size, &mut decompressed_seq);

            for element in &decompressed_seq {
                print!("{} ", element);
            }
            println!();
        }
    }
}

pub fn compress_delta(seq: &[u32]) -> Vec<u8> {
    if seq.len() == 0 {
        return vec![0_u8; 0];
    }
    let size = seq.len();
    let mut buffer = vec![0_u8; 16*size];
    let mut compressed_size = -1;
    //println!("size {}, capacity {}", size, buffer.capacity());
    unsafe {
        compressed_size = p4ndenc32(seq.as_ptr(), size, buffer.as_mut_ptr()) as isize;
    }
    if compressed_size < 0 {
        panic!("failed to compress buffer");
    }
    buffer[..compressed_size as usize].to_vec()
}

pub fn decompress_delta(arr: &[u8], size: usize, seq: &mut Vec<u32>) -> usize {
    let mut bytes_read = -1;
    seq.resize(size, 0);
    unsafe {
        bytes_read = p4nddec32(arr.as_ptr(), size, seq.as_mut_ptr()) as isize;
    }
    bytes_read as usize
}
