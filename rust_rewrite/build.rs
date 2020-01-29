// build.rs

use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    //let flags = [
    //    "-march=corei7-avx",
    //    "-mtune=corei7-avx",
    //    "-Iext",
    //    "-falign-loops",
    //    "-fstrict-aliasing",
    //    "-DAVX2_ON", //TODO: something more portable
    //];

    //let build_files = [
    //    ("TurboPFor/vp4c.c", "vp4c"),
    //    ("TurboPFor/vp4d.c", "vp4d"),
    //    ("TurboPFor/bitunpack.c", "bitunpack"),
    //    ("TurboPFor/bitpack.c", "bitpack"),
    //    ("TurboPFor/bitutil.c", "bitutil"),
    //    ("TurboPFor/vint.c", "vint"),
    //];

    //cc::Build::new()
    //    .file("TurboPFor/bitunpack.c")
    //    .include("TurboPFor/")
    //    .include("TurboPFor/bitpack.h")
    //    .flag("-march=corei7-avx")
    //    .flag("-mtune=corei7-avx")
    //    .flag("-Iext")
    //    .flag("-falign-loops")
    //    .flag("-fstrict-aliasing")
    //    .flag("-lm")
    //    .flag("-lrt")
    //    .warnings(false)
    //    .compile("bitunpack");

    //cc::Build::new()
    //    .object(format!("TurboPFor/bitunpack.o"))
    //    .file("TurboPFor/bitunpack.c")
    //    .file("TurboPFor/bitpack.c")
    //    .file("TurboPFor/bitutil.c")
    //    .file("TurboPFor/vint.c")
    //    .file("TurboPFor/vp4c.c")
    //    .file("TurboPFor/vp4d.c")
    //    .include("TurboPFor/")
    //    .include("TurboPFor/bitpack.h")
    //    .flag("-march=corei7-avx")
    //    .flag("-mtune=corei7-avx")
    //    .flag("-Iext")
    //    .flag("-falign-loops")
    //    .flag("-fstrict-aliasing")
    //    .flag("-D__SSSE3__")
    //    //.flag("-lm")
    //    //.flag("-lrt")
    //    //.flag("-lbitunpack")
    //    .warnings(false)
    //    .compile("test_out");

    //for (filename, output) in &build_files {
    //    cc::Build::new()
    //        .file(filename)
    //        .include("TurboPFor/")
    //        .flag("-march=corei7-avx")
    //        .flag("-mtune=corei7-avx")
    //        .flag("-Iext")
    //        .flag("-falign-loops")
    //        .flag("-fstrict-aliasing")
    //        .warnings(false)
    //        .compile(output);
    //}

    println!("cargo:rustc-link-search=native={}", manifest_dir);
    println!("cargo:rustc-link-lib=static=ic");
    //for (source, libname) in &build_files {
    //    println!("cargo:rustc-link-lib=static={}", libname);
    //}
    //panic!();
}
