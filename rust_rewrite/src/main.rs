use clap::{Arg, App};
use lasso_rust::*;

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
    assert_eq!(X_col.cols[0].len(), Y.len());

    let X2 = x_to_x2_sparse_col_turbopfor(&X_col);

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
