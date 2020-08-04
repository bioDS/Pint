let
	pkgs = import <nixpkgs> {};
in
pkgs.stdenv.mkDerivation {
	name = "my-env";
	buildInputs = [
		pkgs.pkgconfig
		pkgs.R
		pkgs.gcc9
		pkgs.gsl
		pkgs.glib.dev
		pkgs.ninja
		pkgs.meson
		pkgs.ncurses
	];
}
