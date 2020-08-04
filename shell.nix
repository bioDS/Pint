let
	pkgs = import (fetchTarball https://github.com/NixOS/nixpkgs/archive/20.03.tar.gz) {};
in
pkgs.stdenv.mkDerivation {
	name = "my-env";
	buildInputs = [
		pkgs.pkgconfig
		pkgs.R
		pkgs.rPackages.dplyr
		pkgs.rPackages.Matrix
		pkgs.gcc9
		pkgs.gsl
		pkgs.glib.dev
		pkgs.ninja
		pkgs.meson
		pkgs.ncurses
	];
}
