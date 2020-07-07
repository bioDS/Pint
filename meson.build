project('lasso', 'c',
	default_options : ['c_std=gnu99'])

add_project_arguments([
  '-I' + meson.build_root(),
  '-ffast-math',
  '-Ofast',
  '-fPIC',
], language: 'c')

subdir('src')
subdir('tests')
subdir('utils')
