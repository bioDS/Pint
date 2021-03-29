project('lasso', 'c',
	default_options : ['c_std=gnu11'])


add_project_arguments([
  '-I' + meson.build_root(),
  '-ffast-math',
  '-fno-stack-protector',
  '-Ofast',
  '-g',
], language: 'c')

subdir('src')
subdir('tests')
subdir('utils')