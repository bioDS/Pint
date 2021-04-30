project('lasso', 'cpp',
	default_options : ['b_lto=true'])


add_project_arguments([
  '-I' + meson.build_root(),
  '-ffast-math',
  '-fno-stack-protector',
  '-fpermissive',
 '-Ofast',
  '-g',
  # '-Os',
  # '-fno-inline',
], language: 'cpp')

subdir('src')
subdir('tests')
subdir('utils')