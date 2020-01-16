project('lasso', 'c')

add_project_arguments([
  '-I' + meson.build_root(),
  '-ffast-math',
  '-Ofast'
], language: 'c')

subdir('src')
subdir('tests')
subdir('utils')
