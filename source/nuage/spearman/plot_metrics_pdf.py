import os
from infrastructure.file_system import get_path
from routines.plot import plot_table_pdf

path = get_path()

path_in = path + '/spearman/table/'
path_out = path + '/spearman/plot/'

if not os.path.isdir(path_out):
    os.makedirs(path_out)

fn = 'T0_T1_spearman'
header = 'rho'

plot_table_pdf(path_in, path_out, fn, 'rho')
plot_table_pdf(path_in, path_out, fn, 'p-value(FDR)')