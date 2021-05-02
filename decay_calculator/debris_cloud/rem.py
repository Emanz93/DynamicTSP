# -*- coding: utf-8 -*-
import numpy as np

def remove_from_tls(filename_number, filename_in, filename_out):
    numbers = np.loadtxt(filename_number, dtype='int')
    i = 0
    j = 0

    f = open(filename_in, 'r')
    f_out = open(filename_out, 'w')
    line0 = f.readline()
    while line0 != '':
        i = i + 1
        line1 = f.readline()
        line2 = f.readline()

        if not int(line2.split(' ')[1]) in numbers:
            j = j + 1
            f_out.write(line0)
            f_out.write(line1)
            f_out.write(line2)

        line0 = f.readline()

    f.close()
    f_out.close()
    print("Read {} lines. Written {}.".format(i,j))

filename_input = "./debris_cloud/fengyun1C_tle.txt"
filename_output = "./debris_cloud/fengyun1C_tle_new.txt"
file_numbers = "./numbers.txt"
remove_from_tls(file_numbers, filename_input, filename_output)