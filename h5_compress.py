import h5py
import numpy as np
#

def compress_and_write_to_hdf5(data, filename):
    with h5py.File(filename, 'w') as hf:
        dset = hf.create_dataset('data', data=data, compression='gzip', compression_opts=9)

# Пример использования




def read_compressed_hdf5(filename):
    with h5py.File(filename, 'r') as hf:
        data = hf['data'][:]
    return data


#decompressed_data = read_compressed_hdf5(input('input file name(123.h5): '))


s = input('[1] compres\n[2] decompress\nChoose option: ')
if  s == '1':
    data_to_compress = np.random.random((1000, 1000))
    compress_and_write_to_hdf5(data_to_compress, input('input file name(123.h5): '))
if s == '2':
    decompressed_data = read_compressed_hdf5(input('input file name(123.h5): '))
    with h5py.File(input('output fileName: '), 'w') as hf:
        hf.write(decompressed_data)