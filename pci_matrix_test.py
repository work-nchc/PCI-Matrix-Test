from sys import argv
from time import time
from os import environ
environ['PATH'] = environ['PATH'] + 'N:\\nchc\\cuda101\\bin;'
import numpy
import cupy

n_gpu = int(argv[1])
print('n_gpu', n_gpu)

matrix16 = numpy.load(argv[2])
matrix32 = numpy.array(matrix16, 'float32')
matrix64 = numpy.array(matrix16, 'float64')
print('mem', matrix16.__sizeof__())

begin_gpu0 = time()
with cupy.cuda.Device(0):
    matrix_gpu0 = cupy.asarray(matrix16)
cupy.cuda.Stream.null.synchronize()
end_gpu0 = time()
print('gpu0 < cpu', end_gpu0 - begin_gpu0)

for i in range(1, n_gpu):
    begin_gpu = time()
    with cupy.cuda.Device(i):
        matrix_gpu = cupy.asarray(matrix_gpu0)
        cupy.cuda.Stream.null.synchronize()
        end_gpu = time()
        print('gpu{} < gpu0'.format(i), end_gpu - begin_gpu)
        del matrix_gpu, begin_gpu, end_gpu
    cupy.cuda.Stream.null.synchronize()

begin_cpu = time()
with cupy.cuda.Device(0):
    matrix_cpu = matrix_gpu0.get()
    cupy.cuda.Stream.null.synchronize()
    end_cpu = time()
    print('cpu < gpu0', end_cpu - begin_cpu)
    
    begin_16_gpu = time()
    product_gpu0 = cupy.tensordot(matrix_gpu0, matrix_gpu0, 1)
    cupy.cuda.Stream.null.synchronize()
    end_16_gpu = time()
    print('gpu float16', end_16_gpu - begin_16_gpu)
    del matrix_gpu0, product_gpu0
    cupy.cuda.Stream.null.synchronize()
    
    matrix_gpu0 = cupy.asarray(matrix32)
    cupy.cuda.Stream.null.synchronize()
    begin_32_gpu = time()
    product_gpu0 = cupy.tensordot(matrix_gpu0, matrix_gpu0, 1)
    cupy.cuda.Stream.null.synchronize()
    end_32_gpu = time()
    print('gpu float32', end_32_gpu - begin_32_gpu)
    del matrix_gpu0, product_gpu0
    cupy.cuda.Stream.null.synchronize()
    
    matrix_gpu0 = cupy.asarray(matrix64)
    cupy.cuda.Stream.null.synchronize()
    begin_64_gpu = time()
    product_gpu0 = cupy.tensordot(matrix_gpu0, matrix_gpu0, 1)
    cupy.cuda.Stream.null.synchronize()
    end_64_gpu = time()
    print('gpu float64', end_64_gpu - begin_64_gpu)
'''
begin_16_cpu = time()
product16 = numpy.tensordot(matrix16, matrix16, 1)
end_16_cpu = time()
print('cpu float16', end_16_cpu - begin_16_cpu)
'''
begin_32_cpu = time()
product32 = numpy.tensordot(matrix32, matrix32, 1)
end_32_cpu = time()
print('cpu float32', end_32_cpu - begin_32_cpu)

begin_64_cpu = time()
product64 = numpy.tensordot(matrix64, matrix64, 1)
end_64_cpu = time()
print('cpu float64', end_64_cpu - begin_64_cpu)
