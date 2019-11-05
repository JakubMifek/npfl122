import multiprocessing as mp

def power(x):
    return x**2

pool = mp.Pool()
print(pool.map(power, [1,2,3]))
print(pool.close())
print(pool.join())