# Lesson NEW

So far, we've looked at RAJA kernel launch methods, where a user provides a 
kernel body to implement a parallel algorithm. RAJA provides other parallel
constructs that are important for HPC applications and which can be specialized
for application needs. Such constructs include atomic operations, scans, and sorts,
which are available in RAJA for all supported programming model back-ends.
We discuss each of these constructs in this section.

## Atomic Operations

An **atomic operation** is a parallel construct that allows only one thread or
process at a time to modify data at a memory location. When a thread/process is 
about to write to a memory location, the address is "locked" until the write 
operation is complete. Then, the lock is released and another thread/process can
write to the memory location. If the memory location is locked when a thread/process
is about to write to it, the thread/process must wait until the lock is released
to do so. Such atomic behavior prevents potential memory corruption and can be 
essential for a parallel algorithm to be correct and generate reproducible results.

In lesson 5, we looked at a kernel that used a `RAJA::ReduceSum` object to 
approximate pi by computing a discrete Riemann integral. The OpenMP parallel
implementation looked like the following:

```
using EXEC_POL   = RAJA::omp_parallel_for_exec;
using REDUCE_POL = RAJA::omp_reduce; 

RAJA::ReduceSum<REDUCE_POL, double> pi(0.0);

RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N),
  [=] (int i) {
    double x = (double(i) + 0.5) * dx;
    pi += dx / (1.0 + x * x);
  });
double pi_val = 4.0 * pi.get();
```

An analogous implementation of this pi approximation using a RAJA atomic operation
looks like this:

```
using EXEC_POL   = RAJA::omp_parallel_for_exec;
using ATOMIC_POL = RAJA::omp_atomic;

double* pi = new double{0.0};

RAJA::forall<EXEC_POL>(RAJA::TypedRangeSegment<int>(0, N),
  [=] (int i) {
    double x = (double(i) + 0.5) * dx;
    RAJA::atomicAdd<ATOMIC_POL>( pi, dx / (1.0 + x * x) );
  });
double pi_val = 4.0 * pi;
```

The only substantial difference in the atomic version of the kernel is that a
call to the `RAJA::atomicAdd` method is made to perform the atomic update to the
memory location defined by the pointer `pi`. Note that the method is specialized
to be compatible with the programming model used, OpenMP in this case. This is 
similar to the specialization of the `RAJA::ReduceSum` object above.

Feel free to experiment with other RAJA atomic methods and execution policies in the
example code located in the lesson sub-directory as in the other lessons. To
compile and run:

```
$ make NEW_raja_atomic
$ .bin/NEW_raja_atomic
```

## Parallel Scan

A **scan operation** is an important building block for parallel algorithms. It is
a key primitive to convert sequential operations into parallel implementations.
Scans are based on *reduction tree* and *reverse reduction tree* structures. It is
an example of a computation that looks inherently serial, but for which there exist
efficient parallel implementations. A useful reference that describes usage of
parallel scans is [Prefix Sums and Their Applications by Guy E. Blelloch](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf).

RAJA provides common forms of scan operations, including operators that allow users to implement a particular type of scan. First, we illustrate the former with a **parallel prefix sum**, which is the most common scan operation. A prefix sum operation takes
an array as input and produces an output array containing partial sums of the input
array. Consider the following code:

```
constexpr N = 10;

int in[N] = {8,-1,2,9,10,3,4,1,6,7};
int is_out[N] = {};

RAJA::inclusive_scan<EXEC_POL>( RAJA::make_span(in, N), RAJA::make_span(is_out, N) );

std::cout << "Output (inclusive): "; 
for (int i = 0; i < N; ++i) }
  std::cout << is_out[i] << "  ";
}
std::cout << std::endl;

int es_out[N] = {};

RAJA::exclusive_scan<EXEC_POL>( RAJA::make_span(in, N), RAJA::make_span(es_out, N) );

std::cout << "Output (exclusive): "; 
for (int i = 0; i < N; ++i) }
  std::cout << es_out[i] << "  ";
}
std::cout << std::endl;
```

When this code is run, it will produce the following output:
```
Output (inclusive): 8  7  9  18  28  31  35  36  42  49
Output (exclusive): 0  8  7  9  18  28  31  35  36  42
```

We point out a couple of things about the code itself. First, each scan method
is specialized on an execution policy. RAJA scan methods use the same execution
policies as `RAJA::forall`. Second, the arguments to RAJA scan methods are RAJA
*span* objects which can be created using the `RAJA::make_span` helper method that
takes the address of an array element and the number of elements to define the span
over, in this example `N`.

Regarding the code output, the inclusive scan fills the output array with partial
sums of the output array. Here, the first element of the input array is 8, so the
first element of the output array is 8. The second element of the input array is
-1, so the second element of the output array is 7, which is 8 + (-1). And so on.
The exclusive scan produces the same partial sums but shifts them one slot to the 
right in the output array. The first element in the output array for the exclusive
scan is `0`, the identity of the scan operator, which is `+`, since the RAJA default
scan is a prefix sum when no operator is specified.

RAJA also provides **in-place** scan operations where the scan output is produced
directly in the input array. For example:

```
constexpr N = 10;

int in[N] = {8,-1,2,9,10,3,4,1,6,7};

RAJA::inclusive_scan_inplace<EXEC_POL>( RAJA::make_span(in, N) );

std::cout << "Output (inclusive in-place): ";
for (int i = 0; i < N; ++i) }
  std::cout << in[i] << "  ";
}
std::cout << std::endl;
```

Will produce the following output, which is the same as in the previous code example:
```
Output (inclusive-inplace): 8  7  9  18  28  31  35  36  42  49
```

RAJA provides other operators that can be used with all scan methods. These include
`RAJA::operators::minimum<T>`, `RAJA::operators::maximum<T>`, and
`RAJA::operators::plus<T>`. These operator types are specialized on data type `T`,
which must match the data type of the array(s) you pass to the methods to produce
what you expect. When no operator is specified, as in the code examples above, the 
plus operator will be used.

Feel free to experiment with other RAJA scan operations and execution policies in the
example code located in the lesson sub-directory as in the other lessons. To
compile and run:

```
$ make NEW_raja_scan
$ .bin/NEW_raja_scan
```

## Parallel Sort
