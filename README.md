# AVX2-optimized sin(), cos(), exp() and log() functions

And also some other functions for converting between different output layout.

# Introduction

This is based on the work by [Yu Yang](https://github.com/reyoung) in [reyoung/avx_mathfun](https://github.com/reyoung/avx_mathfun) that made the old code [here](http://software-lisc.fbk.eu/avx_mathfun/) to be compatible with newer GCC versions.

Main changes are:

* AVX2 is required. (This is mostly due to the usecase I'm interested in)
* Rename functions to be under a namespace (`a2m_` prefix)
* Reorganize headers to be compatible with both C and C++
* Move some code into `*.cpp` file and compile it as a shared library.

    `a2m_sincosf` is still available in the header for cases where inlining is desired.

* Remove alignment requirment on the input parameter to `a2m_sincosf`.
* Add `a2m_cisf` function as well as a few other ones to make it easier to deal
  with complex numbers.
* Tests are not included (yet)

## License

The origin file uses zlib license. It is not changed.
