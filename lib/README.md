# Introduction  

This directory contains libraries for the client library.
There are four flavors, the product of OpenMP/No-OpenMP and
static archive (*.a) and shared library.

On Darwin the shared library will end in .dylib and on Linux
it will end in .so 

The shared library has a major advantage for incorperating mpi\_jm
bug fixes.  All you have to do is drop in a new shared library and
rerun your application to get the new code.

Todo:  Implement version in the usual way.

Obviously, none of the libraries should be added to the repository,
they are products of the build.

