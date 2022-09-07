# Device Copy Constructor
Offers a potential solution to a problem involving a user-created device-side copy constructor.

Running the following should yield an error: 

    $ make error
    ...
    error: cannot pass an argument with a user-provided copy-constructor to a device-side kernel launch
    
A different compilation will fix this error: 

    $ make fixed
    $ ./a.out
    
The code file `error.cu` demonstrates these two scenarios (compiled separately, where `make error` adds the `DEVICE_COPY_CONSTRUCTOR_ERROR` macro while `make fixed` does not). 

A base-kernel cannot invoke a sub-kernel with an argument containing a copy-constructor. This can be fixed by: 
1. Copying the argument to a `char*` in the base-kernel.  
2. Interpreting the argument as its correct type in the sub-kernel.
