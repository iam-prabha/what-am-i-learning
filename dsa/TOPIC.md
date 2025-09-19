## Algorithms analysis

* Time complexity (runtime)
* Space complexity (memory)
* I/O (disks reads/writes) 

### RAM model

* single processor
* random-access machine (RAM)
* computer programs
* instruction executed serially

### Operations

* arithmetic - add,substract,multiply,divide
* data movement - loads,store,copy
* control - branching,subroutine calls/returns
* shift left - 2k
* Not in RAM model:
    * sort
    * exponentiation

* data types: integers and floating points
* word size can't grow arbitrarily
* word = unit of data with a defined bit length that can be handled by the instruction set and moved between storage and the processor.
* ignore memory hierarchy (caches)   

### Runtime 

* number of primitive steps or operations
* each line of pseudocode executes in constant time
* i th executes in c , where c is a constant
* subroutines - calling is constant time, but executing may not be constant time.
* generally grow with size of the input
* input size 
    * sorting - number of items in array
    * graphs - number of vertices or edges