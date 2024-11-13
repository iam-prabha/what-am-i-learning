/* ---String objects can produce unexpected results and slows down execution speed.--
 let x = "John";
  let y = new String("John");
 console.log(x === y); // false

 let z = new String("John");
 console.log(z === x); console.log(z === y); // Both will return false

NOTE:: Comparing two JavaScript objects always returns false.
*/

/*-----------------Js timing------------------

setTimeout(() => {
     console.log("Delayed for 1 second.");
 }, 1000);
*/

/* -----Dot notation------

 let x = {
     name: 'bob',
     age: 22
 }

 let person = {
     ...x,
     age: 24,
     gender: 'male'
 }
 console.log(person);
 console.log(x);
 console.log(person);
*/

/*-------------------Manipulating Arrays--------------
Pushing and popping
Arrays can also function as a stack. The push and pop methods insert and remove variables from the end of an array.

For example, let's create an empty array and push a few variables.

let myStack = [];
myStack.push(1);
myStack.push(2);
myStack.push(3);
console.log(myStack);

After pushing variables to the array, we can then pop variables off from the end.

console.log(myStack.pop());
console.log(myStack);


Queues using shifting and unshifting
The unshift and shift methods are similar to push and pop, only they work from the beginning of the array. We can use the push and shift methods consecutively to utilize an array as a queue. For example:

var myQueue = [];
myQueue.push(1);
myQueue.push(2);
myQueue.push(3);

console.log(myQueue.shift());
console.log(myQueue.shift());
console.log(myQueue.shift());

The unshift method is used to insert a variable at the beginning of an array. For example:

var myArray = [1, 2, 3];
myArray.unshift(0);
console.log(myArray);
*/

/*-------Pop-up Boxes----------
There are three types of pop-up boxes in javascript: confirm, alert, and prompt. To use any of them, type

confirm("Hi!");
prompt("Bye!");
alert("Hello");

Confirm boxes will return "true" if ok is selected, and return "false" if cancel is selected. Alert boxes will not return anything. Prompt boxes will return whatever is in the text box. Note: prompt boxes also have an optional second parameter, which is the text that will already be in the text box.
*/


/*-----Promises----
Promises are the basics of asynchronous programming in JavaScript, and are very important to master.

What is Asynchronous Programming?
Asynchronous programming, or in short, async programming, is a method of programming which enables different parts of code to run at changing times, instead of immediately.

This is mostly required when we want to fetch information from some remote server, and write code which does something with what the server returned:

function getServerStatus() {
    const result = fetch("/server/status");

    // THIS WILL NOT WORK!
    console.log("The status from the server is: ", result.ok);
}

In many programming languages such as Python, this approach would work, because functions are by default synchronous functions.

In JavaScript, most APIs which require waiting for a function to do something, are asynchronous by default which means that this code will not do what we think it will do, since the fetch function is asynchronous, and therefore will return something which is not exactly the result, but will eventually be the result.This "thing" which is returned from the fetch function is called a Promise in JavaScript.

To make the code above work, we will need to write the function in the following manner:

function getServerStatus() {
    // Could be GET or POST/PUT/PATCH/DELETE
    const result = fetch('https://dummyjson.com/test');

    // THIS WILL WORK!
    result.then(function (status) {
        console.log("The status from the server is: ", status.ok);
    });
}
getServerStatus();
//Notice that we used the then function here, which is one of the methods of a Promise.

---The Promise Object---
A Promise is a native JavaScript object which has two traits: 1. It receives a single argument which is a function. This function needs to have two arguments, a resolve function and a reject function. The code written inside the promise needs to use one of these two functions. 2. It can be waited on using the then method (and other similar methods), or the await statement. (The async / await statements have a separate tutorial).

An asynchronous function is defined by a function, which instead of returning the value it was supposed to return, it returns a Promise object, which will eventually resolve and give the user the answer.

For example, let's say that we would like to calculate the sum of two numbers, but by writing a function which returns a Promise and not the value.

function sumAsync(x, y) {
    const p = new Promise((resolve, reject) => {
        // this resolves the promise we just created with the output of x+y
        resolve(x + y);
    });

    // This returns the promise, not the value
    return p;
}
console.log(sumAsync(1, 2));
let's use the function now
sumAsync(5, 7).then((result) => {
    console.log("The result of the addition is:", result);
});
When can this be very useful ? When the calculation needs to happen indirectly, for example after waiting a while or when retrieving information from the server using the fetch command for example.

Let's modify the example to resolve the solution only after a half a second: 

function sumAsync(x, y) {
    console.log("1. sumAsync is executed");
    const p = new Promise((resolve, reject) => {
        // run this in 500ms from now
        setTimeout(() => {
            console.log("4. Resolving sumAsync's Promise with the result after 500ms");
            resolve(x + y);
        }, 500);

        // we don't need to return anything
        console.log("2. sumAsync Promise is initialized");
    });
    console.log("3. sumAsync has returned the Promise");
    return p;
}

console.log(sumAsync(2, 2));

let's use the function now
sumAsync(5, 7).then((result) => {
    console.log("5. The result of the addition is:", result);
});

Rejecting promises
In a synchronous flow, if we want to tell the user that something went wrong so he can catch an exception, we throw an exception using the throw argument.When using promises, we need to trigger the reject function instead.

Let's say we want to write the same function, but with a rejection if a value is negative: 

function sumAsync(x, y) {
    return new Promise((resolve, reject) => {
        // run this in 500ms from now
        setTimeout(() => {
            if (x < 0 || y < 0) {
                reject("Negative values received");
            } else {
                resolve(x + y);
            }
        }, 500);

        // we don't need to return anything
    });
}

sumAsync(-5, 7).then((result) => {
    console.log("The result of the addition is:", result);
}).catch((error) => {
    console.log("Error received:", error);
});*/