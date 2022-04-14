# NumMy

I redesigned **NumPy library** and implemented it in C++. Many functions works faster than original NumPy impelementation.
This functions are enough to implement self-written **Neural Network framework** in pure C++.

Faster speed are achieved by redesigning structure of NumPy to more optimal and erasing checking in functions (if function gets wrong parameters --> UB). 

My tests https://docs.google.com/spreadsheets/d/1mevF21Vb8EghB1VBgujWzGI51sdI0pcsKuoB3skUPHo/edit?usp=sharing

There are plans to add more functions, change h-type to classic design with .h/.cpp files, add more unit-tests, documentation.
