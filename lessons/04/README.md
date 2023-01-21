# Lesson Four

In this lesson you will learn to write a loop using the RAJA::forall statement.

The RAJA::forall loop execution method is a template that takes an execution
policy type template parameter. A RAJA::forall method takes two arguments: an
iteration space object, such as a contiguous range of loop indices as shown
here, and a single lambda expression representing the loop kernel body.




