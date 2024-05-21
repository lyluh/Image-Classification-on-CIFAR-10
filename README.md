## Testing
In this class we will use unittest framework in python to automatically grade coding problems.
Some of the tests are provided to you, so that you can validate your results.

To run tests:
```
inv test
```

The output should look something like this:
```
> inv test

FFF..
======================================================================
FAIL: test_polyfeatures_fives (public.poly_regression.test_poly_regression.TestPolyReg)
----------------------------------------------------------------------
Traceback (most recent call last):
  File ...
AssertionError: 
Arrays are not almost equal to 6 decimals

(shapes (1,), (20, 1) mismatch)
 x: array([1.])
 y: array([[5.],
       [5.],
       [5.],...
```

You can see that in the top there are 3 `F`'s and 2 `.`'s. `F`'s correspond to failed tests and `.` correspond to correct tests.

There are few things to note:

- Not all tests are equal. Some are worth more points. This will not be displayed when you run `inv test`.
- We **do not** provide you with all tests. There are many that hidden. Even if you pass all *public* tests you may still fail some *hidden* ones. We recommend submitting your code to Gradescope early and often so you can check the autograder score before the deadline, in case you would like to resubmit.

### Testing specific problem
Unfortunately the `unittest` framework doesn't allow for testing specific file.
However, you can run tests against specific problem, using the problem's directory name.
To do so run:
```
inv test --problem <problem-name>
```
For example:
```
> inv test --problem poly_regression

test_fit_and_predict_cubic (test_poly_regression.TestPolyReg) ... ok
test_fit_and_predict_straight_line (test_poly_regression.TestPolyReg) ... ok
test_fit_cubic (test_poly_regression.TestPolyReg) ... ok
test_fit_hard (test_poly_regression.TestPolyReg) ... ok
test_fit_linear (test_poly_regression.TestPolyReg) ... ok
test_fit_straight_line (test_poly_regression.TestPolyReg) ... ok
test_mean_squared_error (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_fives (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_ones (test_poly_regression.TestPolyReg) ... ok
test_polyfeatures_twos (test_poly_regression.TestPolyReg) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.197s

OK
```

