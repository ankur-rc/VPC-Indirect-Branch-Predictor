
This is a C++ based implementation of the **VPC indirect branch prediction algorithm** [1].

It uses the **Merging Path & GShare Indexing Perceptron Predictor** [2] for conditional branch prediction. 

To run all traces, run:

    ./run.sh traces

The file to check out is: my_predictor.h

References:
 1. Kim, H., Joao, J. A., Mutlu, O., Lee, C. J., Patt, Y. N., and Cohn,
R. (2007). VPC prediction. ACM SIGARCH Computer Architecture
News, 35(2), 424. doi:10.1145/1273440.1250715
 2. Tarjan, D., and Skadron, K. (2005). Merging path and gshare indexing
in perceptron branch prediction. ACM Transactions on Architecture
and Code Optimization, 2(3), 280-300. doi:10.1145/1089008.1089011
