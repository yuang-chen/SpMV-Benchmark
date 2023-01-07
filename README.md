## PAVA
pava is an SpMV benchmarking framework, supporting almost SpMV methods on Intel Skylake. When the CPU model is changed, please modify the compiler flags `-xSKYLAKE-AVX512` to the corresponding model. 

Available methods based on MKL library:
1. CSR
2. CSC
3. COO
4. DIA
5. IE
6. BSR

Fancy approaches from papers: 

7. ESB
8. CVR
9. CSR5
10. VHCC

The codes are outdated, but still usable. 

requires:
`g++ 4.8, intel OneAPI (icpc, mkl) `

execute:
`./pava /path/to/data #threads`