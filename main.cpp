#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <sys/mman.h>
#include <dirent.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <list>
#include <vector>
#include <set>
#include <cmath>

#include <immintrin.h>
#include <mkl.h>



#include "csr5/csr5.h"
#include "cvr/cvr.h"
#include "vhcc/vhcc.h"
#include "esb/esb.h"

#include "utils/microtime.h"
#include "utils/basicMatrix.h"
#include "utils/pava_formats.h"


using namespace std;

#ifndef TYPE_PRECISION
#define TYPE_PRECISION

//#define TYPE_SINGLE
#define TYPE_DOUBLE

#define FLOAT_TYPE double

#ifdef TYPE_SINGLE
#define SIMD_LEN 16
#else
#define SIMD_LEN 8
#endif

#endif

#define EPS 1.0e-15


#define LARGE_ITERS 100
#define SMALL_ITERS 100

int conduct_benchmark(char* fileName, int numThreads, _PAVA_CSRMatrix* fileCSRMatrix);

int readCSR(char* matrixName, struct _PAVA_COOMatrix *cooMatrix) {
    int nRows, nCols, nElements;  
   std::ifstream input_file (matrixName, std::ios::binary);
   if(!input_file.is_open()) {
      std::cout << "cannot open the input csr file!" << '\n';
      exit(1);
   }

   input_file.read(reinterpret_cast<char*>(&nRows), sizeof(int));
   input_file.read(reinterpret_cast<char*>(&nElements), sizeof(int));

   nCols = nRows;
   int nElements_padding = (nElements%16 == 0) ? nElements : (nElements + 16)/ 16 * 16;


  std::cout<<"==========================================================================="<<std::endl;
    std::cout<<"=========*********  Informations of the sparse matrix   *********=========="<<std::endl;
    std::cout<<std::endl;
//    std::cout<<" numRows is "<<nRows<<"; numCols is "<< nCols<<";  Number of Elements is "<< nElements<<";  After Padding, Number of Elements is "<< nElements_padding<<std::endl;
    std::cout<<"     Number of Rows is :"<< nRows<<std::endl;
    std::cout<<"  Number of Columns is :"<< nCols<<std::endl;
    std::cout<<" Number of Elements is :"<< nElements<<std::endl;
    std::cout<<"       After Alignment :"<< nElements_padding<<std::endl;
    std::cout<<std::endl;
  std::cout<<"==========================================================================="<<std::endl;
   
    double* val_ptr = (double*) MKL_malloc(sizeof(double) * nElements_padding, ALIGN512); 
    int *cols_ptr = (int*) MKL_malloc(sizeof(int) * nElements_padding, ALIGN512);
    int *rows_ptr = (int*) MKL_malloc(sizeof(int) * nElements_padding, ALIGN512);

    int *rowDelimiters_ptr = (int*) MKL_malloc(sizeof(int) * (nRows+2), ALIGN512);
    
    double *vals = val_ptr; 
    int *cols = cols_ptr; 
    int *rows = rows_ptr;
    int *rowDelimiters = rowDelimiters_ptr; 

   input_file.read(reinterpret_cast<char*>(rowDelimiters), nRows * sizeof(int));
   input_file.read(reinterpret_cast<char*>(cols), nElements * sizeof(int));

   #pragma omp parallel for
   for(int i =0; i < nElements; i++) {
      cols[i] += 1; // increment the id value
      vals[i] = (double) (i % 13);
   }

   rowDelimiters[nRows] = nElements;
   rowDelimiters[nRows+1] = nElements_padding;

   #pragma omp parallel for
   for(int i =0; i < nRows; i++) {
      for(int j = rowDelimiters[i]; j < rowDelimiters[i+1]; j++)
		rows[j] = i+1;
   }

   for(int i = nElements; i < nElements_padding; i++) {
      cols[i] = cols[nElements-1];
      vals[i] = 0;
   }

    cooMatrix->nnz = nElements_padding; 
    cooMatrix->numRows = nRows; 
    cooMatrix->numCols = nCols; 
    cooMatrix->rows  = rows;
    cooMatrix->cols  = cols;
    cooMatrix->vals  = vals;

    MKL_free(rowDelimiters_ptr);
}

int readCSR(char* matrixName, struct _PAVA_CSRMatrix *csrMatrix) {
    int nRows, nCols, nElements;  
   std::ifstream input_file (matrixName, std::ios::binary);
   if(!input_file.is_open()) {
      std::cout << "cannot open the input csr file!" << '\n';
      exit(1);
   }

   input_file.read(reinterpret_cast<char*>(&nRows), sizeof(int));
   input_file.read(reinterpret_cast<char*>(&nElements), sizeof(int));

   nCols = nRows;
   int nElements_padding = (nElements%16 == 0) ? nElements : (nElements + 16)/ 16 * 16;


  std::cout<<"==========================================================================="<<std::endl;
    std::cout<<"=========*********  Informations of the sparse matrix   *********=========="<<std::endl;
    std::cout<<std::endl;
//    std::cout<<" numRows is "<<nRows<<"; numCols is "<< nCols<<";  Number of Elements is "<< nElements<<";  After Padding, Number of Elements is "<< nElements_padding<<std::endl;
    std::cout<<"     Number of Rows is :"<< nRows<<std::endl;
    std::cout<<"  Number of Columns is :"<< nCols<<std::endl;
    std::cout<<" Number of Elements is :"<< nElements<<std::endl;
    std::cout<<"       After Alignment :"<< nElements_padding<<std::endl;
    std::cout<<std::endl;
  std::cout<<"==========================================================================="<<std::endl;
   
    double* val_ptr = (double*) MKL_malloc(sizeof(double) * nElements_padding, ALIGN512); 
    int *cols_ptr = (int*) MKL_malloc(sizeof(int) * nElements_padding, ALIGN512);
    int *rows_ptr = (int*) MKL_malloc(sizeof(int) * (nRows+1), ALIGN512);

    
    double *vals = val_ptr; 
    int *cols = cols_ptr; 
    int *rows = rows_ptr;

   input_file.read(reinterpret_cast<char*>(rows), nRows * sizeof(int));
   input_file.read(reinterpret_cast<char*>(cols), nElements * sizeof(int));

   #pragma omp parallel for
   for(int i =0; i < nElements; i++) {
      vals[i] = (double) (i % 13);
   }

   rows[nRows] = nElements_padding;
   //rows[nRows+1] = nElements_padding;


   for(int i = nElements; i < nElements_padding; i++) {
      cols[i] = cols[nElements-1];
      vals[i] = 0;
   }

    csrMatrix->nnz = nElements_padding; 
    csrMatrix->numRows = nRows; 
    csrMatrix->numCols = nCols; 
    csrMatrix->rowOffsets  = rows;
    csrMatrix->cols  = cols;
    csrMatrix->vals  = vals;

}


int benchFile(char* fullpath, int thread_idx)
{

    int minSize = 4096;
    int maxSize = INT32_MAX ;   // int max
    

   
    struct _PAVA_CSRMatrix* csrMatrix = (struct _PAVA_CSRMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));

    if ( 0 != readCSR( fullpath, csrMatrix ) )
    {
        fprintf(stderr, "Reading CSR matrix in failed\n" );
        return -2;
    }

    // std::cout<<"Basic informations"<<endl;
    // std::cout<<"        numRows = "<<csrMatrix->numRows<<endl;
    // std::cout<<"        numCols = "<<csrMatrix->numCols<<endl;
    // std::cout<<"            nnz = "<<csrMatrix->nnz<<endl;
    if(csrMatrix->nnz < minSize)
    {
        std::cout<<" FileWarning! "<<fullpath <<" is too small (4k)"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
    else if (csrMatrix->nnz > maxSize)
    {
        std::cout<<" FileWarning! "<<fullpath <<" is too large (2^31))"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
    else if (csrMatrix->numRows < 2048 || csrMatrix->numCols < 2048)
    {
        std::cout<<" FileWarning! "<<fullpath <<" has too few rows or cols"<<endl;
        std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
        return -1;
    }
        
    std::cout<<endl<<"Print out result in [groupName matrixName][format][thread][convertTime][executionTime]"<<endl;

    if(thread_idx ==0)
        for(int iterThreads=68; iterThreads<=272; iterThreads+=68)
        {
            conduct_benchmark(fullpath, iterThreads, csrMatrix);
        }
    else
        conduct_benchmark(fullpath, thread_idx, csrMatrix);

    deleteCSRMatrix( csrMatrix );
    std::cout<<endl<<"Congratulations! This File comes to an Normal End! Flag[NormalEnding]"<<endl<<endl;
    return 0;

}





void benchDir(char *path)  
{  
    DIR              *pDir ;  
    struct dirent    *ent  ;  
    int               i=0  ;  
    char              childpath[512];  
    char              fullpath[512];
  
    pDir=opendir(path);  
    memset(childpath,0,sizeof(childpath)); 

  
    while((ent=readdir(pDir))!=NULL)  
    {  
  
        if(ent->d_type & DT_DIR)
        {  
            if(strcmp(ent->d_name,".")==0 || strcmp(ent->d_name,"..")==0)  
                continue;  
  
            sprintf(childpath,"%s/%s",path,ent->d_name);  
//            printf("path:%s\n",childpath);  
 
            cout<<" "<<ent->d_name;

            benchDir(childpath);  
        }  
        else
        {
            strcpy(fullpath, path);
            strcat(fullpath, "/");
            strcat(fullpath, ent->d_name);

            benchFile(fullpath, 0);

//            cout<<" -- "<<fullpath<<endl;
//            cout<<ent->d_name<<endl;
        }
    }  
  
}  


int main(int argc, char** argv)
{

//    conduct_benchmark(argv[1], 136);
//    return 0;


    cout<<" =================================       now we begin PAVA     ==================================="<<endl;

    if ( argc < 2 )
    {
        fprintf( stderr, "Insufficient number of input parameters:\n");
        fprintf( stderr, "File name with sparse matrix for benchmarking is missing\n" );
        fprintf( stderr, "Usage: %s [data.csr] [threads] [iterations]\n", argv[0] );
        exit(1);
    }

    char* input_path = argv[1];


    struct stat s_buf;
    stat(input_path, &s_buf);

    if(S_ISDIR(s_buf.st_mode))
    {
//        printf(" %s is a dir\n", input_path);
        benchDir(input_path);        
    }

    if(S_ISREG(s_buf.st_mode))
    {
//        printf(" %s is a regular file \n", input_path);
        benchFile(input_path, atoi(argv[2]));
    }


    return 0;

}

int conduct_benchmark(char* fileName, int numThreads, _PAVA_CSRMatrix* fileCSRMatrix)
{
//    char* fileName = argv[1];

    omp_set_num_threads(numThreads);
    int fileLen = strlen(fileName);
//    char* tmpMatrixName = (char*)malloc(sizeof(char) * (fileLen+1));
    char* tmpMatrixName = (char*)malloc(sizeof(char) * 512);
    strcpy(tmpMatrixName, fileName);

    char ch = '/';
    char ch2 = '.';

    if( strrchr(tmpMatrixName,ch2) )
        strrchr(tmpMatrixName,ch2)[0] = '\0';

    if( strrchr(tmpMatrixName,ch) )
        strrchr(tmpMatrixName,ch)[0] = ' ';
        
    char* matrixName;
    if( strrchr(tmpMatrixName,ch) )
        matrixName = strrchr(tmpMatrixName,ch) + 1;
    else
        matrixName = tmpMatrixName;

    
    double *y, *y_ref;
    double *x;

/*
    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));
    if ( 0 != readCOOMatrix( fileName, cooMatrix ) )
    {
        fprintf(stderr, "Reading COO matrix in matrix market format failed\n" );
        return -2;
    }
*/


    // Align allocated memory to boost performance 
#ifdef MMAP
    x     = (double*)mmap(0, fileCSRMatrix->numCols* sizeof(double), PROT_READ|PROT_WRITE,MAP_ANONYMOUS|MAP_PRIVATE|MAP_HUGETLB,-1,0);
#else
    x     = ( double* ) MKL_malloc ( fileCSRMatrix->numCols * sizeof( double ), ALIGN );
#endif
    y     = ( double* ) MKL_malloc ( fileCSRMatrix->numRows * sizeof( double ), ALIGN );
    y_ref = ( double* ) MKL_malloc ( fileCSRMatrix->numRows * sizeof( double ), ALIGN );

    if ( NULL == x || NULL == y || NULL == y_ref )
    {
        fprintf( stderr, "Could not allocate memory for vectors!\n" );
        MKL_free( x );
        MKL_free( y );
        MKL_free( y_ref );

        deleteCSRMatrix( fileCSRMatrix );
        return -1;
    }



    // res is the number of items with error
    int res = 0;
    int numIterations;
    double t1, t2, t3, t4;
    int omp_threads;                              // get the numThreads from env variable of openMP
    int mkl_threads;                              // get the numThreads from env variable of MKL

    double  alpha = 1;
    double beta = 1;

    double normX, normY, residual;
    double estimatedAccuracy;
    double matrixFrobeniusNorm = calcFrobeniusNorm ( fileCSRMatrix->nnz, fileCSRMatrix->vals );
    normX = calcFrobeniusNorm ( fileCSRMatrix->numCols, x );
    normY = calcFrobeniusNorm ( fileCSRMatrix->numRows, y );
    // estimate accuracy of SpMV oparation: y = alpha * A * x + beta * y
    // || y1 - y2 || < eps * ( |alpha| * ||A|| * ||x|| + |beta| * ||y||)
    estimatedAccuracy = fabs( alpha ) * matrixFrobeniusNorm * normX + fabs( beta ) * normY;
    estimatedAccuracy *= EPS;


///////////////////////
//      CSR      
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      CSR     "<<endl;
    std::cout<<"**************"<<endl;

    numIterations = LARGE_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    {

        // first time to init x, y, y_ref
        initVectors( fileCSRMatrix->numRows, fileCSRMatrix->numCols, x, y, y_ref );
        referenceSpMV(fileCSRMatrix, x, y_ref);

        std::cout<<" Executing CSR"<<endl;
        benchmark_CSR_SpMV( fileCSRMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
        res = checkResults(fileCSRMatrix->numRows, y, y_ref);

        t3 = microtime();
        benchmark_CSR_SpMV( fileCSRMatrix, alpha, x, beta, y, y_ref, 0, matrixName, numIterations);
        t4 = microtime();
    //    std::cout<<" The SpMV Execution Time of CSR    is "<< (t4 - t3)/nuIterations<<endl;
       
        printPerformance(matrixName, "CSR", mkl_threads, t2 - t1, (t4 - t3)/numIterations);

        // we leave fileCOOMarix for the next iteration
    //    deleteCOOMatrix( fileCOOMatrix );    
    }

    struct _PAVA_CSRMatrix* csrMatrix = fileCSRMatrix;

///////////////////////
//      COO      
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      COO     "<<endl;
    std::cout<<"**************"<<endl;
    struct _PAVA_COOMatrix* cooMatrix = (struct _PAVA_COOMatrix*) malloc(sizeof(struct _PAVA_COOMatrix));
    numIterations = SMALL_ITERS;

    omp_threads = omp_get_max_threads();
    mkl_set_num_threads_local(omp_threads);
    mkl_threads = mkl_get_max_threads();

    std::cout<<" Converting CSR->COO"<<endl;
    t1 = microtime();
    res = convertCSR2COO(csrMatrix, cooMatrix);
    t2 = microtime();
    if(res!=0)
    {
        std::cout<<" COO Converting Failed"<<std::endl;

        printPerformance(matrixName, "COO", mkl_threads, -1, -1);
        free(cooMatrix);
    }
    // else
    // {
    //     initVectors( cooMatrix->numRows, cooMatrix->numCols, NULL, y, NULL );

    //     std::cout<<" Executing COO"<<endl;
    //     benchmark_COO_SpMV( cooMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
    //     res = checkResults(cooMatrix->numRows, y, y_ref);

    //     t3 = microtime();
    //     benchmark_COO_SpMV( cooMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
    //     t4 = microtime();

    //     printPerformance(matrixName, "COO", mkl_threads, t2-t1, (t4 - t3)/numIterations);

    //     //deleteCOOMatrix(cooMatrix);
    // }
///////////////////////
//      CSC
///////////////////////
//     std::cout<<"**************"<<endl;
//     std::cout<<"      CSC     "<<endl;
//     std::cout<<"**************"<<endl;
//     struct _PAVA_CSCMatrix* cscMatrix = (struct _PAVA_CSCMatrix* ) malloc (sizeof (struct _PAVA_CSCMatrix));
//     numIterations = SMALL_ITERS;

//     omp_threads = omp_get_max_threads();
//     mkl_set_num_threads_local(omp_threads);
//     mkl_threads = mkl_get_max_threads();

//     std::cout<<" Converting CSR->CSC"<<endl;
//     t1 = microtime();
//     res = convertCSR2CSC(csrMatrix, cscMatrix);
//     t2 = microtime();
//     if(res!=0)
//     {
//         std::cout<<" CSC Converting Failed"<<std::endl;

//         printPerformance(matrixName, "CSC", mkl_threads, -1, -1);
//         free(cscMatrix);
//     }
//     else
//     {

//         initVectors( cscMatrix->numRows, cscMatrix->numCols, NULL, y, NULL );
// //        referenceSpMV(csrMatrix, x, y_ref);

//         std::cout<<" Executing CSC"<<endl;
//         benchmark_CSC_SpMV( cscMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
//         res = checkResults(cscMatrix->numRows, y, y_ref);
        
//         initVectors( cscMatrix->numRows, cscMatrix->numCols, x, y, y_ref );
//         referenceSpMV(csrMatrix, x, y_ref);

//         t3 = microtime();
//         benchmark_CSC_SpMV( cscMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
//         t4 = microtime();

//         printPerformance(matrixName, "CSC", mkl_threads, t2 - t1, (t4 - t3)/numIterations);
//         deleteCSCMatrix(cscMatrix);
//     }


///////////////////////
//      DIA
///////////////////////

//     std::cout<<"**************"<<endl;
//     std::cout<<"      DIA     "<<endl;
//     std::cout<<"**************"<<endl;
//     struct _PAVA_DIAMatrix* diaMatrix = (struct _PAVA_DIAMatrix* ) malloc (sizeof (struct _PAVA_DIAMatrix));
//     numIterations = SMALL_ITERS;

//     omp_threads = omp_get_max_threads();
//     mkl_set_num_threads_local(omp_threads);
//     mkl_threads = mkl_get_max_threads();

//     std::cout<<" Converting CSR->DIA"<<endl;
//     t1 = microtime();
//     res = convertCSR2DIA(csrMatrix, diaMatrix);
//     t2 = microtime();
//     if(res!=0)
//     {
//         std::cout<<" DIA Converting Failed"<<std::endl;

//         printPerformance(matrixName, "DIA", mkl_threads, -1, -1);
//         free(diaMatrix);
//     }
//     else
//     {
//         initVectors( diaMatrix->numRows, diaMatrix->numCols, NULL, y, NULL );
// //        referenceSpMV(csrMatrix, x, y_ref);

//         std::cout<<" Executing DIA"<<endl;
//         benchmark_DIA_SpMV( diaMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 1);
//         res = checkResults(csrMatrix->numRows, y, y_ref);

//         t3 = microtime();
//         benchmark_DIA_SpMV( diaMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);
//         t4 = microtime();

//         printPerformance(matrixName, "DIA", mkl_threads, t2 - t1, (t4 - t3)/numIterations);
//         deleteDIAMatrix(diaMatrix);
//     }
///////////////////////
//      IE
///////////////////////
//     std::cout<<"**************"<<endl;
//     std::cout<<"      IE      "<<endl;
//     std::cout<<"**************"<<endl;
//     numIterations = LARGE_ITERS;

//     initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

//     benchmark_IE_SpMV( csrMatrix, alpha, x, beta, y, y_ref, 0, matrixName, 10);

// //    printPerformance(matrixName, "IE", mkl_threads, -1, -1);
//     res = checkResults(csrMatrix->numRows, y, y_ref);

///////////////////////
//      BSR
///////////////////////

//     std::cout<<"**************"<<endl;
//     std::cout<<"      BSR      "<<endl;
//     std::cout<<"**************"<<endl;
//     struct _PAVA_BSRMatrix* bsrMatrix = (struct _PAVA_BSRMatrix* ) malloc (sizeof (struct _PAVA_BSRMatrix));
//     numIterations = SMALL_ITERS;
    
//     omp_threads = omp_get_max_threads();
//     mkl_set_num_threads_local(omp_threads);
//     mkl_threads = mkl_get_max_threads();

//     std::cout<<" Converting CSR->BSR"<<endl;
//     t1 = microtime();
//     res = convertCSR2BSR(csrMatrix, bsrMatrix, 4);
//     t2 = microtime();

//     if(res!=0)
//     {
//         std::cout<<" BSR Converting Failed"<<std::endl;

//         printPerformance(matrixName, "BSR", mkl_threads, -1, -1);
//         free(bsrMatrix);
//     }
//     else
//     {
//         initVectors( bsrMatrix->numRows, bsrMatrix->numCols, NULL, y, NULL );
// //        referenceSpMV(csrMatrix, x, y_ref);
        
//         std::cout<<" Executing BSR"<<endl;

//         benchmark_BSR_SpMV( bsrMatrix, alpha, x, beta, bsrMatrix->y_bsr, y_ref, 0, matrixName, 1);
//         res = checkResults(csrMatrix->numRows, bsrMatrix->y_bsr, y_ref);

//         t3 = microtime();
//         benchmark_BSR_SpMV( bsrMatrix, alpha, x, beta, bsrMatrix->y_bsr, y_ref, 0, matrixName, 10);
//         t4 = microtime();

//         printPerformance(matrixName, "BSR", mkl_threads, t2 - t1, (t4 - t3)/numIterations);

//         deleteBSRMatrix(bsrMatrix);

//     }



///////////////////////
//      ESB
///////////////////////
    
    std::cout<<"**************"<<endl;
    std::cout<<"      ESB      "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

    benchmark_ESB_SpMV(csrMatrix, alpha, x, beta, y, NULL, 0.0, 0.0, INTEL_SPARSE_SCHEDULE_DYNAMIC, matrixName, numIterations);
    res = checkResults(csrMatrix->numRows, y, y_ref);

    benchmark_ESB_SpMV(csrMatrix, alpha, x, beta, y, NULL, 0.0, 0.0, INTEL_SPARSE_SCHEDULE_STATIC, matrixName, numIterations);
    res = checkResults(csrMatrix->numRows, y, y_ref);

//    printPerformance(matrixName, "ESB", mkl_threads, -1, -1);
///////////////////////////////////////
///////////////////////
//      CVR
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"      CVR      "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;
    if(csrMatrix->numRows > 4096 && csrMatrix->numCols > 4096)
    {

        initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

        conduct_cvr(csrMatrix->numRows, csrMatrix->numCols, csrMatrix->nnz, csrMatrix->rowOffsets, csrMatrix->cols, csrMatrix->vals, x, y, alpha, matrixName, numIterations);

        res = checkResults(csrMatrix->numRows, y, y_ref);
    }
    else
    {
        omp_threads = omp_get_max_threads();
        printPerformance(matrixName, "CVR", omp_threads, -1,-1);
    }

////////////////////////////////////////
///////////////////////
//      CSR5
///////////////////////

    std::cout<<"**************"<<endl;
    std::cout<<"     CSR5     "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

    initVectors( csrMatrix->numRows, csrMatrix->numCols, NULL, y, NULL );

    conduct_csr5(csrMatrix->numRows, csrMatrix->numCols, csrMatrix->nnz, csrMatrix->rowOffsets, csrMatrix->cols, csrMatrix->vals, x, y, alpha, matrixName, numIterations);

    res = checkResults(csrMatrix->numRows, y, y_ref);

////////////////////////////////////////
///////////////////////
//      VHCC
///////////////////////
    std::cout<<"**************"<<endl;
    std::cout<<"     VHCC     "<<endl;
    std::cout<<"**************"<<endl;
    numIterations = LARGE_ITERS;

    initVectors( cooMatrix->numRows, cooMatrix->numCols, NULL, y, NULL );

    conduct_vhcc(cooMatrix->numRows, cooMatrix->numCols, cooMatrix->nnz, cooMatrix->rows, cooMatrix->cols, cooMatrix->vals, x, y, alpha, matrixName, numIterations);
//
//    printPerformance(matrixName, "VHCC", omp_threads, -1, -1);

    res = checkResults(cooMatrix->numRows, y, y_ref);
    
////////////////////////////////////////
    MKL_free( x );
    MKL_free( y );
    MKL_free( y_ref );
    deleteCOOMatrix(cooMatrix);
    free(tmpMatrixName);
    std::cout<<" x.x.x.x.x.x.x.x.x.x.x.x"<<endl<<endl;

    return 0;
}




