
#include "mem.h"
//#include "mmio.h"
#include "timer.h"
#include <cstring>
#include <string>
#include <cmath>
#include <cstdint>
#include <getopt.h>
#include "omp.h"
#include "immintrin.h"
#include "mkl.h"
#include "SparseMatrixReader.h"
//#include "compute.h"

#include <unistd.h>
#include <sys/mman.h>


#include "vhcc_matrix.h"

#include "../utils/mmio.h"
#include "../utils/basicMatrix.h"



/*
#include <time.h>
#include <sys/time.h>
#define MICRO_IN_SEC 1000000.00

double microtime(){
        int tv_sec,tv_usec;
        double time;
        struct timeval tv;
        struct timezone tz;
        gettimeofday(&tv,&tz);

        return tv.tv_sec+tv.tv_usec/MICRO_IN_SEC;
}

*/

typedef int index_t;
typedef double value_t;

#define LEN16 16
#define LEN8 8


typedef __attribute__((aligned(64))) union zmmi {
	__m512i reg;
	unsigned int elems[LEN16];
} zmmi_t;
/*
typedef __attribute__((aligned(64))) union zmmd {
	__m512d reg;
	__m512i regi32;
	double elems[LEN8];
} zmmd_t;
*/

int count_trailing_zero(int a, __mmask8 x)
{
   int idx = a+1;
   __mmask8 mask[8] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
   while((x & mask[idx]) == 0)
   {
      idx ++;
   }
  
   return idx;
   
}


void compute_spmv(int n_threads, int num_vectors,
									int threads_per_core,
									int num_panels,
									panel_info_t * panel_info,
									thr_info_t   * thr_info,
									index_t * veceor_ptr,
									uint8_t * scan_mask,
									index_t * row_arr,
									index_t * col_arr,
									value_t * vals_arr,
									value_t * input,
									value_t * result)
{
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();

		int core_id = id / threads_per_core;
		int local_thr_id = id % threads_per_core;
		
		int panel_id = thr_info[id].panel_id;
		
		value_t *tmp_result = panel_info[panel_id].tmp_result;
		
		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;
		
		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;
		
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].sbase;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;
		
		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];
			
			__assume_aligned(&vals_arr[vec_idx], 64);
			
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * 
				input[col.elems[0:LEN8]];
			vec_idx += LEN8;
			
			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));
				
				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
//				int a = -1;
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
//					int y = _mm_tzcnti_32(a, x);
					int y = count_trailing_zero(a,maskwr);
					index_t r = row_arr[ridx+i];
					tmp_result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {

				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				nrval = res.elems[LEN8-1] + res.elems[3];
			}
			
			res.elems[:] = 0;
			res.elems[0] = nrval;
		}
		
#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = tmp_result[thr_info[id].overflow_row];

#pragma omp atomic update
		tmp_result[nridx] += nrval;
	
#pragma omp barrier
		
		index_t merge_start = thr_info[id].merge_start;
		index_t merge_end   = thr_info[id].merge_end;
		index_t blk_size    = 512;
		
		for (index_t i = merge_start; i < merge_end; i += blk_size) {
			index_t blk_end = i + blk_size > merge_end ? merge_end : i + blk_size;
			for (int c = 0; c < num_panels; ++c) {
				for (index_t b = i; b < blk_end; b += LEN8) {
					result[b:LEN8] += panel_info[c].tmp_result[b:LEN8];
				}
			}
		}
	}
}




void compute_spmv1(int n_threads, int num_vectors,
									 thr_info_t * thr_info,
									 index_t * veceor_ptr,
									 uint8_t * scan_mask,
									 index_t * row_arr,
									 index_t * col_arr,
									 value_t * vals_arr,
									 value_t * input,
									 value_t * result)
{
#pragma omp parallel default(shared) num_threads(n_threads)
	{

		int id = omp_get_thread_num();

		index_t start_vec = thr_info[id].start_vec;
		index_t end_vec   = thr_info[id].end_vec;

		zmmi_t row, col, wrmask;
		zmmd_t res, tmp;
		__mmask8 mask1, mask2, mask3, maskwr;

		index_t cidx       = thr_info[id].vbase;
		index_t veceor_idx = thr_info[id].vbase;
		index_t scan_idx   = thr_info[id].vbase * 4;
		index_t ridx       = thr_info[id].rbase;
		index_t vec_idx    = start_vec * LEN8;

		value_t nrval = 0;
		index_t eor_vec = veceor_ptr[veceor_idx++];
		res.elems[:] = 0;
                 
//                std::cout<<" start = "<< start_vec <<";  end = "<< end_vec<<endl;

		for (index_t v = start_vec; v < end_vec; ++v) {
			
			col.elems[0:LEN8] = col_arr[vec_idx:LEN8];

			__assume_aligned(&vals_arr[vec_idx], 64);
			res.elems[0:LEN8] += vals_arr[vec_idx:LEN8] * input[col.elems[0:LEN8]];
			vec_idx += LEN8;

			nrval = 0;
			if (v == eor_vec) {
				mask1 = (__mmask8)scan_mask[scan_idx++];
				mask2 = (__mmask8)scan_mask[scan_idx++];
				mask3 = (__mmask8)scan_mask[scan_idx++];
				maskwr = (__mmask8)scan_mask[scan_idx++];
				
				res.reg = _mm512_mask_add_pd(res.reg, mask1, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
				res.reg = _mm512_mask_add_pd(res.reg, mask2, res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
				tmp.regi32 = _mm512_permute4f128_epi32(res.regi32, _MM_PERM_BBBA);
				res.reg = _mm512_mask_add_pd(res.reg, mask3, res.reg, _mm512_swizzle_pd(tmp.reg, _MM_SWIZ_REG_BBBB));

				if ((maskwr & 0x80) == 0)
					nrval = res.elems[LEN8-1];

				int bcnt = _mm_countbits_32(maskwr);
//				int a = -1;
				int a = -1;
				int x = maskwr;
				for (int i = 0; i < bcnt; ++i) {
//					int y = _mm_tzcnti_32(a, x);
					int y = count_trailing_zero(a,maskwr);
//                                            std::cout<<"bcnt = "<< bcnt<<"; y = "<< y<<"; v= "<< v<<"; start = "<< start_vec<<"; end = "<< end_vec<<endl;      

					index_t r = row_arr[ridx+i];
					result[r] += res.elems[y];
					a = y;
				}
				ridx += bcnt;

				eor_vec = veceor_ptr[veceor_idx++];
				
			} else {
				
//				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_CDAB));
//				res.reg = _mm512_add_pd(res.reg, _mm512_swizzle_pd(res.reg, _MM_SWIZ_REG_BBBB));
//				nrval = res.elems[LEN8-1] + res.elems[3];
                                nrval = _mm512_reduce_add_pd(res.reg);				
			}

			res.elems[:] = 0;
			res.elems[0] = nrval;
		}

#pragma omp barrier

		index_t nridx = thr_info[id].last_row;
		nrval = result[thr_info[id].overflow_row];
#pragma omp atomic update
		result[nridx] += nrval;
//        assert(nridx<36683);
	}
}


__attribute__((noinline)) int run_spmv_vhcc1(int n_threads, int num_vectors,
									 thr_info_t * thr_info,
									 index_t * veceor_ptr,
									 uint8_t * scan_mask,
									 index_t * row_arr,
									 index_t * col_arr,
									 value_t * vals_arr,
									 value_t * input,
									 value_t * result,
                                     int numIterations)
{
     for (int i = 0; i < numIterations; ++i) {
          compute_spmv1(n_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
     }

   return 0;

}
__attribute__((noinline)) void run_spmv_vhcc(int n_threads, int num_vectors,
									int threads_per_core,
									int num_panels,
									panel_info_t * panel_info,
									thr_info_t   * thr_info,
									index_t * veceor_ptr,
									uint8_t * scan_mask,
									index_t * row_arr,
									index_t * col_arr,
									value_t * vals_arr,
									value_t * input,
									value_t * result,
                                    int numIterations)
{


		for (int i = 0; i < numIterations; ++i) {
			compute_spmv(n_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}

}


/*
int main4(int argc, char *argv[])
{
        printf(" in main");
	int c = 0;
	char *matrixName = NULL;
//	int num_threads = 240;
	int num_threads;
//	int tile_row = -1;
//	int tile_col = -1;
        int tile_row = 512;
        int tile_col = 8192;	
	int num_panel = 1;
	int threads_per_core = MAX_THREADS_PER_CORE;
	bool set_iter = false;
	bool use_binary_input = false;
        printf(" before reading option");

        matrixName = argv[1];
	if (matrixName == NULL) {
		printf("Error: specify matrix matrixName.\n");
		exit(1);
	}
	if (num_panel > 60) {
		printf("Error. Expect number of panels to be between 1 and 60 inclusive\n");
		exit(1);
	}

        num_threads = atoi(argv[2]);
        omp_set_num_threads(num_threads);
        if(num_threads == 0)
           return 0;
        threads_per_core = num_threads / 68;
        num_panel = atoi(argv[4]);
	
	kmp_set_defaults("KMP_AFFINITY=compact");

	int m, n, nnz;
	int *row_idx, *col_idx;
	double *tvals;
 
        printf(" before reading matrix\n");
	if(use_binary_input) {
		if(!SparseMatrixReader::ReadEncodedData(matrixName, &m, &n, &nnz, &row_idx, &col_idx, &tvals)) {
			return EXIT_FAILURE;
		}
	} else {
                printf(" before reading matrix  2");
		if (!SparseMatrixReader::ReadRawData(matrixName, &m, &n, &nnz, &row_idx, &col_idx, &tvals)) {
			printf("Reading input matrix fails!\n");
			return -1;
		} 
	}

  value_t *vals = (value_t *)MALLOC(nnz * sizeof(value_t));
	for (index_t i = 0; i < nnz; ++i)
		vals[i] = tvals[i];

//    _mm_free(tvals);


}

*/

void conduct_vhcc(int m, int n, int nnz, int* row_idx, int* col_idx, value_t* vals, value_t* input, value_t* result, int alpha, char* matrixName, int numIterations )
{
        int tile_row = 512;
        int tile_col = 8192;	

        if(m < 2048)
            tile_row = 128;
        if(n < 8192)
            tile_col = 128;




	int num_panel = 1;

  int num_threads = omp_get_max_threads();
	int threads_per_core = 1;



    int max = m > n ? m : n;

    int    sizeM = max;
    int    sizeN = max;

//    sizeM = (sizeM+15)/16*16;
//    sizeN = (sizeN+15)/16*16;



    // return 'result' and check the correctness; use 'result_iter' for iterative spmv

//  kmp_set_defaults("KMP_AFFINITY=compact");
//  std::cout<<" start vhcc"<<endl;

  vhcc_matrix<index_t, value_t> mat(sizeM, sizeN, nnz, row_idx, col_idx, vals);

//  std::cout<<" start convert"<<endl;

  std::cout<<" Converting COO->VHCC"<<endl;
  double t1 = microtime();
  mat.convert(num_threads, threads_per_core, num_panel, tile_row, tile_col, matrixName);
  double t2 = microtime();

//        std::cout<<" after convert"<<endl;





  index_t n_rows = m;
	index_t n_cols = n;
	index_t padrows            = mat.get_pad_rows();
	index_t padcols            = mat.get_pad_cols();
	index_t num_vectors        = mat.get_num_vectors();
	int         num_panels     = mat.get_num_panels();
	panel_info_t *panel_info   = mat.get_panel_info();
	thr_info_t *thr_info       = mat.get_thr_info();
	index_t     thr_info_size  = mat.get_thr_info_size();
	index_t    *veceor_ptr     = mat.get_veceor_ptr();
	index_t     veceor_size    = mat.get_veceor_size();
	uint8_t    *scan_mask      = mat.get_scan_mask();
	index_t     scan_mask_size = mat.get_scan_mask_size();
	index_t    *row_arr        = mat.get_row_arr();
	index_t     row_arr_size   = mat.get_row_arr_size();
	index_t    *col_arr        = mat.get_col_arr();
	index_t     col_arr_size   = mat.get_col_arr_size();
	value_t    *vals_arr       = mat.get_vals_arr();
	index_t     vals_arr_size  = mat.get_vals_arr_size();


	cputimer timer;
	double elapsed;
	double mytime;


    value_t* result_iter= ( value_t* )MALLOC( padrows * sizeof(value_t));
    memset(result_iter, 0, padrows*sizeof(value_t));




//	run_spmv_vhcc1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result, numIterations);

    std::cout<<" Executing VHCC"<<endl;

		run_spmv_vhcc1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result_iter, 1);

#pragma omp parallel for
        for(int ttsk=0; ttsk < m; ttsk++)
            result[ttsk] = result_iter[ttsk];

        double t3, t4;
        t3 = microtime();
		run_spmv_vhcc1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result_iter, numIterations);
        t4 = microtime();

        if(0)
	if (num_panels > 1) {
/*
		// warm up
		for (int i = 0; i < 1; ++i) {
			compute_spmv(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
*/
//		for (int i = 0; i < padrows; ++i) result[i] = 0;
		for (int i = 0; i < num_panels; ++i) {
			memset(panel_info[i].tmp_result, 0, padrows * sizeof(value_t));
		}

	    run_spmv_vhcc(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result, 1);

//		double tstart = dsecnd();
                t3 = microtime();
/*
		for (int i = 0; i < numIterations; ++i) {
			compute_spmv(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
*/

	    run_spmv_vhcc(num_threads, num_vectors, threads_per_core, num_panels, panel_info, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result_iter, numIterations);
//                t2 = microtime();
        t4 = microtime();    

//		elapsed = (dsecnd() - tstart) * 1000;
                
	} else {
/*
//                std::cout<<" now in else"<<endl;
		// warm up
		for (int i = 0; i < 1; ++i) {
			compute_spmv1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
*/
//		for (int i = 0; i < padrows; ++i) result[i] = 0;
		for (int i = 0; i < num_panels; ++i) {
			memset(panel_info[i].tmp_result, 0, padrows * sizeof(value_t));
		}

		run_spmv_vhcc1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result, 1);
//		double tstart = dsecnd();
        t3 = microtime();
/*
		for (int i = 0; i < numIterations; ++i) {
			compute_spmv1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result);
		}
*/
		run_spmv_vhcc1(num_threads, num_vectors, thr_info, veceor_ptr, scan_mask, row_arr, col_arr, vals_arr, input, result_iter, numIterations);
                
//                t2 = microtime();
        t4 = microtime();

//		elapsed = (dsecnd() - tstart) * 1000;
	}


    printPerformance(matrixName, "VHCC", num_threads, t2-t1,(t4 - t3)/numIterations);

    FREE(result_iter);



/*
        for(int i=0; i<256; i++)
           if(i%16==0)
              std::cout<<" result["<<i<<"] = "<<result[i]<<endl;
*/
//        for(int i=m-256; i<m; i++)
//           if(i%16==0)
//              std::cout<<" result["<<i<<"] = "<<result[i]<<endl;

//        printf(" My time is %f\n", (t2-t1)/numIterations);
//        printf("VHCC EXE Time %s %f\n", matrixName, (mytime)/numIterations);
//        printf("The SpMV Execution Time of VHCC is %f seconds\n", (mytime)/numIterations);
//	elapsed = elapsed / numIterations;
//	printf("Gflops: %f, Time: %f\n", double(nnz)/(mytime/numIterations*1000)/1000000, mytime/numIterations);

//	FREE(input);
//	FREE(result);

}
  
