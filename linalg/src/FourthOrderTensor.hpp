/*

Copyright (c) 2005-2022, University of Oxford.
All rights reserved.

University of Oxford means the Chancellor, Masters and Scholars of the
University of Oxford, having an administrative office at Wellington
Square, Oxford OX1 2JD, UK.

This file is part of Chaste.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of the University of Oxford nor the names of its
   contributors may be used to endorse or promote products derived from this
   software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef _FOURTHORDERTENSOR_HPP_
#define _FOURTHORDERTENSOR_HPP_

#include <cassert>
#include <vector>
#include <memory_resource>
#include <cstring>
#include <immintrin.h>

#include "UblasIncludes.hpp"
#include "Exception.hpp"

#define AVX_WIDTH1      4

/**
 * FourthOrderTensor
 *
 * A class of fourth order tensors (i.e. tensors with four indices), over arbitrary dimension.
 */
template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
class FourthOrderTensor
{
private:
  /**
   * Round size up to AVX vector register width
   */
  static constexpr unsigned newDim1()
  {
    unsigned ret = DIM1;
    unsigned rem = DIM1 % AVX_WIDTH1;
    if( rem )
      ret += AVX_WIDTH1 / rem;
    return ret;
  }
  /**
   * calc memory to allocate
   */
  static constexpr unsigned makeAvxWidth()
  {
    return newDim1() * DIM2 * DIM3 * DIM4 * sizeof(double);
  }

  /**
   * replace heap memory with stack memory.
   * round DIM1 up to AVX vector width multiple
   */
  char		 	data_[ makeAvxWidth() ] __attribute__ ((aligned (64)));
  std::pmr::monotonic_buffer_resource
  			rsrc_; //( data_, sizeof data_);
  typedef std::pmr::vector<double> Vec;
  
  Vec 	mData;  /**< The components of the tensor. */

    /** @return the index into the mData vector corresponding to this set of indices
      * @param M  first index
      * @param N  second index
      * @param P  third index
      * @param Q  fourth index
      */
    unsigned GetVectorIndex(unsigned M, unsigned N, unsigned P, unsigned Q)
    {
      assert(M< newDim1() ); //DIM1);
        assert(M<DIM1);
        assert(N<DIM2);
        assert(P<DIM3);
        assert(Q<DIM4);
        return M + DIM1*N + DIM1*DIM2*P + DIM1*DIM2*DIM3*Q;
    }

public:

    /**
     * Constructor.
     */
    FourthOrderTensor();

    /**
     * Set to be the inner product of a matrix another fourth order tensor, contracting on first component,
     * i.e. sets this tensor to be R, where
     * R_{abcd} = X_{aN} T_{Nbcd}
     *
     * @param rMatrix A matrix
     * @param rTensor A fourth order tensor
     */
    template<unsigned CONTRACTED_DIM>
    void SetAsContractionOnFirstDimension(const c_matrix<double,DIM1,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<CONTRACTED_DIM,DIM2,DIM3,DIM4>& rTensor);


    /**
     * Set to be the inner product of a matrix another fourth order tensor, contracting on second component,
     * i.e. sets this tensor to be R, where
     * R_{abcd} = X_{bN} T_{aNcd}
     *
     * @param rMatrix A matrix
     * @param rTensor A fourth order tensor
     */
    template<unsigned CONTRACTED_DIM>
    void SetAsContractionOnSecondDimension(const c_matrix<double,DIM2,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,CONTRACTED_DIM,DIM3,DIM4>& rTensor);

    /**
     * Set to be the inner product of a matrix another fourth order tensor, contracting on third component,
     * i.e. sets this tensor to be R, where
     * R_{abcd} = X_{cN} T_{abNd}
     *
     * @param rMatrix A matrix
     * @param rTensor A fourth order tensor
     */
    template<unsigned CONTRACTED_DIM>
    void SetAsContractionOnThirdDimension(const c_matrix<double,DIM3,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,DIM2,CONTRACTED_DIM,DIM4>& rTensor);

    /**
     * Set to be the inner product of a matrix another fourth order tensor, contracting on fourth component,
     * i.e. sets this tensor to be R, where
     * R_{abcd} = X_{dN} T_{abcN}
     *
     * @param rMatrix A matrix
     * @param rTensor A fourth order tensor
     */
    template<unsigned CONTRACTED_DIM>
    void SetAsContractionOnFourthDimension(const c_matrix<double,DIM4,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,DIM2,DIM3,CONTRACTED_DIM>& rTensor);

    /**
     * @return the MNPQ-component of the tensor.
     *
     * @param M  first index
     * @param N  second index
     * @param P  third index
     * @param Q  fourth index
     */
    double& operator()(unsigned M, unsigned N, unsigned P, unsigned Q);

    /**
     * Set all components of the tensor to zero.
     */
    void Zero();

    /**
     * @return a reference to the internal data of the tensor.
     */
    Vec& rGetData()
    {
        return mData;
    }
  constexpr unsigned size()
  {
    return DIM1*DIM2*DIM3*DIM4;
  }
  
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation (lots of possibilities for the dimensions so no point with explicit instantiation)
///////////////////////////////////////////////////////////////////////////////////////////////////

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::FourthOrderTensor()
  : rsrc_( data_, sizeof data_ )
  , mData( size(), 0.0, &rsrc_ )
{
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
template<unsigned CONTRACTED_DIM>
void FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::SetAsContractionOnFirstDimension(const c_matrix<double,DIM1,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<CONTRACTED_DIM,DIM2,DIM3,DIM4>& rTensor)
{
    Zero();

    Vec::iterator iter = mData.begin();
    Vec::iterator other_tensor_iter = rTensor.rGetData().begin();

    for (unsigned d=0; d<DIM4; d++)
    {
        for (unsigned c=0; c<DIM3; c++)
        {
            for (unsigned b=0; b<DIM2; b++)
            {
                for (unsigned a=0; a<DIM1; a++)
                {
                    for (unsigned N=0; N<CONTRACTED_DIM; N++)
                    {
                        /*
                         * The following just does
                         *
                         * mData[GetVectorIndex(a,b,c,d)] += rMatrix(a,N) * rTensor(N,b,c,d);
                         *
                         * but more efficiently using iterators into the data vector, not
                         * using random access.
                         */
                        *iter += rMatrix(a,N) * *other_tensor_iter;
                        other_tensor_iter++;
                    }

                    iter++;

                    if (a != DIM1-1)
                    {
                        other_tensor_iter -= CONTRACTED_DIM;
                    }
                }
            }
        }
    }
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
template<unsigned CONTRACTED_DIM>
void FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::SetAsContractionOnSecondDimension(const c_matrix<double,DIM2,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,CONTRACTED_DIM,DIM3,DIM4>& rTensor)
{
    Zero();

    Vec::iterator iter = mData.begin();
    Vec::iterator other_tensor_iter = rTensor.rGetData().begin();

    for (unsigned d=0; d<DIM4; d++)
    {
        for (unsigned c=0; c<DIM3; c++)
        {
            for (unsigned b=0; b<DIM2; b++)
            {
                for (unsigned N=0; N<CONTRACTED_DIM; N++)
                {
                    for (unsigned a=0; a<DIM1; a++)
                    {
                        /*
                         * The following just does
                         *
                         * mData[GetVectorIndex(a,b,c,d)] += rMatrix(b,N) * rTensor(a,N,c,d);
                         *
                         * but more efficiently using iterators into the data vector, not
                         * using random access.
                         */
                        *iter += rMatrix(b,N) * *other_tensor_iter;
                        iter++;
                        other_tensor_iter++;
                    }

                    if (N != CONTRACTED_DIM-1)
                    {
                        iter -= DIM1;
                    }
                }
                if (b != DIM2-1)
                {
                    other_tensor_iter -= CONTRACTED_DIM*DIM1;
                }
            }
        }
    }
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
template<unsigned CONTRACTED_DIM>
void FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::SetAsContractionOnThirdDimension(const c_matrix<double,DIM3,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,DIM2,CONTRACTED_DIM,DIM4>& rTensor)
{
    Zero();

    Vec::iterator iter = mData.begin();
    Vec::iterator other_tensor_iter = rTensor.rGetData().begin();

    for (unsigned d=0; d<DIM4; d++)
    {
        for (unsigned c=0; c<DIM3; c++)
        {
            for (unsigned N=0; N<CONTRACTED_DIM; N++)
            {
                for (unsigned b=0; b<DIM2; b++)
                {
                    for (unsigned a=0; a<DIM1; a++)
                    {
                        /*
                         * The following just does
                         *
                         * mData[GetVectorIndex(a,b,c,d)] += rMatrix(c,N) * rTensor(a,b,N,d);
                         *
                         * but more efficiently using iterators into the data vector, not
                         * using random access.
                         */
                        *iter += rMatrix(c,N) * *other_tensor_iter;
                        iter++;
                        other_tensor_iter++;
                    }
                }

                if (N != CONTRACTED_DIM-1)
                {
                    iter -= DIM1*DIM2;
                }
            }

            if (c != DIM3-1)
            {
                other_tensor_iter -= CONTRACTED_DIM*DIM1*DIM2;
            }
        }
    }
}
template<> //unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
template<unsigned CONTRACTED_DIM>
void FourthOrderTensor<10, 3, 10, 3>::SetAsContractionOnThirdDimension(const c_matrix<double, 10, CONTRACTED_DIM>& rMatrix, FourthOrderTensor<10, 3, CONTRACTED_DIM, 3>& rTensor)
{
    Zero();
    // __m256d _mm256_setzero_pd(void)
    //    data_[0:4] += rTensor[0:4];
    
    Vec::iterator iter = mData.begin();
    Vec::iterator other_tensor_iter = rTensor.rGetData().begin();

    for (unsigned d=0; d < 3 /*DIM4*/; d++)
    {
      for (unsigned c=0; c < 10 /*DIM3*/; c++)
        {
	  for (unsigned N=0; N < CONTRACTED_DIM; N++)
            {
	      for (unsigned b=0; b < 3 /*DIM2*/; b++)
                {
		  //__m256d rM=_m256d_setr_pd( rMatrix( c, N ), rMatrix( c, N ), rMatrix( c, N ), rMatrix( c, N ) );
		  __m256d rM = _mm256_set1_pd( rMatrix( c, N ) );

		  double* p = &(*iter), *p2 = &(*other_tensor_iter);
		  for( unsigned a = 0; a < 12; a+=4, p += 4, p2 += 4 )
		    {
		      //		      std::cout << a << ',' << b << ',' << c << ',' << d << " += (c,N)" << c << ',' << N << " * (a,b,N,d)" << a << ',' << b << ',' << N << ',' << d << '\n';

		      // __m256d mD = _mm256_load_pd( p 		); // *iter
		      // __m256d rT = _mm256_load_pd( p2  	); // *other_tensor_iter

                      // mData[GetVectorIndex(a,b,c,d)] += rMatrix(c,N) * rTensor(a,b,N,d);
		      // iter += rMatrix(c, N) * other_tensor_iter
		      // mD = mD + rM * rT
		      
      		      //mD = _mm256_add_pd(mD,_mm256_mul_pd( rM, rT ) );
     		      // _mm256_store_pd( p, mD );
		      
		      _mm256_store_pd( p,_mm256_add_pd(_mm256_load_pd(p),_mm256_mul_pd( rM, _mm256_load_pd(p2))));
		    }
		  iter += 12; other_tensor_iter += 12;

		  // for (unsigned a=0; a < 12 /*DIM1*/; a++)
                  //   {
		  //     //      		      std::cout << " a" << a;
		  //     std::cout << a << ',' << b << ',' << c << ',' << d << " += (c,N)" << c << ',' << N << " * (a,b,N,d)" << a << ',' << b << ',' << N << ',' << d << '\n';
                  //       /*
                  //        * The following just does
                  //        *
                  //        * mData[GetVectorIndex(a,b,c,d)] += rMatrix(c,N) * rTensor(a,b,N,d);
                  //        *
                  //        * but more efficiently using iterators into the data vector, not
                  //        * using random access.
                  //        */
                  //       *iter += rMatrix(c,N) * *other_tensor_iter;
                  //       iter++;
                  //       other_tensor_iter++;
                  //   }
		  //		  std::cout << "==================\n";
                }
                if (N != CONTRACTED_DIM-1)
                {
		  iter -= 12 * 3; //DIM1*DIM2;
                }
            }

            if (c != 10 - 1 ) //DIM3-1)
            {
	      other_tensor_iter -= CONTRACTED_DIM * 12 * 3; //DIM1 * DIM2;
            }
        }
    }
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
template<unsigned CONTRACTED_DIM>
void FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::SetAsContractionOnFourthDimension(const c_matrix<double,DIM4,CONTRACTED_DIM>& rMatrix, FourthOrderTensor<DIM1,DIM2,DIM3,CONTRACTED_DIM>& rTensor)
{
    Zero();

    Vec::iterator iter = mData.begin();
    Vec::iterator other_tensor_iter = rTensor.rGetData().begin();

    for (unsigned d=0; d<DIM4; d++)
    {
        for (unsigned N=0; N<CONTRACTED_DIM; N++)
        {
            for (unsigned c=0; c<DIM3; c++)
            {
                for (unsigned b=0; b<DIM2; b++)
                {
                    for (unsigned a=0; a<DIM1; a++)
                    {
                        /*
                         * The following just does
                         *
                         * mData[GetVectorIndex(a,b,c,d)] += rMatrix(d,N) * rTensor(a,b,c,N);
                         *
                         * but more efficiently using iterators into the data vector, not
                         * using random access.
                         */
                        *iter += rMatrix(d,N) * *other_tensor_iter;

                        iter++;
                        other_tensor_iter++;
                    }
                }
            }

            if (N != CONTRACTED_DIM-1)
            {
                iter-= DIM1*DIM2*DIM3;
            }
        }

        other_tensor_iter -= CONTRACTED_DIM*DIM1*DIM2*DIM3;
    }
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
double& FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::operator()(unsigned M, unsigned N, unsigned P, unsigned Q)
{
    assert(M<DIM1);
    assert(N<DIM2);
    assert(P<DIM3);
    assert(Q<DIM4);

    return mData[GetVectorIndex(M,N,P,Q)];
}

template<unsigned DIM1, unsigned DIM2, unsigned DIM3, unsigned DIM4>
void FourthOrderTensor<DIM1,DIM2,DIM3,DIM4>::Zero()
{
  std::memset( &data_[0], 0x00, sizeof data_ ); //DIM1 * DIM2 * DIM3 * DIM4 * sizeof(double) );
    // for (unsigned i=0; i<mData.size(); i++)
    // {
    //     mData[i] = 0.0;
    // }
}

#endif //_FOURTHORDERTENSOR_HPP_
