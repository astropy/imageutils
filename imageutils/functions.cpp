/*
 * functions.c
 *
 *  Created on: May 27, 2011
 *      Author: cmccully
 */
using namespace std;
//#include<malloc/malloc.h>
#include<stdlib.h>
#include<iostream>
#include<math.h>
#include<string.h>
#include<stdio.h>
#include "functions.h"

#define ELEM_SWAP(a,b) { float t=(a);(a)=(b);(b)=t; }

float median(float* a, int n) {
	/*
	 *  This Quickselect routine is based on the algorithm described in
	 *  "Numerical recipes in C", Second Edition,
	 *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
	 *  This code by Nicolas Devillard - 1998. Public domain.
	 */

	//Make a copy of the array
	float* arr;
	arr = new float[n];
	float med;
	int i;
	for (i = 0; i < n; i++) {
		arr[i] = a[i];

	}
	int low, high;
	int median;
	int middle, ll, hh;

	low = 0;
	high = n - 1;
	median = (low + high) / 2;
	for (;;) {
		if (high <= low) { /* One element only */
			med = arr[median];
			delete[] arr;
			return med;
		}

		if (high == low + 1) { /* Two elements only */
			if (arr[low] > arr[high])
				ELEM_SWAP(arr[low], arr[high]);
			med = arr[median];
			delete[] arr;
			return med;
		}

		/* Find median of low, middle and high items; swap into position low */
		middle = (low + high) / 2;
		if (arr[middle] > arr[high])
			ELEM_SWAP(arr[middle], arr[high]);
		if (arr[low] > arr[high])
			ELEM_SWAP(arr[low], arr[high]);
		if (arr[middle] > arr[low])
			ELEM_SWAP(arr[middle], arr[low]);

		/* Swap low item (now in position middle) into position (low+1) */ELEM_SWAP(
				arr[middle], arr[low + 1]);

		/* Nibble from each end towards middle, swapping items when stuck */
		ll = low + 1;
		hh = high;
		for (;;) {
			do
				ll++;
			while (arr[low] > arr[ll]);
			do
				hh--;
			while (arr[hh] > arr[low]);

			if (hh < ll)
				break;

			ELEM_SWAP(arr[ll], arr[hh]);
		}

		/* Swap middle item (in position low) back into correct position */ELEM_SWAP(
				arr[low], arr[hh]);

		/* Re-set active partition */
		if (hh <= median)
			low = ll;
		if (hh >= median)
			high = hh - 1;
	}

}

#undef ELEM_SWAP

/** All of the optimized median methods were written by Nicolas Devillard and are in public domain */
#define PIX_SORT(a,b) { if (a>b) PIX_SWAP(a,b); }
#define PIX_SWAP(a,b) { float temp=a;a=b;b=temp; }

/*----------------------------------------------------------------------------
 Function :   opt_med3()
 In       :   pointer to array of 3 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 3 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed3(float* p) {
	PIX_SORT(p[0], p[1]);
	PIX_SORT(p[1], p[2]);
	PIX_SORT(p[0], p[1]);
	return (p[1]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med5()
 In       :   pointer to array of 5 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 5 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed5(float* p) {
	PIX_SORT(p[0], p[1]);
	PIX_SORT(p[3], p[4]);
	PIX_SORT(p[0], p[3]);
	PIX_SORT(p[1], p[4]);
	PIX_SORT(p[1], p[2]);
	PIX_SORT(p[2], p[3]);
	PIX_SORT(p[1], p[2]);
	return (p[2]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med7()
 In       :   pointer to array of 7 pixel values
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 7 pixel values
 Notice   :   found on sci.image.processing
 cannot go faster unless assumptions are made
 on the nature of the input signal.
 ---------------------------------------------------------------------------*/

float optmed7(float* p) {
	PIX_SORT(p[0], p[5]);
	PIX_SORT(p[0], p[3]);
	PIX_SORT(p[1], p[6]);
	PIX_SORT(p[2], p[4]);
	PIX_SORT(p[0], p[1]);
	PIX_SORT(p[3], p[5]);
	PIX_SORT(p[2], p[6]);
	PIX_SORT(p[2], p[3]);
	PIX_SORT(p[3], p[6]);
	PIX_SORT(p[4], p[5]);
	PIX_SORT(p[1], p[4]);
	PIX_SORT(p[1], p[3]);
	PIX_SORT(p[3], p[4]);
	return (p[3]);
}
/*----------------------------------------------------------------------------
 Function :   opt_med9()
 In       :   pointer to an array of 9 pixelvalues
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 9 pixelvalues
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Formula from:
 XILINX XCELL magazine, vol. 23 by John L. Smith

 The input array is modified in the process
 The result array is guaranteed to contain the median
 value
 in middle position, but other elements are NOT sorted.
 ---------------------------------------------------------------------------*/

float optmed9(float* p) {
	PIX_SORT(p[1], p[2]);
	PIX_SORT(p[4], p[5]);
	PIX_SORT(p[7], p[8]);
	PIX_SORT(p[0], p[1]);
	PIX_SORT(p[3], p[4]);
	PIX_SORT(p[6], p[7]);
	PIX_SORT(p[1], p[2]);
	PIX_SORT(p[4], p[5]);
	PIX_SORT(p[7], p[8]);
	PIX_SORT(p[0], p[3]);
	PIX_SORT(p[5], p[8]);
	PIX_SORT(p[4], p[7]);
	PIX_SORT(p[3], p[6]);
	PIX_SORT(p[1], p[4]);
	PIX_SORT(p[2], p[5]);
	PIX_SORT(p[4], p[7]);
	PIX_SORT(p[4], p[2]);
	PIX_SORT(p[6], p[4]);
	PIX_SORT(p[4], p[2]);
	return (p[4]);
}

/*----------------------------------------------------------------------------
 Function :   opt_med25()
 In       :   pointer to an array of 25 pixelvalues
 Out      :   a pixelvalue
 Job      :   optimized search of the median of 25 pixelvalues
 Notice   :   in theory, cannot go faster without assumptions on the
 signal.
 Code taken from Graphic Gems.
 ---------------------------------------------------------------------------*/

float optmed25(float* p) {

	PIX_SORT(p[0], p[1]);
	PIX_SORT(p[3], p[4]);
	PIX_SORT(p[2], p[4]);
	PIX_SORT(p[2], p[3]);
	PIX_SORT(p[6], p[7]);
	PIX_SORT(p[5], p[7]);
	PIX_SORT(p[5], p[6]);
	PIX_SORT(p[9], p[10]);
	PIX_SORT(p[8], p[10]);
	PIX_SORT(p[8], p[9]);
	PIX_SORT(p[12], p[13]);
	PIX_SORT(p[11], p[13]);
	PIX_SORT(p[11], p[12]);
	PIX_SORT(p[15], p[16]);
	PIX_SORT(p[14], p[16]);
	PIX_SORT(p[14], p[15]);
	PIX_SORT(p[18], p[19]);
	PIX_SORT(p[17], p[19]);
	PIX_SORT(p[17], p[18]);
	PIX_SORT(p[21], p[22]);
	PIX_SORT(p[20], p[22]);
	PIX_SORT(p[20], p[21]);
	PIX_SORT(p[23], p[24]);
	PIX_SORT(p[2], p[5]);
	PIX_SORT(p[3], p[6]);
	PIX_SORT(p[0], p[6]);
	PIX_SORT(p[0], p[3]);
	PIX_SORT(p[4], p[7]);
	PIX_SORT(p[1], p[7]);
	PIX_SORT(p[1], p[4]);
	PIX_SORT(p[11], p[14]);
	PIX_SORT(p[8], p[14]);
	PIX_SORT(p[8], p[11]);
	PIX_SORT(p[12], p[15]);
	PIX_SORT(p[9], p[15]);
	PIX_SORT(p[9], p[12]);
	PIX_SORT(p[13], p[16]);
	PIX_SORT(p[10], p[16]);
	PIX_SORT(p[10], p[13]);
	PIX_SORT(p[20], p[23]);
	PIX_SORT(p[17], p[23]);
	PIX_SORT(p[17], p[20]);
	PIX_SORT(p[21], p[24]);
	PIX_SORT(p[18], p[24]);
	PIX_SORT(p[18], p[21]);
	PIX_SORT(p[19], p[22]);
	PIX_SORT(p[8], p[17]);
	PIX_SORT(p[9], p[18]);
	PIX_SORT(p[0], p[18]);
	PIX_SORT(p[0], p[9]);
	PIX_SORT(p[10], p[19]);
	PIX_SORT(p[1], p[19]);
	PIX_SORT(p[1], p[10]);
	PIX_SORT(p[11], p[20]);
	PIX_SORT(p[2], p[20]);
	PIX_SORT(p[2], p[11]);
	PIX_SORT(p[12], p[21]);
	PIX_SORT(p[3], p[21]);
	PIX_SORT(p[3], p[12]);
	PIX_SORT(p[13], p[22]);
	PIX_SORT(p[4], p[22]);
	PIX_SORT(p[4], p[13]);
	PIX_SORT(p[14], p[23]);
	PIX_SORT(p[5], p[23]);
	PIX_SORT(p[5], p[14]);
	PIX_SORT(p[15], p[24]);
	PIX_SORT(p[6], p[24]);
	PIX_SORT(p[6], p[15]);
	PIX_SORT(p[7], p[16]);
	PIX_SORT(p[7], p[19]);
	PIX_SORT(p[13], p[21]);
	PIX_SORT(p[15], p[23]);
	PIX_SORT(p[7], p[13]);
	PIX_SORT(p[7], p[15]);
	PIX_SORT(p[1], p[9]);
	PIX_SORT(p[3], p[11]);
	PIX_SORT(p[5], p[17]);
	PIX_SORT(p[11], p[17]);
	PIX_SORT(p[9], p[17]);
	PIX_SORT(p[4], p[10]);
	PIX_SORT(p[6], p[12]);
	PIX_SORT(p[7], p[14]);
	PIX_SORT(p[4], p[6]);
	PIX_SORT(p[4], p[7]);
	PIX_SORT(p[12], p[14]);
	PIX_SORT(p[10], p[14]);
	PIX_SORT(p[6], p[7]);
	PIX_SORT(p[10], p[12]);
	PIX_SORT(p[6], p[10]);
	PIX_SORT(p[6], p[17]);
	PIX_SORT(p[12], p[17]);
	PIX_SORT(p[7], p[17]);
	PIX_SORT(p[7], p[10]);
	PIX_SORT(p[12], p[18]);
	PIX_SORT(p[7], p[12]);
	PIX_SORT(p[10], p[18]);
	PIX_SORT(p[12], p[20]);
	PIX_SORT(p[10], p[20]);
	PIX_SORT(p[10], p[12]);

	return (p[12]);
}

#undef PIX_SORT
#undef PIX_SWAP

/**
 * All of these median filters don't do anything to a border of pixels the size of the half width
 */
float* medfilt3(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 1; j < ny - 1; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {

				counter = 0;
				for (k = -1; k < 2; k++) {
					nxk = nx * k;
					for (l = -1; l < 2; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = optmed9(medarr);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + nx - 1] = data[nxj + nx - 1];
	}

	return output;
}

float* medfilt5(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[25];

#pragma omp for nowait
		for (j = 2; j < ny - 2; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {

				counter = 0;
				for (k = -2; k < 3; k++) {
					nxk = nx * k;
					for (l = -2; l < 3; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = optmed25(medarr);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
	}

	return output;
}

float* medfilt7(float* data, int nx, int ny) {
	/**
	 * To save on space and computation we just leave the border pixels alone. Most data has blank edges
	 * and funny things happen at the edges anyway so we don't worry too much about it.
	 *
	 */

	int i;
	int j;
	int nxj;
	int nxny = nx * ny;

	float* output;
	output = new float[nxny];
	int k, l, nxk;
	float* medarr;
	int counter;

#pragma omp parallel firstprivate(output,data,nx,ny) private(i,j,k,l,medarr,nxj,counter,nxk)
	{
		medarr = new float[49];

#pragma omp for nowait
		for (j = 3; j < ny - 3; j++) {
			nxj = nx * j;
			for (i = 3; i < nx - 3; i++) {

				counter = 0;
				for (k = -3; k < 4; k++) {
					nxk = nx * k;
					for (l = -3; l < 4; l++) {
						medarr[counter] = data[nxj + i + nxk + l];
						counter++;
					}
				}

				output[nxj + i] = median(medarr, 49);
			}
		}

		delete[] medarr;
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[i + nx + nx] = data[i + nx + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
		output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
	}
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
	}

	return output;
}

float* sepmedfilt3(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median separates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[3];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				rowmed[nxj + i] = optmed3(medarr);
			}
		}
		delete[] medarr;
	}
	//Fill in the edges of the row med with the data values
#pragma omp parallel for firstprivate(rowmed, nx,ny,data) private(j,nxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		rowmed[nxj] = data[nxj];
		rowmed[nxj + nx - 1] = data[nxj + nx - 1];

	}
	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[3];

#pragma omp for nowait
		for (j = 1; j < ny - 1; j++) {
			nxj = nx * j;
			for (i = 1; i < nx - 1; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj];
				output[nxj + i] = optmed3(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + nx - 1] = data[nxj + nx - 1];
	}

	return output;
}

float* sepmedfilt5(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median seperates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[5];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				medarr[3] = data[nxj + i - 2];
				medarr[4] = data[nxj + i + 2];
				rowmed[nxj + i] = optmed5(medarr);
			}
		}
		delete[] medarr;
	}

	//Fill in the edges of the row med with the data values
#pragma omp parallel for firstprivate(rowmed, nx,ny) private(j,nxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		rowmed[nxj] = data[nxj];
		rowmed[nxj + 1] = data[nxj + 1];
		rowmed[nxj + nx - 1] = data[nxj + nx - 1];
		rowmed[nxj + nx - 2] = data[nxj + nx - 2];

	}
	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[5];

#pragma omp for nowait
		for (j = 2; j < ny - 2; j++) {
			nxj = nx * j;
			for (i = 2; i < nx - 2; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj + nx + nx];
				medarr[3] = rowmed[i + nxj - nx - nx];
				medarr[4] = rowmed[i + nxj];
				output[nxj + i] = optmed5(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
	}

	return output;
}

float* sepmedfilt7(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median separates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[7];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 3; i < nx - 3; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				medarr[3] = data[nxj + i - 2];
				medarr[4] = data[nxj + i + 2];
				medarr[5] = data[nxj + i - 3];
				medarr[6] = data[nxj + i + 3];
				rowmed[nxj + i] = optmed7(medarr);
			}
		}
		delete[] medarr;
	}
//Fill in the edges of the row med with the data values
#pragma omp parallel for firstprivate(rowmed, nx,ny) private(j,nxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		rowmed[nxj] = data[nxj];
		rowmed[nxj + 1] = data[nxj + 1];
		rowmed[nxj + 2] = data[nxj + 2];
		rowmed[nxj + nx - 1] = data[nxj + nx - 1];
		rowmed[nxj + nx - 2] = data[nxj + nx - 2];
		rowmed[nxj + nx - 3] = data[nxj + nx - 3];

	}
	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 3; j < ny - 3; j++) {
			nxj = nx * j;
			for (i = 3; i < nx - 3; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj + nx + nx];
				medarr[3] = rowmed[i + nxj - nx - nx];
				medarr[4] = rowmed[i + nxj];
				medarr[5] = rowmed[i + nxj + nx + nx + nx];
				medarr[6] = rowmed[i + nxj - nx - nx - nx];
				output[nxj + i] = optmed7(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[i + nx + nx] = data[i + nx + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
		output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
	}

	return output;
}

float* sepmedfilt9(float* data, int nx, int ny) {
	//Just ignore the borders, fill them with data as strange things happen along the edges anyway
	int nxny = nx * ny;

	float* rowmed;
	rowmed = new float[nxny];
	int i;
	int j;
	int nxj;

	//The median seperates so we can median the rows and then median the columns
	float* medarr;
#pragma omp parallel firstprivate(data,rowmed,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			for (i = 4; i < nx - 4; i++) {
				medarr[0] = data[nxj + i];
				medarr[1] = data[nxj + i - 1];
				medarr[2] = data[nxj + i + 1];
				medarr[3] = data[nxj + i - 2];
				medarr[4] = data[nxj + i + 2];
				medarr[5] = data[nxj + i - 3];
				medarr[6] = data[nxj + i + 3];
				medarr[7] = data[nxj + i - 4];
				medarr[8] = data[nxj + i + 4];
				rowmed[nxj + i] = optmed9(medarr);
			}
		}
		delete[] medarr;
	}
	//Fill in the edges of the row med with the data values
#pragma omp parallel for firstprivate(rowmed, nx,ny) private(j,nxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		rowmed[nxj] = data[nxj];
		rowmed[nxj + 1] = data[nxj + 1];
		rowmed[nxj + 2] = data[nxj + 2];
		rowmed[nxj + 3] = data[nxj + 3];
		rowmed[nxj + nx - 1] = data[nxj + nx - 1];
		rowmed[nxj + nx - 2] = data[nxj + nx - 2];
		rowmed[nxj + nx - 3] = data[nxj + nx - 3];
		rowmed[nxj + nx - 4] = data[nxj + nx - 4];

	}
	float* output;
	output = new float[nxny];

#pragma omp parallel firstprivate(rowmed,output,nx,ny) private(i,j,nxj,medarr)
	{
		medarr = new float[9];

#pragma omp for nowait
		for (j = 4; j < ny - 4; j++) {
			nxj = nx * j;
			for (i = 4; i < nx - 4; i++) {

				medarr[0] = rowmed[i + nxj - nx];
				medarr[1] = rowmed[i + nxj + nx];
				medarr[2] = rowmed[i + nxj + nx + nx];
				medarr[3] = rowmed[i + nxj - nx - nx];
				medarr[4] = rowmed[i + nxj];
				medarr[5] = rowmed[i + nxj + nx + nx + nx];
				medarr[6] = rowmed[i + nxj - nx - nx - nx];
				medarr[7] = rowmed[i + nxj + nx + nx + nx + nx];
				medarr[8] = rowmed[i + nxj - nx - nx - nx - nx];
				output[nxj + i] = optmed9(medarr);
			}
		}
		delete medarr;
	}
	delete[] rowmed;
	//Fill up the skipped borders
#pragma omp parallel for firstprivate(output,nx,ny,nxny) private(i,j,nxj)
	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[i + nx] = data[i + nx];
		output[i + nx + nx] = data[i + nx + nx];
		output[i + nx + nx + nx] = data[i + nx + nx + nx];
		output[nxny - nx + i] = data[nxny - nx + i];
		output[nxny - nx - nx + i] = data[nxny - nx - nx + i];
		output[nxny - nx - nx - nx + i] = data[nxny - nx - nx - nx + i];
		output[nxny - nx - nx - nx - nx + i] =
				data[nxny - nx - nx - nx - nx + i];
	}
#pragma omp parallel for firstprivate(output,nx,ny) private(i,nxj)
	for (i = 0; i < ny; i++) {
		nxj = nx * i;
		output[nxj] = data[nxj];
		output[nxj + 1] = data[nxj + 1];
		output[nxj + 2] = data[nxj + 2];
		output[nxj + 3] = data[nxj + 3];
		output[nxj + nx - 1] = data[nxj + nx - 1];
		output[nxj + nx - 2] = data[nxj + nx - 2];
		output[nxj + nx - 3] = data[nxj + nx - 3];
		output[nxj + nx - 4] = data[nxj + nx - 4];
	}

	return output;
}

float* subsample(float* data, int nx, int ny) {
	float* output;
	output = new float[4 * nx * ny];
	int padnx = 2 * nx;
	int i;
	int j;
	int nxj;
	int padnxj;
#pragma omp parallel for firstprivate(padnx,data,output,nx,ny) private(i,j,nxj,padnxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		padnxj = 2 * padnx * j;
		for (i = 0; i < nx; i++) {
			output[2 * i + padnxj] = data[i + nxj];
			output[2 * i + padnxj + padnx] = data[i + nxj];
			output[2 * i + 1 + padnxj + padnx] = data[i + nxj];
			output[2 * i + 1 + padnxj] = data[i + nxj];
		}
	}

	return output;
}

bool* dilate(bool* data, int iter, int nx, int ny) {
	/**
	 * Here we do a boolean dilation of the image to connect the cosmic rays for the masks
	 * We use a kernel that looks like
	 * 0 1 1 1 0
	 * 1 1 1 1 1
	 * 1 1 1 1 1
	 * 1 1 1 1 1
	 * 0 1 1 1 0
	 *
	 * Since we have to do multiple iterations, this takes a little more memory.
	 * But it's bools so its ok.
	 */
	//Pad the array with a border of zeros
	int padnx = nx + 4;
	int padny = ny + 4;
	int padnxny = padnx * padny;
	int nxny = nx * ny;
	bool* padarr;
	padarr = new bool[padnxny];
	int i;
	for (i = 0; i < padnx; i++) {
		padarr[i] = false;
		padarr[i + padnx] = false;
		padarr[padnxny - padnx + i] = false;
		padarr[padnxny - padnx - padnx + i] = false;
	}
	for (i = 0; i < padny; i++) {

		padarr[padnx * i] = false;
		padarr[padnx * i + 1] = false;
		padarr[padnx * i + padnx - 1] = false;
		padarr[padnx * i + padnx - 2] = false;
	}

	bool* output;
	output = new bool[nxny];

	//Set the first iteration output array to the input data
	for (i = 0; i < nxny; i++) {
		output[i] = data[i];
	}

	int counter;
	int j;
	int nxj;
	int padnxj;
	for (counter = 0; counter < iter; counter++) {
#pragma omp parallel for firstprivate(padarr,output,nx,ny,padnx,padny,counter) private(nxj,padnxj,i,j)
		for (j = 0; j < ny; j++) {
			padnxj = padnx * j;
			nxj = nx * j;
			for (i = 0; i < nx; i++) {
				padarr[i + 2 + padnx + padnx + padnxj] = output[i + nxj];
			}
		}
#pragma omp parallel for firstprivate(padarr,output,nx,ny,padnx,padny,counter) private(nxj,padnxj,i,j)
		for (j = 0; j < ny; j++) {
			nxj = nx * j;
			padnxj = padnx * j;
			for (i = 0; i < nx; i++) {

				//Start in the middle and work out
				output[i + nxj] = padarr[i + 2 + padnx + padnx + padnxj]
						||
						//right 1
						padarr[i + 3 + padnx + padnx + padnxj]
						||
						//left 1
						padarr[i + 1 + padnx + padnx + padnxj]
						||
						//up 1
						padarr[i + 2 + padnx + padnx + padnx + padnxj]
						||
						//down 1
						padarr[i + 2 + padnx + padnxj]
						||
						//up 1 right 1
						padarr[i + 3 + padnx + padnx + padnx + padnxj]
						||
						//up 1 left 1
						padarr[i + 1 + padnx + padnx + padnx + padnxj]
						||
						//down 1 right 1
						padarr[i + 3 + padnx + padnxj]
						||
						//down 1 left 1
						padarr[i + 1 + padnx + padnxj]
						||
						//right 2
						padarr[i + 4 + padnx + padnx + padnxj]
						||
						//left 2
						padarr[i + padnx + padnx + padnxj]
						||
						//up 2
						padarr[i + 2 + padnx + padnx + padnx + padnx + padnxj]
						||
						//down 2
						padarr[i + 2 + padnxj]
						||
						//right 2 up 1
						padarr[i + 4 + padnx + padnx + padnx + padnxj]
						||
						//right 2 down 1
						padarr[i + 4 + padnx + padnxj]
						||
						//left 2 up 1
						padarr[i + padnx + padnx + padnx + padnxj]
						||
						//left 2 down 1
						padarr[i + padnx + padnxj]
						||
						//up 2 right 1
						padarr[i + 3 + padnx + padnx + padnx + padnx + padnxj]
						||
						//up 2 left 1
						padarr[i + 1 + padnx + padnx + padnx + padnx + padnxj]
						||
						//down 2 right 1
						padarr[i + 3 + padnxj] ||
						//down 2 left 1
						padarr[i + 1 + padnxj];

			}
		}

	}
	delete[] padarr;

	return output;
}
float* laplaceconvolve(float* data, int nx, int ny) {
	/*
	 * Here we do a short circuited convolution using the kernel
	 *  0 -1  0
	 * -1  4 -1
	 *  0 -1  0
	 */

	int nxny = nx * ny;
	float* output;
	output = new float[nxny];
	int i;
	int j;
	int nxj;
	//Do all but the edges that we will do explicitly to save memory.
#pragma omp parallel for firstprivate(nx,ny,nxny,output,data) private(i,j,nxj)
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;
		for (i = 1; i < nx - 1; i++) {

			output[nxj + i] = 4.0 * data[nxj + i] - data[i + 1 + nxj]
					- data[i - 1 + nxj] - data[i + nxj + nx]
					- data[i + nxj - nx];
		}
	}

	//bottom row and top row
	for (i = 1; i < nx - 1; i++) {
		output[i] = 4.0 * data[i] - data[i + 1] - data[i - 1] - data[i + nx];
		output[i + nxny - nx] = 4.0 * data[i + nxny - nx]
				- data[i + 1 + nxny - nx] - data[i + nxny - nx - 1]
				- data[i - nx + nxny - nx];
	}

	//first and last column
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;
		output[nxj] = 4.0 * data[nxj] - data[nxj + 1] - data[nxj + nx]
				- data[nxj - nx];
		output[nxj + nx - 1] = 4.0 * data[nxj + nx - 1] - data[nxj + nx - 2]
				- data[nxj + nx + nx - 1] - data[nxj - 1];
	}

	//bottom left corner
	output[0] = 4.0 * data[0] - data[1] - data[nx];
	//bottom right corner
	output[nx - 1] = 4.0 * data[nx - 1] - data[nx - 2] - data[nx + nx - 1];
	//top left corner
	output[nxny - nx] = 4.0 * data[nxny - nx] - data[nxny - nx + 1]
			- data[nxny - nx - nx];
	//top right corner
	output[nxny - 1] = 4.0 * data[nxny - 1] - data[nxny - 2]
			- data[nxny - 1 - nx];

	return output;
}

bool* growconvolve(bool* data, int nx, int ny) {
	/* This basically does a binary dilation with all ones in a 3x3 kernel:
	 * I have not decided if this is exactly equivalent or which is faster to calculate.
	 * In python this looks like
	 *  np.cast['bool'](signal.convolve2d(np.cast['float32'](cosmics), growkernel, mode="same", boundary="symm"))
	 * For speed and memory savings, I just convolve the whole image except the border. The border is just copied from the input image
	 * This is not technically correct, but it should be good enough.
	 */

	//Pad the array with a border of zeros
	int nxny = nx * ny;
	int i;
	int j;
	int nxj;
	bool* output;
	output = new bool[nxny];

#pragma omp parallel for firstprivate(output,data,nxny,nx,ny) private(i,j,nxj)
	for (j = 1; j < ny - 1; j++) {
		nxj = nx * j;

		for (i = 1; i < nx - 1; i++) {
			//Start in the middle and work out
			output[i + nxj] = data[i + nxj] ||
			//right 1
					data[i + 1 + nxj] ||
					//left 1
					data[i - 1 + nxj] ||
					//up 1
					data[i + nx + nxj] ||
					//down 1
					data[i - nx + nxj] ||
					//up 1 right 1
					data[i + 1 + nx + nxj] ||
					//up 1 left 1
					data[i - 1 + nx + nxj] ||
					//down 1 right 1
					data[i + 1 - nx + nxj] ||
					//down 1 left 1
					data[i - 1 - nx + nxj];

		}
	}

	for (i = 0; i < nx; i++) {
		output[i] = data[i];
		output[nxny - nx + i] = data[nxny - nx + i];
	}
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		output[nxj] = data[nxj];
		output[nxj - 1 + nx] = data[nxj - 1 + nx];
	}

	return output;
}

float* rebin(float* data, int nx, int ny) {
	//Basically we want to do the opposite of subsample averaging the 4 pixels back down to 1
	//nx and ny are the final dimensions of the rebinned image
	float* output;
	output = new float[nx * ny];
	int padnx = nx * 2;
	int i;
	int j;
	int nxj;
	int padnxj;
#pragma omp parallel for firstprivate(output,data,nx,ny,padnx) private(i,j,nxj,padnxj)
	for (j = 0; j < ny; j++) {
		nxj = nx * j;
		padnxj = 2 * padnx * j;
		for (i = 0; i < nx; i++) {
			output[i + nxj] = (data[2 * i + padnxj]
					+ data[2 * i + padnxj + padnx]
					+ data[2 * i + 1 + padnxj + padnx]
					+ data[2 * i + 1 + padnxj]) / 4.0;
		}
	}
	return output;
}

void updatemask(float* data, bool* mask, float satlevel, int nx, int ny,
		bool fullmedian) {
	/*
	 Uses the satlevel to find saturated stars (not cosmics !), and puts the result as a mask in the mask.
	 This can then be used to avoid these regions in cosmic detection and cleaning procedures.
	 */

	// DETECTION
	int nxny = nx * ny;
	//Find all of the saturated pixels
	bool* satpixels;
	satpixels = new bool[nxny];

	int i;

#pragma omp parallel for firstprivate(nxny,data,satlevel,satpixels) private(i)
	for (i = 0; i < nxny; i++) {
		satpixels[i] = data[i] >= satlevel;
	}

	//in an attempt to avoid saturated cosmic rays we try prune the saturated stars using the large scale structure
	float* m5;
	if (fullmedian) {
		m5 = medfilt5(data, nx, ny);
	} else {
		m5 = sepmedfilt7(data, nx, ny);
	}
	//This mask will include saturated pixels and masked pixels

#pragma omp parallel for firstprivate(nxny,satlevel,m5,satpixels) private(i)
	for (i = 0; i < nxny; i++) {
		satpixels[i] = satpixels[i] && (m5[i] > satlevel / 10.0);
	}
	delete[] m5;

	// BUILDING THE MASK

	//Combine the saturated pixels with the given input mask
	//Grow the input mask by one pixel to make sure we cover bad pixels
	bool* grow_mask = growconvolve(mask, nx, ny);

	//We want to dilate both the mask and the saturated stars to remove false detections along the edges of the mask
	bool* dilsatpixels = dilate(satpixels, 2, nx, ny);
	delete[] satpixels;

#pragma omp parallel for firstprivate(nxny,mask,dilsatpixels,grow_mask) private(i)
	for (i = 0; i < nxny; i++) {
		mask[i] = dilsatpixels[i] || grow_mask[i];
	}
	delete[] dilsatpixels;
	delete[] grow_mask;
}

void clean(float* cleanarr, bool* crmask, int nx, int ny,
		float backgroundlevel) {
	//Go through all of the pixels, ignore the borders
	int i;
	int j;
	int nxj;
	int idx;
	int k, l;
	int nxl;
	float sum;
	int numpix;

#pragma omp parallel for firstprivate(nx,ny,crmask,cleanarr,backgroundlevel) private(i,j,nxj,idx,numpix,sum,nxl,k,l)
	for (j = 2; j < ny - 2; j++) {
		nxj = nx * j;
		for (i = 2; i < nx - 2; i++) {
			//if the pixel is in the crmask
			idx = nxj + i;
			if (crmask[idx]) {
				numpix = 0;
				sum = 0.0;
				//sum the 25 pixels around the pixel ignoring any pixels that are masked

				for (l = -2; l < 3; l++) {
					nxl = nx * l;
					for (k = -2; k < 3; k++) {
						if (!crmask[idx + k + nxl]) {
							sum += cleanarr[idx + k + nxl];
							numpix++;
						}
					}

				}

				//if the pixels count is 0 then put in the background of the image
				if (numpix == 0) {
					sum = backgroundlevel;
				} else {
					//else take the mean
					sum /= float(numpix);
				}
				cleanarr[idx] = sum;
			}
		}
	}
}

int lacosmiciteration(float* cleanarr, bool* mask, bool* crmask, float sigclip,
		float objlim, float sigfrac, float backgroundlevel, float readnoise,
		int nx, int ny, bool fullmedian) {
	/*
	 Performs one iteration of the L.A.Cosmic algorithm.
	 It operates on cleanarray, and afterwards updates crmask by adding the newly detected
	 cosmics to the existing crmask. Cleaning is not done automatically ! You have to call
	 clean() after each iteration.
	 This way you can run it several times in a row to to L.A.Cosmic "iterations".
	 See function lacosmic, that mimics the full iterative L.A.Cosmic algorithm.

	 Returns numcr : the number of cosmic pixels detected in this iteration

	 */

	// We subsample, convolve, clip negative values, and rebin to original size
	float* subsam = subsample(cleanarr, nx, ny);
	int nxny = nx * ny;
	float sigcliplow = sigclip * sigfrac;
	float* conved = laplaceconvolve(subsam, 2 * nx, 2 * ny);
	delete[] subsam;

	int i;
	int nxny4 = 4 * nxny;
#pragma omp parallel for firstprivate(nxny4,conved) private(i)
	for (i = 0; i < nxny4; i++) {
		if (conved[i] < 0.0) {
			conved[i] = 0.0;
		}
	}

	float* s = rebin(conved, nx, ny);
	delete[] conved;

	// We build a custom noise map, to compare the laplacian to

	float* m5;
	if (fullmedian) {
		m5 = medfilt5(cleanarr, nx, ny);
	} else {
		m5 = sepmedfilt7(cleanarr, nx, ny);
	}
	float* noise;
	noise = new float[nxny];

#pragma omp parallel for firstprivate(nxny,readnoise,m5,noise) private(i)
	for (i = 0; i < nxny; i++) {
		// We clip noise so that we can take a square root
		m5[i] < 0.0001 ? noise[i] = 0.0001 : noise[i] = m5[i];
		noise[i] = sqrt(noise[i] + readnoise * readnoise);
	}

	delete[] m5;

	// Laplacian signal to noise ratio :

#pragma omp parallel for firstprivate(nxny,noise,s) private(i)
	for (i = 0; i < nxny; i++) {
		s[i] = s[i] / (2.0 * noise[i]);
		// the 2.0 is from the 2x2 subsampling
		// This s is called sigmap in the original lacosmic.cl
	}

	float* sp;
	if (fullmedian) {
		sp = medfilt5(s, nx, ny);
	} else {
		sp = sepmedfilt7(s, nx, ny);
	}

	// We remove the large structures (s prime) :
#pragma omp parallel for firstprivate(nxny,s,sp) private(i)
	for (i = 0; i < nxny; i++) {
		sp[i] = s[i] - sp[i];
	}
	delete[] s;

	// We build the fine structure image :
	float* m3;
	if (fullmedian) {
		m3 = medfilt3(cleanarr, nx, ny);
	} else {
		m3 = sepmedfilt5(cleanarr, nx, ny);
	}
	float* f;
	if (fullmedian) {
		f = medfilt7(m3, nx, ny);
	} else {
		f = sepmedfilt9(m3, nx, ny);
	}

#pragma omp parallel for firstprivate(nxny,f,m3,noise) private(i)
	for (i = 0; i < nxny; i++) {
		f[i] = (m3[i] - f[i]) / noise[i];
		if (f[i] < 0.01) {
			// as we will divide by f. like in the iraf version.
			f[i] = 0.01;
		}
	}

	delete[] m3;
	delete[] noise;

	//Comments from Malte Tewes
	// In the article that's it, but in lacosmic.cl f is divided by the noise...
	// Ok I understand why, it depends on if you use sp/f or L+/f as criterion.
	// There are some differences between the article and the iraf implementation.
	// So we will stick to the iraf implementation.

	// Now we have our better selection of cosmics :
	// Note the sp/f and not lplus/f ... due to the f = f/noise above.
	bool* cosmics = new bool[nxny];

#pragma omp parallel for firstprivate(cosmics,sp,f,objlim,sigclip,nxny,mask) private(i)
	for (i = 0; i < nxny; i++) {
		cosmics[i] = (sp[i] > sigclip) && !mask[i] && ((sp[i] / f[i]) > objlim);
	}

	delete[] f;

	// What follows is a special treatment for neighbors, with more relaxed constraints.
	// We grow these cosmics a first time to determine the immediate neighborhood  :
	bool* growcosmics = growconvolve(cosmics, nx, ny);

	delete[] cosmics;
	// From this grown set, we keep those that have sp > sigmalim
	// so obviously not requiring sp/f > objlim, otherwise it would be pointless
	//This step still feels pointless to me, but we leave it in because the iraf implementation has it
#pragma omp parallel for firstprivate(nxny,sp,growcosmics,mask,sigclip) private(i)
	for (i = 0; i < nxny; i++) {
		growcosmics[i] = (sp[i] > sigclip) && growcosmics[i] && !mask[i];
	}

	// Now we repeat this procedure, but lower the detection limit to sigmalimlow :
	bool* finalsel = growconvolve(growcosmics, nx, ny);
	delete[] growcosmics;

	//Our CR counter
	int numcr = 0;
#pragma omp parallel for reduction(+ : numcr) firstprivate(finalsel,sp,sigcliplow,nxny,mask) private(i)
	for (i = 0; i < nxny; i++) {
		finalsel[i] = (sp[i] > sigcliplow) && finalsel[i] && (!mask[i]);
		if (finalsel[i]) {
			numcr++;
		}
	}
	delete[] sp;

	// We update the crmask with the cosmics we have found :

#pragma omp parallel for firstprivate(crmask,finalsel,nxny) private(i)
	for (i = 0; i < nxny; i++) {
		crmask[i] = crmask[i] || finalsel[i];
	}

	// Now the replacement of the cosmics...
	// we outsource this to the function clean(), as for some purposes the cleaning might not even be needed.
	// Easy way without masking would be :
	//self.cleanarray[finalsel] = m5[finalsel]
	//Go through and clean the image using a masked mean filter, we outsource this to the clean method
	clean(cleanarr, crmask, nx, ny, backgroundlevel);

	delete[] finalsel;
	// We return the number of cr pixels
	// (used by function lacosmic)

	return numcr;
}

