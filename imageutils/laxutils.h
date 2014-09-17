/*
 * laxutils.h
 *
 *  Created on: May 27, 2011
 *      Author: cmccully
 */

#ifndef LAXUTILS_H_
#define LAXUTILS_H_
#include <stdbool.h>
float _median(float* a, int n);
float _optmed25(float* a);
float _optmed3(float* a);
float _optmed7(float* a);
float _optmed5(float* a);
float _optmed9(float* a);
float* _medfilt3(float* data, int nx, int ny);
float* _medfilt5(float* data, int nx, int ny);
float* _medfilt7(float* data, int nx, int ny);
float* _sepmedfilt3(float* data, int nx, int ny);
float* _sepmedfilt5(float* data, int nx, int ny);
float* _sepmedfilt7(float* data, int nx, int ny);
float* _sepmedfilt9(float* data, int nx, int ny);
bool* _dilate(bool* data, int iter, int nx, int ny);
float* _subsample(float* data, int nx, int ny);
float* _laplaceconvolve(float* data, int nx, int ny);
float* _rebin(float* data, int nx, int ny);
bool* _growconvolve(bool* data, int nx, int ny);
void updatemask(float* data, bool* mask, float satlevel, int nx, int ny, bool fullmedian);
int lacosmiciteration(float* cleanarr, bool* mask, bool* crmask, float sigclip, float objlim, float sigfrac, float backgroundlevel, float readnoise, int nx, int ny, bool fullmedian);


#endif /* LAXUTILS_H_ */
