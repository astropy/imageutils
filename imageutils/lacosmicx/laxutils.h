/*
 * laxutils.h
 *
 * Author: Curtis McCully
 */

#ifndef LAXUTILS_H_
#define LAXUTILS_H_
/* Define a bool type because there isn't one built in to ANSI C */
typedef uint8_t bool;
#define true 1
#define false 0

float PyMedian(float* a, int n);
float PyOptMed3(float* a);
float PyOptMed5(float* a);
float PyOptMed7(float* a);
float PyOptMed9(float* a);
float PyOptMed25(float* a);
float* PyMedFilt3(float* data, int nx, int ny);
float* PyMedFilt5(float* data, int nx, int ny);
float* PyMedFilt7(float* data, int nx, int ny);
float* PySepMedFilt3(float* data, int nx, int ny);
float* PySepMedFilt5(float* data, int nx, int ny);
float* PySepMedFilt7(float* data, int nx, int ny);
float* PySepMedFilt9(float* data, int nx, int ny);
float* PySubsample(float* data, int nx, int ny);
float* PyRebin(float* data, int nx, int ny);
float* PyConvolve(float* data, float* kernel, int nx, int ny, int kernx,
                 int kerny);
float* PyLaplaceConvolve(float* data, int nx, int ny);
bool* PyDilate3(bool* data, int nx, int ny);
bool* PyDilate5(bool* data, int iter, int nx, int ny);

#endif /* LAXUTILS_H_ */
