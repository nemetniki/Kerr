/*************************************************** 
 *   Copyright (C) 2008 by Alexander Carmele   *
 *   221b Baker St, Marylebone, London W1.     *
 ***************************************************/

// compile with gcc 2NS_par_OmL.c -lm
// ./a.out

#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#include <time.h>
#define hbar ( 0.658212196 )     // eVfs 
#define PI M_PI
// ################ ZEITRAUM #######################
#define TIME_DYNAMICS_TO ( 1.000 )
#define T_STEPS 50000
#define TAU_FAKTOR (2)
int N     = T_STEPS;
int N_TAU = TAU_FAKTOR*T_STEPS;
int STEP_IS_SAVED= 1; // jeder ... Schritt wird in ein File geschrieben!!
int TIME_OUTPUT  =2000;
int TIME_OUTPUT_TAU  = 2000;
double delta_t = ((double)TIME_DYNAMICS_TO) * 1000. * 1000. /( (double)T_STEPS); 

#define Mc    ( 0.0004  * 1.    )
//#define CAVITY_OFF // !!! in derivates Mcc -> 0 !!!
#define KAPPA ( Mc * 0.5   * 1.  ) //( Mc/2) 
#define OmL   ( Mc * 1.5   * 0.  ) // quantum dot driving
#define OmC   ( Mc * 0.025  * 1.  ) // cavity driving 
#define GAM_R ( Mc * 0.05  * 0.0 ) //( Mc/50)
#define GAM_P ( Mc * 0.24  * 0.0 ) //( Mc/5)
#define INC_P ( Mc * 0.05  * 0.0 ) // only for 1EXCITON !!!!!!! 
// Zahl der Photonen
 #define N_p    (5)
// Spektrum
#define omega_qd        (1.500 * 1. + 1.75 * Mc       )
#define omega_laser     (1.500 * 1. - 0.0 * Mc       )
#define omega_cavity    (1.500 * 1. - 0.0 * Mc       )
#define LASER_DETUNING  ( omega_qd     - omega_laser  )
#define CAVITY_DETUNING ( omega_cavity - omega_laser  )
// -------------- spectrum output ----------------------------
#define NORMALIZE_SPECTRUM
#define Nk (1280)
#define k_interval ( 32.*Mc ) // symmetric output if (2xoutput interval + max_detuning)
#define delta_k ( k_interval/Nk )
#define CENTER_OF_SPECTRUM ( 0.*Mc)
#define SPECTRUM_OUTPUT_INTERVAL (16.*Mc ) // minus plus this around zero as output
#define MAX_STEPS (300000) // to reach steady state in tau

// #########################################################
// ############# RK4 #######################################
// #########################################################

#define CALCULATE_TEMP(a,b,cb,dest,i) ({for(i=0;i<N_DGL;i++) (dest)[i]=(a)[i]+(cb)*(b)[i];})
#define ADD_VECTOR(dest,in,in_fak,i) ({for(i=0;i<N_DGL;i++) dest[i]=dest[i]+(in_fak)* (in)[i];})

#define SECHSTEL 0.1666666666666666666666667
#define DRITTEL  0.3333333333333333333333333
#define HALFTE   0.5


// ############################################
// ############################################

// Fourier transform -- of the correlation function
void calculate_spectrum
(
 complex double *correlation,
 double *spec_inc,
 double *spec_sum,
 int    k_MAX,
 double obs_coh)
{
complex double spectrum_inc[Nk];
complex double spectrum_sum[Nk];
complex double spectrum_step[Nk];
complex double spectrum_fak[Nk];
// find the [q] on resonance
double spectrum_freq;
double min_arg = 2.;
int q_res=-10;
int q,k;

for(q=0;q<Nk;q++) 
{ spectrum_inc[q] = 0.+I*0.;
  spectrum_sum[q] = 0.+I*0.; 
  spectrum_fak[q] = 1.+I*0.; 
//  spectrum_freq   = -0.5*k_interval+DETUNING+delta_k*q;
  spectrum_freq   = -0.5*k_interval+delta_k*q-CENTER_OF_SPECTRUM;
  spectrum_step[q]= cexp(-I*spectrum_freq*delta_t);
  // find the position of the laser
  if ( fabs(spectrum_freq)<min_arg) { min_arg=fabs(spectrum_freq); q_res=q; }
} 
//printf("Resonance on [%i].\n",q_res); 
for (k=0; k<k_MAX; k++)
{  
 for(q=0;q<Nk;q++) 
 { 
  spectrum_inc[q] += delta_t * correlation[k] * spectrum_fak[q] ;
  // to singularly show, where the laser is !!! 
   //if (q==q_res)   spectrum_inc[q] += delta_t * correlation[k] * spectrum_fak[q] ;
  // laser peak with a Gaussian to allow a good spectral representation ... by hand
  spectrum_sum[q] += delta_t * correlation[k] * spectrum_fak[q];
  // to exclude artificial peak if q_res=0 and q starts with zero!!!!                  +
  if (q==q_res) spectrum_sum[q] +=delta_t *2.*PI*obs_coh*spectrum_fak[q];// *exp(-1.2*(q-q_res)*(q-q_res)) ; 
  spectrum_fak[q] *= spectrum_step[q];  
 }  
}
// calculate the minimum to substract minimum
double minimum_spectrum =creal(spectrum_inc[0]);
for(q=0;q<Nk;q++) 
{ 
 if (creal(spectrum_inc[q])<minimum_spectrum) 
      minimum_spectrum=creal(spectrum_inc[q]);
}  
// substract minimum to guarantee positivity
double INTEGRATED_INTENSITY = 0.;
double INTEGRATED_INTENSITY_PLUS_COHERENT = 0.;
for(k=0;k<Nk;k++) 
{
#ifdef NORMALIZE_SPECTRUM
spec_inc[k] = creal(spectrum_inc[k])-minimum_spectrum+1.;
spec_sum[k] = creal(spectrum_sum[k])-minimum_spectrum+1.;
INTEGRATED_INTENSITY               += spec_inc[k] * delta_k;
INTEGRATED_INTENSITY_PLUS_COHERENT += spec_sum[k] * delta_k;
#else
spec_inc[k] = creal(spectrum_inc[k]);
spec_sum[k] = creal(spectrum_sum[k]);
#endif
}

return ;  
}  

// ###############################################################################
// ###############################################################################


// ###############################################################################
// ###############################################################################
void STEADY_STATE_TEST
(complex double *derivates,
 complex double *deriv_stst,
         double *OUTPUT,
	 int N
)
{
 int i;
 OUTPUT[99]=0.;
 for (i=0;i<N;i++)
 {
   OUTPUT[99] += cabs(derivates[i]-deriv_stst[i]);
   deriv_stst[i] = derivates[i];
 }
 return;
}
// ###############################################################################
// ###############################################################################

// Zur Definition der Zustaende
#define N_l (2) // Zahl der Level
#define rho(a,m,b,n)      ( *(derivates     + (a)*N_l*(N_p+1)*(N_p+1) + (m)*N_l*(N_p+1) + (b)*(N_p+1)+ (n)  ))
#define rho_out(a,m,b,n)  ( *(derivates_out + (a)*N_l*(N_p+1)*(N_p+1) + (m)*N_l*(N_p+1) + (b)*(N_p+1)+ (n)  ))
#define Zahl_der_DGL      ( N_l*(N_p+1)*N_l*(N_p+1) + (N_p+1)*N_l*(N_p+1) + N_l*(N_p+1)+ (N_p+1) + 10  )
int N_DGL=Zahl_der_DGL;
#define rho_stst(a,m,b,n)      ( *(deriv_stst + (a)*N_l*(N_p+1)*(N_p+1) + (m)*N_l*(N_p+1) + (b)*(N_p+1)+ (n)  ))

// #################################################################################
// ############## DGL system  ######################################################
// #################################################################################


void calculate_next_time_step 
(
complex double *derivates, 
complex double *derivates_out,  
        double t,
        double *OUTPUT
)
{   

  
int a,b,m,n;
#ifdef CAVITY_OFF
double Mcc=0.;
#else
double Mcc=Mc;
#endif
double pOmL=OmL;
double pOmC=OmC;
double pLASER_DETUNING=LASER_DETUNING;
double pCAVITY_DETUNING=CAVITY_DETUNING;

for (n=0;n<100;n++) OUTPUT[n]=0.;

for (a=0;a<N_l;a++)
{
for (b=0;b<N_l;b++)
{
for (m=0;m<N_p;m++)
{
for (n=0;n<N_p;n++)
{
  rho_out(a,m,b,n) = 0. + I*0.;	
// ----------------------------------------------------------------------
// ------------- Left part - Density Matrix -----------------------------
// ----------------------------------------------------------------------
// ##### ground state couples via the laser and cavity mode to both exciton states   
if ((a==0) && (m>0  )){rho_out(a,m,b,n) += +I * Mcc * sqrt(m   )              * rho(1,m-1,b,n);}
if  (a==0)            {rho_out(a,m,b,n) += -1.* INC_P                         * rho(a,m  ,b,n)
                                           +I * pOmL                          * rho(1,m  ,b,n);}
// ##### first exciton couples via the laser and cavity mode to the ground and biexciton state        
if ((a==1) && (m<N_p)){rho_out(a,m,b,n) += +I * Mcc * sqrt(m+1.)              * rho(0,m+1,b,n);}
if  (a==1)            {rho_out(a,m,b,n) += -1.* GAM_R                         * rho(a,m  ,b,n)
                                           +I * pLASER_DETUNING               * rho(a,m  ,b,n)
                                           +I * pOmL                          * rho(0,m  ,b,n);}                                           
// #####  cavity mode includes the shift of the rotating frame 
if  (m<N_p)           {rho_out(a,m,b,n) += +I * pOmC              * sqrt(m+1.) * rho(a,m+1,b,n);}
if  (m>0)             {rho_out(a,m,b,n) += +I * pOmC              * sqrt(m   ) * rho(a,m-1,b,n)
                                           +I * pCAVITY_DETUNING  * m          * rho(a,m  ,b,n);}
// ----------------------------------------------------------------------
// ------------- Right part - Density Matrix -----------------------------
// ----------------------------------------------------------------------
// ##### ground state couples via the laser and cavity mode to both exciton states   
if ((b==0) && (n>0  )){rho_out(a,m,b,n) += -I * Mcc  * sqrt(n   )             * rho(a,m,1,n-1);}
if  (b==0)            {rho_out(a,m,b,n) += -1.* INC_P                         * rho(a,m  ,b,n)
                                           -I * pOmL                          * rho(a,m,1,n  );}
// ##### first exciton couples via the laser and cavity mode to the ground and biexciton state        
if ((b==1) && (n<N_p)){rho_out(a,m,b,n) += -I * Mcc  * sqrt(n+1.)             * rho(a,m,0,n+1);}
if  (b==1)            {rho_out(a,m,b,n) += -1.* GAM_R                         * rho(a,m,b,n  )
                                           -I * pLASER_DETUNING               * rho(a,m,b,n  )
                                           -I * pOmL                          * rho(a,m,0,n  );}
// ##### cavity mode includes the shift of the rotating frame 
if (n<N_p)            {rho_out(a,m,b,n) += -I * pOmC             * sqrt(n+1.) * rho(a,m,b,n+1) ;}
if  (n>0)             {rho_out(a,m,b,n) += -I * pOmC             * sqrt(n   ) * rho(a,m,b,n-1) 
                                           -I * pCAVITY_DETUNING * n          * rho(a,m,b,n  );}
// ----------------------------------------------------------------------
// ------------- Lindblad Cavity loss -----------------------------------
// ----------------------------------------------------------------------
                          rho_out(a,m,b,n) += -1.* KAPPA *      (m+n)          * rho(a,m,b,n);
if ((m<N_p-1)&&(n<N_p-1)) rho_out(a,m,b,n) += +2.* KAPPA * sqrt((m+1.)*(n+1.)) * rho(a,m+1,b,n+1);
// ----------------------------------------------------------------------
// ------------- Lindblad Radiative Verlust  ----------------------------
// ----------------------------------------------------------------------
if ((a==0)&&(b==0)) rho_out(a,m,b,n) += +2.* GAM_R * rho(1,m,1,n)
                                        +2.* GAM_R * rho(2,m,2,n);

if ((a==1)&&(b==1)) rho_out(a,m,b,n) += 2.* INC_P    * rho(0,m,0,n);
// ----------------------------------------------------------------------
// ------------- Lindblad Pure dephasing  ----------------------------
// ----------------------------------------------------------------------
if ((a==1)&&(b==0)) rho_out(a,m,b,n) += -1.* GAM_P * rho(a,m,b,n);
if ((a==0)&&(b==1)) rho_out(a,m,b,n) += -1.* GAM_P * rho(a,m,b,n);
if ((a==2)&&(b==0)) rho_out(a,m,b,n) += -1.* GAM_P * rho(a,m,b,n);
if ((a==0)&&(b==2)) rho_out(a,m,b,n) += -1.* GAM_P * rho(a,m,b,n);
// ----------------------------------------------------------------------
// ------------- Expectation Values -------------------------------------
// ----------------------------------------------------------------------
if ( (a==b) && (m==n)             ) OUTPUT[0] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (a==1)   ) OUTPUT[1] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n)             ) OUTPUT[2] += n*creal(rho(a,m,b,n));
if ( (a==b) && (m==n)             ) OUTPUT[3] += n*(n-1.)*creal(rho(a,m,b,n));
if ( (a==b) && (m==n)             ) OUTPUT[4] += n*(n-1.)*(n-2.)*creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==0)   ) OUTPUT[20] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==1)   ) OUTPUT[21] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==2)   ) OUTPUT[22] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==3)   ) OUTPUT[23] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==4)   ) OUTPUT[24] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==5)   ) OUTPUT[25] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m==6)   ) OUTPUT[26] += creal(rho(a,m,b,n));
if ( (a==b) && (m==n) && (m<N_p-1)) OUTPUT[7] += n*(n-1.)*creal(rho(a,m,b,n));
if ( (a==0) && (m==n) && (b==1)   ) OUTPUT[8] += cabs(rho(a,m,b,n));
if ( (a==b) && (m>0) && (m==n)    ) OUTPUT[5] += sqrt(m)*creal(rho(a,m,a,m-1));
if ( (a==b) && (m>0) && (m==n)    ) OUTPUT[6] += sqrt(m)*cimag(rho(a,m,a,m-1));
OUTPUT[15] = 0.;
OUTPUT[16] = 0.;

} //n
} //m
} //b
} //a

  return ;
}
// ###############################################################################
// ###################### Ende Praeambel #########################################  
// #########################  MAIN    ############################################ 
int main()                                     
{
int a,b,m,k,n;
double t;

// ####################################################
// prepare output file with date, and parameters
// ####################################################
time_t curtime; struct tm *loctime; curtime = time(NULL); loctime = localtime (&curtime); 
int hour = loctime -> tm_hour;int minute = loctime -> tm_min;int second = loctime -> tm_sec;
int year = 1900+loctime -> tm_year;int month = loctime -> tm_mon;int day  = loctime -> tm_mday;
printf("Date: %d.%d.%.d -- Time: %d:%d:%d  \n",day,month+1,year,hour,minute,second);  

char FILE_NAME[2048+1024];
snprintf(FILE_NAME,2048+1024,"%02d_%02d_%02d_%02d_%02d_%02d_cQED_Spectrum_KAPPA%.5f_GAM_R%.5f_GAM_PH%.5f_DELTA_L%.5f_DELTA_C%.5f.dat",year,month+1,day,hour,minute,second,KAPPA/Mc,GAM_R/Mc,GAM_P/Mc,LASER_DETUNING,CAVITY_DETUNING);
FILE *f_spectrum; f_spectrum=fopen(FILE_NAME,"w");

fprintf(f_spectrum,"#delta_t\t %.5f\n",delta_t); 
fprintf(f_spectrum,"#Mc \t %.5f\n",Mc);
fprintf(f_spectrum,"#KAPPA \t %.5f\n",KAPPA/Mc); //( Mc/2) 
fprintf(f_spectrum,"#OmL \t %.5f\n",OmL/Mc); 
fprintf(f_spectrum,"#OmC \t %.5f\n",OmC/Mc);
fprintf(f_spectrum,"#GAM_R \t %.5f\n",GAM_R/Mc);
fprintf(f_spectrum,"#GAM_P \t %.5f\n",GAM_P/Mc);
fprintf(f_spectrum,"#INC_P \t %.5f\n",INC_P/Mc);
fprintf(f_spectrum,"#N_p \t %i\n",N_p);
fprintf(f_spectrum,"#omega_qd \t %.5f\n",omega_qd);
fprintf(f_spectrum,"#omega_laser \t %.5f\n",omega_laser);
fprintf(f_spectrum,"#omega_cavity \t %.5f\n",omega_cavity);
fprintf(f_spectrum,"#LASER_DETUNING \t %.5f\n",LASER_DETUNING/Mc);
fprintf(f_spectrum,"#CAVITY_DETUNING \t %.5f\n",CAVITY_DETUNING/Mc);
fprintf(f_spectrum,"#Nk \t %i\n",Nk);
fprintf(f_spectrum,"#k_interval \t %.5f\n",k_interval/Mc);
fprintf(f_spectrum,"#delta_k \t %.5f\n",delta_k);
fprintf(f_spectrum,"#SPECTRUM_OUTPUT_INTERVAL \t %.5f\n",SPECTRUM_OUTPUT_INTERVAL/Mc);

// ###############################################################################
// ############################## INITIAL CONDITIONS #############################
// ###############################################################################
complex double *derivates = calloc(2*N_DGL,sizeof(double));
complex double *deriv_stst = calloc(2*N_DGL,sizeof(double));
// Um den Anfangswert zu speichern, mit dem die Zwischen-Anfangswerte berechnet werden
complex double *temp = calloc(2*N_DGL,sizeof(double));
// Die errechneten Zwischenfunktionswerte - die Steigungen
complex double *k1 = calloc(2*N_DGL,sizeof(double));
complex double *k2 = calloc(2*N_DGL,sizeof(double));
        double *OUTPUT = calloc(100,sizeof(double)); 
for (n=0;n<N_DGL;n++) derivates[n] = 0. + I* 0.;
for (n=0;n<100;n++)   OUTPUT[n]    =0.;
// set initial conditions: quantum in the ground state
rho(0,0,0,0) = 1. ; // QD in the Ground State with Zero Photons in the Cavity
OUTPUT[0] = 1.; // Trace=1
// ###########################################################################
// ########################## TIME _DOMAIN SOLUTION ##########################
// ###########################################################################
int i; 
k=0; OUTPUT[99] = 100.;
double progress_per_stst = 0.1;
// integrate in t until steady state
while(OUTPUT[99]>0.0000000001)
{  
 k++;
 t=delta_t*k;
 // ############### 4 VECTOR RUNGE KUTTA ###########################
 calculate_next_time_step(derivates,k1,t            ,OUTPUT); 
 CALCULATE_TEMP(derivates,k1,delta_t*HALFTE,temp,i);
 calculate_next_time_step(temp     ,k2,t+delta_t*0.5,OUTPUT); 
 CALCULATE_TEMP(derivates,k2,delta_t*HALFTE,temp,i);
 ADD_VECTOR(k1,k2,2.0,i);
 calculate_next_time_step(temp     ,k2,t+delta_t*0.5,OUTPUT); 
 CALCULATE_TEMP(derivates,k2,delta_t,temp,i);
 ADD_VECTOR(k1,k2,2.0,i);
 calculate_next_time_step(temp     ,k2,t+delta_t    ,OUTPUT); 
 ADD_VECTOR(k1,k2,1.0,i); 
 ADD_VECTOR(derivates,k1,delta_t*SECHSTEL,i);
 // ############ END OF 4 VECTOR RUNGE KUTTA #######################
 STEADY_STATE_TEST(derivates,deriv_stst,OUTPUT,Zahl_der_DGL);
  if (OUTPUT[99]<progress_per_stst)
    {
     progress_per_stst = progress_per_stst/10.;  
     printf("%.0f - "          ,k*100./N);
     printf("RHO_EE=%.6f - "   ,OUTPUT[1]);
     printf("Drive=%.6f - "   ,(OmL+OmC)/Mc);
     printf("I=%.6f - "        ,OUTPUT[2]);
     printf("II=%.6f - "        ,OUTPUT[3]);
     printf("TRACE=%.10f - "    ,OUTPUT[0]);
     printf("II_TEST=%.10f - "  ,OUTPUT[7]-OUTPUT[3]);
     printf("StSt_TEST=%.10f \n",OUTPUT[99]);
    } 
} // end of time integration
// ########################## TAU_INITIAL_CONDITIONS #########################
// dieser steady state wert wird vom spektrumsignal abgezogen, um den kohaerenten anteil abzuziehen
for (a=0;a<N_DGL;a++) { deriv_stst[a]=derivates[a]; derivates[a]=0.+I*0.;}

// INITIAL VALUES FOR THE CAVITY SPECTRUM
complex double obs_inf = OUTPUT[5]+I*OUTPUT[6];
// dieser steady state wert wird vom spektrumsignal abgezogen, um den kohaerenten anteil
// abzuziehen
double obs_coh   = creal(obs_inf*conj(obs_inf));
OUTPUT[5] = OUTPUT[2]; 
OUTPUT[6] = 0.; 
OUTPUT[15] = 0.; // Photon Density
OUTPUT[16] = 0.; 
//double obs_detuning = CAVITY_DETUNING; //to adjust middle position in spectrum
for(a=0;a<N_l;a++){for(b=0;b<N_l;b++){for(m=0;m<N_p;m++){for(n=0;n<N_p;n++){
// < c^\dg(t) c(t+tau)>  
rho(a,m,b,n) = sqrt(n+1.)*rho_stst(a,m,b,n+1);
}}}}

complex double *correlation     = calloc(2*MAX_STEPS,sizeof(double));
OUTPUT[99] = 100.; // steady state value
k=0; // set k back
// ########################## TAU_DOMAIN SOLUTION ############################
// integrate in tau until steady state 

progress_per_stst = 0.1;
while(OUTPUT[99]>0.0000000001)
{  
 k++;
 t=delta_t*k;
 if (k<MAX_STEPS) 
 {
 correlation[k]     =   OUTPUT[5] +I*OUTPUT[6] -1.*obs_coh;
 // ############### 4 VECTOR RUNGE KUTTA ###########################
 calculate_next_time_step(derivates, k1 , t,OUTPUT); 
 CALCULATE_TEMP(derivates,k1,delta_t*HALFTE,temp,i);
 calculate_next_time_step(temp, k2, t+delta_t*0.5,OUTPUT); 
 CALCULATE_TEMP(derivates,k2,delta_t*HALFTE,temp,i);
 ADD_VECTOR(k1,k2,2.0,i);
 calculate_next_time_step(temp, k2, t+delta_t*0.5,OUTPUT); 
 CALCULATE_TEMP(derivates,k2,delta_t,temp,i);
 ADD_VECTOR(k1,k2,2.0,i);
 calculate_next_time_step(temp, k2, t+delta_t,OUTPUT);  
 ADD_VECTOR(k1,k2,1.0,i);
 ADD_VECTOR(derivates,k1,delta_t*SECHSTEL,i);
 // ############ END OF 4 VECTOR RUNGE KUTTA #######################
 STEADY_STATE_TEST(derivates,deriv_stst,OUTPUT,Zahl_der_DGL);
 if (OUTPUT[99]<progress_per_stst)
    {
     progress_per_stst = progress_per_stst/10.;  
     printf("%.0f - "          ,k*100./N);
     printf("CORRELATION=%.6f - "   ,cabs(correlation[k]));
     printf("TRACE=%.10f - "    ,OUTPUT[0]);
     printf("StSt_TEST=%.10f \n",OUTPUT[99]);
    } 
 }  
 else {printf("ACHTUNG: STEADY NICHT ERREICHT!\n");OUTPUT[99]=-0.10000000001;}
} // end of tau integration
int k_MAX=k; // End of correlation array
double spectrum_inc[Nk]; double spectrum_sum[Nk]; // spectrum vector
// calculate and modify spectrum // substract offset in procedure
calculate_spectrum(correlation,spectrum_inc,spectrum_sum,k_MAX,obs_coh);//,-LASER_DETUNING);
// write into file
for(k=0;k<Nk;k++)
 {
 fprintf(f_spectrum,"%.15f \t %.15f \t %.15f  \t %.15f \n",-(-0.5*k_interval-LASER_DETUNING+delta_k*k)/Mc+CENTER_OF_SPECTRUM/Mc,spectrum_inc[k],spectrum_sum[k],obs_coh);
 }
fprintf(f_spectrum,"\n"); 

// free given arrays
free(temp);free(derivates);free(deriv_stst);free(k1);free(k2);free(OUTPUT);free(correlation);
fclose(f_spectrum);
return 0;
}
