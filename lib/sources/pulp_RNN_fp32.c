/*
 * Copyright (C) 2020 University of Bologna
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 * Authors: Francesco Conoscenti (francesco.conoscenti@studio.unibo.it)
 */
#include "pmsis.h"
#include "fasthyperbolic.h"
#include "/home/pulp-user/work/APAI-LAB01-PULP_Embedded/pulp-trainlib-pulp-trainlib/tests/test_RNN_TESTING/net.h"
#include "/home/pulp-user/work/APAI-LAB01-PULP_Embedded/pulp-trainlib-pulp-trainlib/lib/include/pulp_matmul_fp32.h"
#include "/home/pulp-user/work/APAI-LAB01-PULP_Embedded/pulp-trainlib-pulp-trainlib/lib/include/pulp_train_utils_fp32.h"


//FORWARD
void pulp_RNN_fp32_fw_cl(struct blob * input, int RICORS,  struct blob * coeffWx, struct blob * coeffWa, struct blob * state, struct blob * output)
{


  float *coeffDataWx = coeffWx->data;
  float *coeffDataWa = coeffWa->data;
  float *outData = output->data;  
  float *inputData = input->data;  // it is a pointer to a RICORS input vector
  float *inputDataN= input->data;
  float *prev_state= state->data;
  //temporary variables
  float outData1[output->dim];
  float outData2[output->dim];


  for(int i=0;i<RICORS;i++){

    // every cycle it is a different input vector, a column of the input matrix
    inputDataN=&inputData[i*(input->dim)];

    //matmul setup 1
    struct matMul_args matMul_args1;
    matMul_args1.A = coeffDataWx;
    matMul_args1.B = inputDataN; 
    matMul_args1.C = outData1;
    matMul_args1.N = output->dim;
    matMul_args1.K = input->dim;
    matMul_args1.M = 1;
    matMul_args1.trans_B = 0;

    #ifdef DEBUG
    printf("\ninputDataN: \n");
    for (int j=0; j<input->dim; j++){
      printf("%4.2e ", matMul_args1.B[j]);
    }
    printf("\n");

    printf("\nWx: %d %d\n",matMul_args1.N ,matMul_args1.K);
    for (int j=0; j<matMul_args1.N*matMul_args1.K; j++){
     if(!(j%(matMul_args1.K))) printf("\n");
      printf("%4.2e ",  matMul_args1.A[j]);
    }
    printf("\n");
    #endif
 
    pi_cl_team_fork(NUM_CORES,  mm, &matMul_args1);
  

    #ifdef DEBUG
    printf("\noutData1: \n");
    for (int j=0; j<output->dim; j++){
      printf("%4.2e ", outData1[j]);
    }
    printf("\n");
    #endif
  

    //matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = coeffDataWa;
    matMul_args2.B = &prev_state[i*output->dim];   //ogni ciclo il precedente risultato
    matMul_args2.C = outData2;   //var temporanea
    matMul_args2.N = output->dim;
    matMul_args2.K = output->dim;
    matMul_args2.M = 1;
    matMul_args2.trans_B = 0;

 

    #ifdef DEBUG
    printf("\nprev_state: \n");
    for (int j=0; j<output->dim; j++){
      printf("%4.2e ", prev_state[i*output->dim+j]);
    }
    printf("\n");

    printf("\nWa: %d %d\n",matMul_args2.N ,matMul_args2.K);
    for (int j=0; j<matMul_args2.N*matMul_args2.K; j++){
    if(!(j%(matMul_args2.K))) printf("\n");
      printf("%4.2e ",  matMul_args2.A[j]);
    }
    printf("\n");
    #endif


    pi_cl_team_fork(NUM_CORES, mm, &matMul_args2);


    #ifdef DEBUG
    printf("\noutData2: \n");
    for (int j=0; j<output->dim; j++){
      printf("%4.2e ", outData2[j]);
    }
    printf("\n");
    #endif

    #ifdef DEBUG
    printf("\nactual_state: \n");
    #endif


    //output with tanh parallelized
    for (int j=0; j<output->dim; j++){
      outData1[j]=outData1[j]+outData2[j];
    }

    struct tanh_args tanh_arg;
    tanh_arg.input = outData1;
    tanh_arg.dim = output->dim;
    tanh_arg.output = &prev_state[((i+1)*(output->dim))];// in place

    pi_cl_team_fork(NUM_CORES, tanh_prll, &tanh_arg);

  

  }

  //use the output of the last ricorsion as a final output
  for (int j=0; j<output->dim; j++){
    outData[j]=prev_state[((RICORS)*output->dim)+j];
  }


  #ifdef DEBUG
  printf("\nLinear OutData: \n");
  for (int j=0; j<output->dim; j++){
    printf("%4.2e ", outData[j]);
  }
  #endif

  #ifdef DEBUG
  printf("\nALL HIDDEN STATE: \n");
  for (int j=0; j<output->dim*(RICORS+1); j++){
      printf("%4.2e ", state->data[j]);
    }
  printf("\n");
  printf("\n");
  #endif

}



//BACWARD
void pulp_RNN_fp32_bw_cl(struct blob * input, int RICORS, struct blob * coeffWx,  struct blob * coeffWa, struct blob * state, struct blob * output) 
{


  float *coeffDataWx = coeffWx->data;
  float *coeffDataWa = coeffWa->data;
  float *inData = input->data;
  float *outData = output->data;
  float *coeffDiffWx = coeffWx->diff;
  float *coeffDiffWa = coeffWa->diff;

  float *outDiff = output->diff;  

  float *inDiff = input->diff;
  float *hiddState = state->data;
  float *hiddStateDiff = state->diff;
  //temporary variables
  float grad[output->dim];
  float grad_step_Wx[coeffWx->dim];
  float grad_step_Wa[coeffWa->dim];
  float dnext[output->dim];
  int i;


  #ifdef DEBUG
  printf("\nHIDDEN STATE\n");
  for(int i=0; i<(output->dim*(RICORS+1)); i++){
    if(!(i%(output->dim))) printf("\n");
      printf("%4.2e  ",hiddState[i]);
  }
  printf("\n");

  printf("\noutData\n");
  for(int i=0; i<output->dim; i++)
    printf("%4.2e  ",outData[i]);
  printf("\n");

  printf("\nLinear outDiff\n");
  for(int i=0; i<output->dim; i++)
    printf("%4.2e  ", outDiff[i]);
  printf("\n");
  #endif


  for(int timestep=RICORS;timestep>0;timestep--){ //unfolding the net

    // if it is at first timestep use outDiff otherwise use the previous hiddenState derivative
    if(timestep!=RICORS){ 
      for(i=0; i<output->dim; i++)
        grad[i]=(1-(hiddState[timestep*(output->dim) + i] * hiddState[timestep*(output->dim) + i]))  *  hiddStateDiff[(timestep+1)*(output->dim) + i];
    }  

    else{
      for( i=0; i<output->dim; i++)
        grad[i]=(1-(hiddState[timestep*(output->dim) + i] * hiddState[timestep*(output->dim) + i])) *  outDiff[i];
    }


    // matmul setup 1
    struct matMul_args matMul_args1;
    matMul_args1.A = grad; // it is a vector
    matMul_args1.B = &inData[(timestep-1)*input->dim]; //input at the current timestep
    matMul_args1.C = grad_step_Wx;
    matMul_args1.N = output->dim;
    matMul_args1.K = 1;
    matMul_args1.M = input->dim;
    matMul_args1.trans_B = 0;


    #ifdef DEBUG
    printf("\ngrad \n");
    for (int i=0; i<output->dim; i++){
      printf("%4.2e  ", grad[i]);
    }
    printf("\n");

    printf("\nlast in data\n");
    for (int i=0; i<input->dim; i++){
      printf("%4.2e  ", matMul_args1.B[i]);
    }
    printf("\n");
    #endif

  
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args1);


    #ifdef DEBUG
    printf("\nLinear coeffDiffWx step ");
    for (int i=0; i<input->dim*output->dim; i++){
      if(!(i%(input->dim))) printf("\n");
      printf("%4.2e (i=%d)", matMul_args1.C[i], i);
    }
    printf("\n");
    #endif 


    //update of the gradient Wx parallelized
    struct update_weight_args update_weight_arg1;
    update_weight_arg1.accum = coeffWx->diff;
    update_weight_arg1.grad = grad_step_Wx;
    update_weight_arg1.dim = coeffWx->dim;
 

    pi_cl_team_fork(NUM_CORES, update_weight_prll, &update_weight_arg1);


    #ifdef DEBUG
    printf("\nLinear coeffDiffWx ");
    for (int i=0; i<input->dim*output->dim; i++){
      if(!(i%(input->dim))) printf("\n");
      printf("%4.2e (i=%d)", coeffDiffWx[i], i);
    }
    printf("\n");
    #endif


    // matmul setup 2
    struct matMul_args matMul_args2;
    matMul_args2.A = grad; // vector
    matMul_args2.B = &hiddState[(timestep-1)*output->dim]; // hidden_state of the second-last timestep
    matMul_args2.C = grad_step_Wa;
    matMul_args2.N = output->dim;
    matMul_args2.K = 1;
    matMul_args2.M = output->dim;
    matMul_args2.trans_B = 0;
  
    pi_cl_team_fork(NUM_CORES, mm_unroll_4x1, &matMul_args2);



    #ifdef DEBUG
    printf("\nLinear coeffDiffWa step ");
    for (int i=0; i<output->dim*output->dim; i++){
      if(!(i%(output->dim))) printf("\n");
      printf("%4.2e (i=%d)", matMul_args2.C[i], i);
    }
    printf("\n");
    #endif


    //update of the gradient Wa parallelized
    struct update_weight_args update_weight_arg2;
    update_weight_arg2.accum = coeffWa->diff;
    update_weight_arg2.grad = grad_step_Wa;
    update_weight_arg2.dim = coeffWa->dim;

    pi_cl_team_fork(NUM_CORES, update_weight_prll, &update_weight_arg2);



    #ifdef DEBUG
    printf("\ngrad \n");
    for (int i=0; i<output->dim; i++){
      printf("%4.2e  ", matMul_args2.A[i]);
    }
    printf("\n");

    printf("\nlast hidden state\n");
    for (int i=0; i<output->dim; i++){
      printf("%4.2e  ",matMul_args2.B[i]);
    }
    printf("\n");

    printf("\nLinear coeffDiffWa ");
    for (int i=0; i<output->dim*output->dim; i++){
      if(!(i%(output->dim))) printf("\n");
      printf("%4.2e (i=%d)", coeffDiffWa[i], i);
    }
    printf("\n");

    #endif


    // the derivative of tanh depends if is the first timestep (use outData) or not (use last hiddenState)
    if(timestep!=RICORS){
      for( i=0; i<output->dim; i++)
        dnext[i]=(1-(hiddState[(timestep)*output->dim+i]*hiddState[(timestep)*output->dim+i]));
    }
    
    else{
      for( i=0; i<output->dim; i++)
        dnext[i]=(1-(outData[i]*outData[i]));
    }


  
    #ifdef DEBUG
    printf("\nSTATE GRADS\n");
    printf("\nderivative of output or next hiddStateData \n");
    for(int i=0; i<output->dim; i++)
      printf("%4.2e ", dnext[i]);
    printf("\n");
    #endif

    
    // matmul setup 3
    struct matMul_args matMul_args3;
    matMul_args3.A = output->diff; // vector
    matMul_args3.B = coeffDataWa;// matrix
    matMul_args3.C = &hiddStateDiff[timestep*output->dim];
    matMul_args3.N = 1;
    matMul_args3.K = output->dim;
    matMul_args3.M = output->dim;
    matMul_args3.trans_B = 1;

    if(timestep!=RICORS)
      matMul_args3.A = &hiddStateDiff[(timestep+1)*output->dim];


    pi_cl_team_fork(NUM_CORES, mm_M_unroll_4x1, &matMul_args3);

 
    #ifdef DEBUG
    printf("\noutputDiff or next hiddStateDiff\n");
    for (int i=0; i<output->dim; i++){
      printf("%4.2e  ", matMul_args1.A[i]);
    }
    printf("\n");

    printf("\nWa\n");
    for (int i=0; i<output->dim*output->dim; i++){
      if(!(i%(output->dim))) printf("\n");
        printf("%4.2e  ", matMul_args1.B[i]);
    }
    printf("\n");

    printf("\nTemporary result\n");
    for (int i=0; i<output->dim; i++){
      printf("%4.2e  ", matMul_args1.C[i]);
    }
    printf("\n");
    #endif

    //calculate derivative of the hiddenState with the derivative of the tanh (dnext)
    for(i=0; i<output->dim; i++){
      hiddStateDiff[timestep*output->dim+i]=dnext[i]*hiddStateDiff[timestep*output->dim+i];
    }


    #ifdef DEBUG
    printf("\nHidden state diff\n");
    for(i=0; i<output->dim; i++){
    printf("%4.2e ", hiddStateDiff[TimeStep*output->dim+i]);
    }
    #endif

  }
  

}




void tanh_prll(void * args){

  struct tanh_args* args_tanh=(struct tanh_args *) args;

  const int blockSize=(args_tanh->dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > args_tanh->dim ? args_tanh->dim : start+blockSize;

  for(int i=start;i<stop;i++){

    args_tanh->output[i]=fastertanh(args_tanh->input[i]);
  }

}


void update_weight_prll(void * args){

  struct update_weight_args* args_update_weight=(struct update_weight_args *) args;

  const int blockSize=(args_update_weight->dim+NUM_CORES-1)/NUM_CORES;
  const int start = pi_core_id()*blockSize;
  const int stop = start + blockSize > args_update_weight->dim ? args_update_weight->dim : start+blockSize;

  for(int i=start;i<stop;i++){

    args_update_weight->accum[i]= args_update_weight->accum[i] + args_update_weight->grad[i] ;
  }

}
