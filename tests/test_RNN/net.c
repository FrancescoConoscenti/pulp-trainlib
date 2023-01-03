/*
 * Copyright (C) 2021-2022 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pulp_train.h"

#include "RNN-data.h"
#include "stats.h"

#include "net.h"

// DATA DEFINITION


// LINEAR
PI_L1 struct blob layer0_in, layer0_wgt_in, layer0_wgt_h, layer0_state, layer0_out;
// Memory occupation counter
PI_L2 int L1_memocc_bytes = 0;
PI_L2 int L2_memocc_bytes = 0;

#ifdef FORWARD
PI_L1 float l0_in[Tin_l0*RICORS];
PI_L1 float l0_ker_in[Tin_l0*Tout_l0];
PI_L1 float l0_ker_h[Tout_l0*Tout_l0]; 
PI_L1 float l0_state[Tout_l0*(RICORS+1)]; 
PI_L1 float l0_out[Tout_l0]; 
PI_L1 int Ric=RICORS;
#endif

#ifdef BACKWARD
PI_L1 float l0_in[Tin_l0*RICORS];
PI_L1 float l0_ker_in[Tin_l0*Tout_l0];
PI_L1 float l0_ker_h[Tout_l0*Tout_l0]; 
PI_L1 float l0_ker_in_diff[Tker_l0];
PI_L1 float l0_ker_h_diff[Tout_l0*Tout_l0];
PI_L1 float l0_state[Tout_l0*(RICORS+1)]; 
PI_L1 float l0_state_diff[Tout_l0*(RICORS+1)]; 
PI_L1 float l0_out[Tout_l0]; 
PI_L1 float l0_out_diff[Tout_l0]; 
PI_L1 int Ric=RICORS;
PI_L1 int TimeStep;
#endif



#ifdef FORWARD
static inline void tensor_init() 
{
  for (int i=0; i<Tin_l0*RICORS; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tin_l0*Tout_l0; i++)       l0_ker_in[i] = L0_WEIGHTS_params_INPUT[i]; 
  for (int i=0; i<Tout_l0*Tout_l0; i++)       l0_ker_h[i] = L0_WEIGHTS_params_HIDDEN[i];
  for (int i=0; i<Tout_l0*(RICORS+1); i++)       l0_state[i] = 0.0f; 
  for (int i=0; i<Tout_l0; i++)       l0_out[i] = 0.0f; 
}

static inline void connect_blobs() 
{
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_l0;

  layer0_wgt_in.data = l0_ker_in;
  layer0_wgt_in.dim = Tker_l0;

  layer0_wgt_h.data = l0_ker_h;
  layer0_wgt_h.dim = Tout_l0*Tout_l0;

  layer0_state.data = l0_state;
  layer0_state.dim = Tout_l0*RICORS+1;

  layer0_out.data = l0_out;
  layer0_out.dim = Tout_l0;
}

static inline void compute_memory_occupation(){
  // Input
  L1_memocc_bytes += Tin_l0*RICORS *sizeof(float);
  // Kernel input
  L1_memocc_bytes += Tker_l0*sizeof(float); 
  // Kernel state
  L1_memocc_bytes += Tout_l0*Tout_l0*sizeof(float);
  //hidden states
  L1_memocc_bytes += Tout_l0*(RICORS+1)*sizeof(float);
  // Output
  L1_memocc_bytes += Tout_l0*sizeof(float);

  // Input data
  L2_memocc_bytes += L0_IN_CH * RICORS *sizeof(float);
  // Weights input
  L2_memocc_bytes += L0_WEIGHTS_input*sizeof(float);
  // Weights state
  L2_memocc_bytes += L0_WEIGHTS_hidden*sizeof(float);
  // States
  L2_memocc_bytes += L0_OUT_CH*(RICORS+1)*sizeof(float);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);

}
#endif


#ifdef BACKWARD
static inline void tensor_init() 
{
  //backward grad
  for (int i=0; i<Tin_l0*RICORS; i++)        l0_in[i] = INPUT_VECTOR[i];
  for (int i=0; i<Tker_l0; i++)       l0_ker_in_diff[i] = 0.0f;           //initialization to zero, then it is overwritten the result in the function
  for (int i=0; i<Tout_l0*Tout_l0; i++)       l0_ker_h_diff[i] = 0.0f;
  for (int i=0; i<Tout_l0*(RICORS+1); i++)       l0_state[i] = L0_HIDDEN_STATE[i];

  //backward error
  for (int i=0; i<Tout_l0*(RICORS+1); i++)        l0_state_diff[i] = 0.0f;
  for (int i=0; i<Tin_l0*Tout_l0; i++)       l0_ker_in[i] = L0_WEIGHTS_params_INPUT[i];
  for (int i=0; i<Tout_l0*Tout_l0; i++)       l0_ker_h[i] = L0_WEIGHTS_params_HIDDEN[i]; 

  //both
  for (int i=0; i<Tout_l0; i++)       l0_out_diff[i] = L0_OUT_GRAD[i];  
  for (int i=0; i<Tout_l0; i++)       l0_out[i] = L0_OUT_FW[i];
}

static inline void connect_blobs() 
{
  //backward grad
  layer0_in.data = l0_in;
  layer0_in.dim = Tin_l0;

  layer0_wgt_in.diff = l0_ker_in_diff;
  layer0_wgt_in.dim = Tout_l0*Tin_l0;

  layer0_wgt_h.diff = l0_ker_h_diff;
  layer0_wgt_h.dim = Tout_l0*Tout_l0;

  layer0_state.data = l0_state;
  layer0_state.dim = Tout_l0*(RICORS+1);


//backward error
  layer0_state.diff = l0_state_diff;

  layer0_wgt_in.data = l0_ker_in;
  layer0_wgt_in.dim = Tker_l0;

  layer0_wgt_h.data = l0_ker_h;
  layer0_wgt_h.dim = Tout_l0*Tout_l0;

//both
  layer0_out.data = l0_out;
  layer0_out.diff = l0_out_diff;
  layer0_out.dim = Tout_l0;  
}

static inline void compute_memory_occupation(){
  // Input
  L1_memocc_bytes += Tin_l0*RICORS*sizeof(float);
  // Kernel input grad
  L1_memocc_bytes += Tker_l0*sizeof(float); 
  // Kernel state grad
  L1_memocc_bytes += Tout_l0*Tout_l0*sizeof(float);
  //hidden states 
  L1_memocc_bytes += Tout_l0*(RICORS+1)*sizeof(float);
  // Output grad
  L1_memocc_bytes += Tout_l0*sizeof(float);

  // Input data
  L2_memocc_bytes += L0_IN_CH * RICORS *sizeof(float);
  // Weights input
  L2_memocc_bytes += L0_WEIGHTS_input*sizeof(float);
  // Weights state
  L2_memocc_bytes += L0_WEIGHTS_hidden*sizeof(float);
  // States
  L2_memocc_bytes += L0_OUT_CH*(RICORS+1)*sizeof(float);
  // Output
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
  // Output gradient
  L2_memocc_bytes += L0_OUT_CH*sizeof(float);
  // Weight input gradient
  L2_memocc_bytes += L0_WEIGHTS_input*sizeof(float);
  // Weight state gradient
  L2_memocc_bytes += L0_WEIGHTS_hidden*sizeof(float);
  // States gradient
  L2_memocc_bytes += L0_OUT_CH*(RICORS+1)*sizeof(float);
  // Input gradient
  L2_memocc_bytes += L0_IN_CH*sizeof(float);
}
#endif



static inline void net_forward(){
  /**  FORWARD FC #1   **/
  #ifdef FORWARD
  pulp_RNN_fp32_fw_cl(&layer0_in, Ric, &layer0_wgt_in, &layer0_wgt_h, &layer0_state, &layer0_out);
  #endif
}

static inline void compare_tensors(float *A, float *B, int length){

  double mean_err_rel = 0.0;
  float diff = 0.0f;
  float den = 0.000001f;

  for(int i=0; i<length; i++){
     if (A[i]>B[i] && A[i]>0.0f){
        diff = A[i]-B[i];
        if (diff>0) diff = diff;
        else diff=-diff;
        if (A[i]>0) den = A[i];
        else den = -A[i]; // missing A = 0
        mean_err_rel = mean_err_rel + (diff / den)/length;
     }
     else{
       diff = A[i]-B[i];
       if (diff>0) diff = diff;
       else diff=-diff;
       if (A[i]>0) den = A[i];
       else den = -A[i];
       mean_err_rel = mean_err_rel + (diff / den);
       
     }
  }

  mean_err_rel = mean_err_rel/length;

  for(int i=0; i<length; i++){
    if(A[i]==0 && B[i]==0)
    {
      mean_err_rel=0;
    }
  }

  if (mean_err_rel<ERROR_TOLERANCE) printf(">>>TENSOR MATCHING!\n ");
  else printf(">>>TENSOR NOT MATCHING!\n Err=%e \n",mean_err_rel);

}

// Elementwise checker
int check_tensor(float * tensor_out, float * tensor_ref, int size){

    int error_flag = 0;
    for (int i=0; i<size; i++) {
        if ( ABS(tensor_out[i]-tensor_ref[i]) > CHECK_TOLERANCE ) {
            if (error_flag == 0) printf("\n");
            printf("Error at index: %d   (Ideal = %.16f [HEX: %#x]  vs  Actual = %.16f [HEX: %#x])\n", i, 
                tensor_ref[i], *(unsigned int*) &tensor_ref[i], tensor_out[i], *(unsigned int*) &tensor_out[i]);
            error_flag = 1;
        }
    }
    return error_flag;
}



static inline void train(){

  
  pi_perf_conf((1<<PI_PERF_CYCLES) | (1<<PI_PERF_INSTR)  | (1<<PI_PERF_LD)  | (1<<PI_PERF_ACTIVE_CYCLES) );
  pi_perf_stop();
  pi_perf_reset();
  pi_perf_start();
 


  #ifdef PROF_FWD
  //printf("\nForward stats\n");
  START_STATS();
  #endif

  #ifdef FORWARD
  pulp_RNN_fp32_fw_cl(&layer0_in, Ric, &layer0_wgt_in, &layer0_wgt_h, &layer0_state, &layer0_out);
  #endif

  #ifdef PROF_FWD
  STOP_STATS();
  #endif

  #ifdef PROF_BCKWD
  //printf("\nBackward stats\n");
  START_STATS();
  #endif

  #ifdef BACKWARD
  pulp_RNN_fp32_bw_cl(&layer0_in, Ric, &layer0_wgt_in, &layer0_wgt_h, &layer0_state, &layer0_out);
  #endif

  #ifdef PROF_BCKWD
  STOP_STATS();
  #endif


  pi_perf_stop();

  int instr_count=pi_perf_read (PI_PERF_INSTR);
  int cycles_count=pi_perf_read (PI_PERF_CYCLES);
  int load_count=pi_perf_read (PI_PERF_LD);
  int active_cycles_count=pi_perf_read (PI_PERF_ACTIVE_CYCLES);

  printf("performance");
  printf("\n%d \n", cycles_count);
  printf("%d\n", instr_count);
  printf("%d\n", active_cycles_count);
  printf("%d\n", load_count);
  printf("%f\n", (float)cycles_count/instr_count);
  


  #ifdef FORWARD
  printf("FORWARD CHECK: \n");
  compare_tensors(l0_out, L0_OUT_FW, Tout_l0);
  check_tensor(l0_out, L0_OUT_FW, Tout_l0);
  #endif


  #ifdef BACKWARD
  printf("FINAL WEIGHTS GRADIENT CHECK: \n");
  compare_tensors(l0_ker_in_diff, L0_WEIGHT_params_INPUT_GRAD_FINAL, Tker_l0);  
  check_tensor(l0_ker_in_diff, L0_WEIGHT_params_INPUT_GRAD_FINAL, Tker_l0);

  compare_tensors(l0_ker_h_diff, L0_WEIGHT_params_HIDDEN_GRAD_FINAL, Tout_l0*Tout_l0);     
  check_tensor(l0_ker_h_diff, L0_WEIGHT_params_HIDDEN_GRAD_FINAL, Tout_l0*Tout_l0);

  printf("FINAL STATE GRADIENT CHECK: \n");
  compare_tensors(&l0_state_diff[Tout_l0], L0_STATE_GRAD_FINAL, Tout_l0);     //ccomparing GM result and actual result
  check_tensor(&l0_state_diff[Tout_l0], L0_STATE_GRAD_FINAL, Tout_l0);        // L0_STATE_GRAD is "grad_over_time" of numphy model
  #endif 


}


// Most important function: it connects each passage to step the net and perform training
void net_step()
{
  #ifdef PROF_NET
  INIT_STATS();
  PRE_START_STATS();
  #endif

  #ifdef MEMOCC_COMP
  compute_memory_occupation();
  printf("\nL1 memory occupation: %d bytes.", L1_memocc_bytes);
  printf("\nL2 memory occupation: %d bytes.\n", L2_memocc_bytes);
  #endif

  tensor_init();

  connect_blobs();

  train();

  return;
}
