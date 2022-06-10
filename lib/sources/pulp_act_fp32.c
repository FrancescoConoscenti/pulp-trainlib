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

/**
 * Authors: Davide Nadalini, Leonardo Ravaglia
*/ 

#include "pulp_train_utils_fp32.h"
#include "math.h"

void pulp_relu_fp32_fw_cl(struct blob * input, struct blob * output){

  int dim = input->dim;
  float* inData = input->data;
  float* outData = output->data;

  for (int i = 0; i < dim; i++) {
    outData[i] = inData[i] > 0 ? inData[i] : 0;
  }
}

void pulp_relu_fp32_bw_cl(struct blob * input, struct blob * output){

  int dim = input->dim;
  float* inData = input->data;
  float* inDiff = input->diff;
  float* outDiff = output->diff;

  for (int i = 0; i < dim; i++) {
    inDiff[i] = inData[i] > 0 ? outDiff[i] : 0;
  }
}

void pulp_softmax_fp32_fw_cl(struct blob * input, struct blob * output){

  int dim = input->dim;
  float* inData = input->data;
  float* outData = output->data;
  float sum = 0.0;

  for (int i = 0; i < dim; i++) {
    sum += exp(inData[i]);
  }

  for (int i = 0; i < dim; i++) {
    outData[i] = exp(inData[i])/sum;
  }
}

void pulp_softmax_fp32_bw_cl(struct blob * input, struct blob * output){

  int dim = input->dim;
  float* inDiff = input->diff;
  float* outData = output->data;
  float* outDiff = output->diff;
  float sum = 0.0;

  for (int i = 0; i < dim; i++) {
    inDiff[i] = outDiff[i];
  }
}