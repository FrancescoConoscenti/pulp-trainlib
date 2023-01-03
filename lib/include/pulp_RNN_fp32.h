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
 * Linear layer training functions, grouped into FW and BW
*/


// FORWARD FUNCTIONS

/**
 * @brief Forward pass function, forked on PULP cluster.
 * @param input  input column vector for the linear layer
 * @param RICORS  num input values / in_size
 * @param coeffWx  weight input matrix 
 * @param coeffWa  weight state matrix
 * @param state  	vector of vectors of each hidden state
 * @param output  categorical output for the linear layer
 */
void pulp_RNN_fp32_fw_cl(
	struct blob * input, 
	int RICORS,
	struct blob * coeffWx,
	struct blob * coeffWa,
	struct blob * state,
	struct blob * output	
);


// BACKWARD FUNCTIONS

/**
 * @brief Backward pass function, which internally calls both weight gradient and input gradient calculation
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param RICORS  num input values / in_size
 * @param coeffWx  weight input matrix 
 * @param coeffWa  weight state matrix
 * @param state  	vector of vectors of each hidden state
 * @param output  categorical output for the linear layer (from forward perspective)
 */
void pulp_RNN_fp32_bw_cl(
	struct blob * input, 
	int RICORS,
	struct blob * coeffWx,
	struct blob * coeffWa,
	struct blob * state, 
	struct blob * output
);

/**
 * @brief Backward pass function which computes weight's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param RICORS  num input values / in_size
 * @param TimeStep  wich timeStep we are computing
 * @param coeffWx  weight input matrix 
 * @param coeffWa  weight state matrix
 * @param state  	vector of vectors of each hidden state
 * @param output  categorical output for the linear layer (from forward perspective)
 */
void pulp_RNN_fp32_bw_param_grads_cl(
	struct blob * input, 
	int RICORS,
	int TimeStep,
	struct blob * coeffWx,
	struct blob * coeffWa,
	struct blob * state, 
	struct blob * output
);

/**
 * @brief Backward pass function which computes input's gradient only
 * @param input  input column vector for the linear layer (from forward perspective)
 * @param RICORS  num input values / in_size
 * @param TimeStep  wich timeStep we are computing
 * @param coeffWx  weight input matrix 
 * @param coeffWa  weight state matrix
 * @param state  	vector of vectors of each hidden state
 * @param output  categorical output for the linear layer (from forward perspective)
 */
void pulp_RNN_fp32_bw_state_grads_cl(
	struct blob * input, 
	int RICORS,
	int TimeStep,
	struct blob * coeffWx,
	struct blob * coeffWa,
	struct blob * state, 
	struct blob * output
);