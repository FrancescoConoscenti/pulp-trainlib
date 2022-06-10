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

#include "net_args.h"

// User profiling flags

#if (defined(STANDARD) || defined(DW)) && !defined(DEBUG)
#define PROF_MM
#endif

// Tensor checksum definition
#ifdef FLOAT32
#define CHECK_TOLERANCE 1e-3
#define ERROR_TOLERANCE 0.01
#endif
#ifdef FLOAT16
#define CHECK_TOLERANCE 1e0
#define ERROR_TOLERANCE 0.05
#endif
#ifdef BFLOAT16
#define CHECK_TOLERANCE 1e-3
#define ERROR_TOLERANCE 0.05
#endif

// PULP DEFINES
#define STACK_SIZE      4096
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

void net_step();
