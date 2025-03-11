//===- ParallelCombiningInterfaces.h - Parallel combining op interfaces ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for ops that perform
// parallel combining operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARALLELCOMBININGINTERFACES_H_
#define MLIR_INTERFACES_PARALLELCOMBININGINTERFACES_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace detail {
LogicalResult verifyInParallelOpInterface(Operation *op);
LogicalResult verifyParallelCombiningOpInterface(Operation *op);
} // namespace detail
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/ParallelCombiningInterfaces.h.inc"

#endif // MLIR_INTERFACES_PARALLELCOMBININGINTERFACES_H_
