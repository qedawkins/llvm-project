//===- ParallelCombiningInterfaces.cpp - Interfaces for parallel combining ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/ParallelCombiningInterfaces.h"
#include "llvm/ADT/DenseSet.h"

using namespace mlir;

/// Include the definitions of the interfaces.
#include "mlir/Interfaces/ParallelCombiningInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// InParallelOpInterface
//===----------------------------------------------------------------------===//

// TODO: Catch-22 with interface methods used to verify means methods can't
// assume the impl is valid.
LogicalResult mlir::detail::verifyInParallelOpInterface(Operation *op) {
  auto inParallel = cast<InParallelOpInterface>(op);
  auto parent =
      dyn_cast<ParallelIterationOpInterface>(inParallel.getIteratingParent());
  if (!parent) {
    return op->emitError(
        "in_parallel interface op not wrapped by parallel iteration op");
  }

  llvm::SmallDenseSet<BlockArgument> sharedOutSet(
      parent.getRegionSharedOuts().begin(), parent.getRegionSharedOuts().end());
  for (OpOperand &updatedValue : inParallel.getUpdatedDestinations()) {
    auto bbArg = dyn_cast<BlockArgument>(updatedValue.get());
    if (!bbArg) {
      return op->emitError("updating a non block argument");
    }
    if (!sharedOutSet.contains(bbArg)) {
      return op->emitError("updated value not produced by iterating parent");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ParallelCombiningOpInterface
//===----------------------------------------------------------------------===//

// TODO: Single region single block interface on interfaces ?
LogicalResult mlir::detail::verifyParallelCombiningOpInterface(Operation *op) {
  if (op->getNumRegions() != 1)
    return op->emitError("expected single region op");
  if (!op->getRegion(0).hasOneBlock())
    return op->emitError("expected single block op region");
  for (Operation &child : *op->getRegion(0).getBlocks().begin())
    if (!isa<InParallelOpInterface>(&child))
      return op->emitError("expected only in_parallel interface ops");
  return success();
}
