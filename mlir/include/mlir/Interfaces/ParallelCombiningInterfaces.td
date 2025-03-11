//===- ParallelCombiningInterfaces.td - Parallel ifaces ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces for ops that perform parallel combining operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_PARALLELCOMBININGINTERFACES
#define MLIR_INTERFACES_PARALLELCOMBININGINTERFACES

include "mlir/IR/OpBase.td"

/// The interfaces defined here work in conjunction to enable parallelized
/// construction of types with value semantics (e.g. tensors). They encompass
/// three classes of operations:
///
///  - A parent iterating op (ParallelIterationOpInterface)
///  - Disjoint updating ops (InParallelOpInterface)
///  - Optional combining op for separating in parallel updating ops from
///    serial code within the parent iterating op.
///
/// The typical structure for these ops looks like the following:
///
/// ```
/// !out_types = tensor<*>, tensor<*>, ...
///
/// // Parent parallel iteration op carries at least one region with shared
/// // outputs
/// %r:n = parallel_iterator (%iter_id) shared_outs(%0, %1, ... : !out_types) {
///   %work:n = // Some parallelized work based on %iter_id
///
///   // Optional combining op carrying all required updates
///   parallel_combine {
///
///     // Set of parallel update ops combining local work into shared region
///     // arguments. Updated values of all in_parallel ops *must* be produced
///     // by a shared_outs block argument to a parallel_iterator op. Note that
///     // the shared_outs are allowed to have other non-parallel users.
///     in_parallel_update %work#0 into %0 : typeof(%0)
///     in_parallel_update %work#1 into %1 : typeof(%1)
///     ...
///     in_parallel_update %work#n into %n : typeof(%n)
///   }
/// } -> !out_types
/// ```
///
/// The reasoning behind this structure is to allow multiple updates to the same
/// object with value semantics without going all the way to memory semantics.
/// The `shared_outs` operands to `in_parallel_update` ops are tied to results
/// of the parent `parallel_iterator`, and the updates are only reflected in the
/// results. For example:
///
/// ```
/// %init = arith.constant dense<1> : tensor<i32>
/// %2 = parallel_iterator shared_outs(%0 = %init : tensor<i32>) {
///   in_parallel_update 0 -> %0
///   %1 = tensor.extract %0 : tensor<i32>
///   print %1
/// } -> tensor<i32>
/// %3 = tensor.extract %2 : tensor<i32>
/// print %3
/// ```
///
/// In the above example, the update to `%0` is not reflected in the subsequent
/// read. Thus this should print `1` followed by `0`. The validity of
/// overlapping updates depends on the updating op, however they may never
/// assume an iteration order. For example:
///
/// ```
/// // Undefined as multiple threads update the same value.
/// func @cross_thread_collision() {
///   %init = tensor.empty() : tensor<i32>
///   %2 = parallel_iterator (%id : i32) for [0, 1] shared_outs(%0 = %init : tensor<i32>) {
///     in_parallel_update %id -> %0
///   } -> tensor<i32>
///   %3 = tensor.extract %2 : tensor<i32>
///   print %3
/// }
///
/// func @atomic_sum() {
///   %init = arith.constant dense<0> : tensor<i32>
///   %2 = parallel_iterator (%id : i32) for [0, 2] shared_outs(%0 = %init : tensor<i32>) {
///     in_parallel_atomic_accumulate %id -> %0
///   } -> tensor<i32>
///   %3 = tensor.extract %2 : tensor<i32>
///
///   // Prints 0 + 1 + 2 = 3
///   print %3
/// }
///
/// // Undefined if the updates are not atomic.
/// func @non_atomic_intra_thread_collision() {
///   %init = tensor.empty() : tensor<i32>
///   %2 = parallel_iterator shared_outs(%0 = %init : tensor<i32>) {
///     in_parallel_update 1 -> %0
///     in_parallel_update 2 -> %0
///   } -> tensor<i32>
///   %3 = tensor.extract %2 : tensor<i32>
///   print %3
/// }
/// ```
///
/// Multiple regions/blocks in the iterator, multiple updates per shared_out,
/// and control flow within the iterator are all legal. For example, a MIMT
/// like op could be written like follows:
///
/// ```
/// %init = arith.constant dense<1> : tensor<2xi32>
/// %2 = mimt_parallel_iterator shared_outs(%init: tensor<2xi32>) {
///   ^bb0 shared_outs(%0: tensor<2xi32>):
///     in_parallel_update %0[0] = 0
///   ^bb1 shared_outs(%1: tensor<2xi32>):
///     in_parallel_update %1[1] = 1
/// } -> tensor<2xi32>
/// ```

def InParallelOpInterface : OpInterface<"InParallelOpInterface"> {
  let description = [{
    An in_parallel op is an operation that inserts into a shared tensor in
    conjunction with a parent combining and iterating op.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the list of values updated by this op.
      }],
      /*retTy=*/"::mlir::MutableOperandRange",
      /*methodName=*/"getUpdatedDestinations",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the iterating parent for this op.
      }],
      /*retTy=*/"::mlir::Operation*",
      /*methodName=*/"getIteratingParent",
      /*args=*/(ins)
    >,
  ];
  let verify = [{
    return ::mlir::detail::verifyInParallelOpInterface($_op);
  }];
}

def ParallelCombiningOpInterface : OpInterface<"ParallelCombiningOpInterface"> {
  let description = [{
    A parallel combining op is an op with a region.

    This is useful as a terminator to parallel operations that iterate over
    some set and return tensors while avoiding tight coupling between the
    iterating op, the combining op and the individual subtensor producing ops.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return `idx`^th result of the parent operation.
      }],
      /*retTy=*/"::mlir::OpResult",
      /*methodName=*/"getParentResult",
      /*args=*/(ins "int64_t":$idx),
      /*methodBody=*/[{
        return $_op.getParentResult(idx);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the contained ops that yield subvalues that this op combines to
        yield to its parent.
      }],
      /*retTy=*/"::llvm::iterator_range<Block::iterator>",
      /*methodName=*/"getYieldingOps",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getYieldingOps();
      }]
    >,
  ];
  // TODO: Single region single block interface on interfaces ?
  let verify = [{
    return ::mlir::detail::verifyParallelCombiningOpInterface($_op);
  }];
}

def ParallelIterationOpInterface : OpInterface<"ParallelIterationOpInterface"> {
  let description = [{
    A parallel iteration op is an op that performs computation in parallel.
    Such operations are able to yield tensors constructed in parallel by
    its body.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return the set of InParallel ops updating the specified result.
      }],
      /*retTy=*/"SmallVector<::mlir::Operation*>",
      /*methodName=*/"getUpdatingOps",
      /*args=*/(ins "::mlir::OpResult":$result)
    >,
    InterfaceMethod<[{
        Return the region "shared_outs" (block arguments) that correspond to the
        "init" operands. At the moment only a single block + region is
        supported.
      }],
      /*retTy=*/"::mlir::Block::BlockArgListType",
      /*methodName=*/"getRegionSharedOuts",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::Block::BlockArgListType();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the result tied to the given inner shared out.
      }],
      /*retTy=*/"::mlir::OpResult",
      /*methodName=*/"getTiedOpResult",
      /*args=*/(ins "::mlir::BlockArgument":$shared_out)
    >,
  ];
}

#endif // MLIR_INTERFACES_PARALLELCOMBININGINTERFACES
