//===- ConvertConv2DToImg2Col.cpp - im2col implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <utility>

#define DEBUG_TYPE "convert-to-img2col"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir::linalg::detail;

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// Im2Col Utilities
//===----------------------------------------------------------------------===//

static bool hasAllOneValues(DenseIntElementsAttr attr) {
  return llvm::all_of(
      attr, [](APInt element) { return element.getSExtValue() == 1; });
}

static Value createAdd(Location loc, Value x, Value y, OpBuilder &builder) {
  bool isInt = x.getType().isa<IntegerType>();
  if (isInt)
    return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, Type accType,
                       OpBuilder &builder) {
  // Linalg named ops specify signed extend for named ops.
  Value xConvert =
      convertScalarToDtype(builder, loc, x, accType, /*isUnsignedCast=*/false);
  Value yConvert =
      convertScalarToDtype(builder, loc, y, accType, /*isUnsignedCast=*/false);
  if (accType.isa<IntegerType>())
    return builder.create<arith::MulIOp>(loc, xConvert, yConvert);
  return builder.create<arith::MulFOp>(loc, xConvert, yConvert);
}

// Delinearizes the given composite `index` by the basis specified in `factors`.
static SmallVector<Value> unrollIndex(OpBuilder &b, Location loc, Value index,
                                      ArrayRef<int64_t> factors) {
  assert(!factors.empty() && "empty factor list");
  SmallVector<Value> basis;
  for (int64_t f : factors)
    basis.push_back(b.create<arith::ConstantOp>(loc, b.getIndexAttr(f)));
  FailureOr<SmallVector<Value>> multiIndex =
      affine::delinearizeIndex(b, loc, index, basis);
  assert(!failed(multiIndex) && "Failed to linearize img2col index");
  return *multiIndex;
}

// Given indices corresponding to iterators in the output (oIndex) and filter
// (fIndex) for a convolution, compute the convolved index for the
// input as `oIndex * stride + fIndex`.
static Value getConvolvedIndex(OpBuilder &b, Location loc, Value oIndex,
                               Value fIndex, int64_t stride) {
  AffineExpr oExpr, fExpr;
  bindSymbols(b.getContext(), oExpr, fExpr);
  AffineMap convMap = AffineMap::get(0, 2, stride * oExpr + fExpr);
  return affine::makeComposedAffineApply(b, loc, convMap,
                                         ValueRange{oIndex, fIndex});
}

static SmallVector<int64_t> getDimOrder(AffineMap map) {
  SmallVector<int64_t> dimOrder;
  for (AffineExpr expr : map.getResults())
    dimOrder.push_back(expr.cast<AffineDimExpr>().getPosition());
  return dimOrder;
}

static SmallVector<StringRef> getShortDimTypeNames(
        DenseMap<unsigned, ConvolutionDimType> convDimMap,
        SmallVector<int64_t> dimOrder) {
  SmallVector<StringRef> typeNames;
  for (auto dim : dimOrder)
    typeNames.push_back(getShortDimTypeName(convDimMap[dim]));
  return typeNames;
}

static SmallVector<int64_t> getIm2ColDimOrder(AffineMap inputMap,
        DenseMap<unsigned, ConvolutionDimType> convDimMap,
        SmallVector<int64_t> filterDimOrder, SmallVector<int64_t> outputDimOrder) {
  SmallVector<int64_t> inputDimOrder;
  int outputIndex = 0;
  int filterIndex = 0;
  bool foundInputChannel = false;
  for (int filterEnd = filterDimOrder.size(), outputEnd = outputDimOrder.size();
          filterIndex < filterEnd || outputIndex < outputEnd;) {
    if (outputIndex >= outputEnd) {
      for (;filterIndex < filterEnd; filterIndex++)
        inputDimOrder.push_back(filterDimOrder[filterIndex]);
      break;
    }
    if (filterIndex >= filterEnd) {
      for (;outputIndex < outputEnd; outputIndex++)
        inputDimOrder.push_back(outputDimOrder[outputIndex]);
      break;
    }
    if (isa<ConvolutionDimType::Batch>(convDimMap[outputDimOrder[outputIndex]])) {
      inputDimOrder.push_back(outputDimOrder[outputIndex++]);
      continue;
    }
    if (foundInputChannel) {
      if (isa<ConvolutionDimType::FilterLoop>(convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::OutputImage>(convDimMap[outputDimOrder[outputIndex]])) {
        inputDimOrder.push_back(outputDimOrder[outputIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::InputChannel>(convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
    } else {
      if (isa<ConvolutionDimType::InputChannel>(convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        foundInputChannel = true;
        continue;
      }
      if (isa<ConvolutionDimType::OutputImage>(convDimMap[outputDimOrder[outputIndex]])) {
        inputDimOrder.push_back(outputDimOrder[outputIndex++]);
        continue;
      }
      if (isa<ConvolutionDimType::FilterLoop>(convDimMap[filterDimOrder[filterIndex]])) {
        inputDimOrder.push_back(filterDimOrder[filterIndex++]);
        continue;
      }
    }
    // Else it is necessarily an OutputChannel for both so increment both indices.
    filterIndex++;
    outputIndex++;
  }
  return inputDimOrder;
}

//===----------------------------------------------------------------------===//
// Im2Col Rewrite Patterns
//===----------------------------------------------------------------------===//

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase & rewriter, linalg::GenericOp genericOp,
        bool collapseFilter, bool collapseOutput) {
  mlir::linalg::detail::ConvolutionDimensions dimensions;
  if (!linalg::detail::getMatchConvolutionMessage(
              linalg::detail::isConvolutionInterfaceImpl(genericOp, &dimensions)).empty())
    return failure();

  if (dimensions.outputImage.size() < 2 || dimensions.filterLoop.size() < 2)
    return failure();

  assert(dimensions.outputImage.size() == dimensions.filterLoop.size());

  auto inputType = genericOp.getInputs()[0].getType().cast<ShapedType>();
  auto filterType = genericOp.getInputs()[1].getType().cast<ShapedType>();
  auto outputType = genericOp.getOutputs()[0].getType().cast<ShapedType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        genericOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(genericOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!llvm::all_of(
      dimensions.dilations, [](unsigned element) { return element == 1; }))
    return rewriter.notifyMatchFailure(genericOp,
                                       "expected all ones for dilations");

  auto loc = genericOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  Value input = genericOp.getInputs()[0];
  Value filter = genericOp.getInputs()[1];
  Value output = genericOp.getOutputs()[0];

  auto dimSizes = genericOp.getStaticLoopRanges();
  auto convDimMap = getConvolutionDimTypeMap(dimensions);

  auto indexingMaps = genericOp.getIndexingMapsArray();
  AffineMap inputMap = indexingMaps[0];
  AffineMap filterMap = indexingMaps[1];
  AffineMap outputMap = indexingMaps[2];

  SmallVector<int64_t> filterDimOrder = getDimOrder(filterMap);
  SmallVector<int64_t> outputDimOrder = getDimOrder(outputMap);
  SmallVector<int64_t> im2ColDimOrder = getIm2ColDimOrder(
          inputMap, convDimMap, filterDimOrder, outputDimOrder);

  LLVM_DEBUG({
    DBGS() << "im2col: Selected dimension orders.\n";
    llvm::interleaveComma(im2ColDimOrder, DBGS() << "im2ColDimOrder: ");
    llvm::dbgs() << "\n";
    llvm::interleaveComma(getShortDimTypeNames(convDimMap, im2ColDimOrder),
            DBGS() << "im2ColDimTypes: ");
    llvm::dbgs() << "\n";
    llvm::interleaveComma(filterDimOrder, DBGS() << "filterDimOrder: ");
    llvm::dbgs() << "\n";
    llvm::interleaveComma(getShortDimTypeNames(convDimMap, filterDimOrder),
            DBGS() << "filterDimTypes: ");
    llvm::dbgs() << "\n";
    llvm::interleaveComma(outputDimOrder, DBGS() << "outputDimOrder: ");
    llvm::dbgs() << "\n";
    llvm::interleaveComma(getShortDimTypeNames(convDimMap, outputDimOrder),
        DBGS() << "outputDimTypes: ");
    llvm::dbgs() << "\n";
  });

  SmallVector<ReassociationIndices> im2ColReassocIndices;
  SmallVector<int64_t> im2ColShape;
  SmallVector<ReassociationIndices> filterReassocIndices;
  SmallVector<int64_t> filterShape;
  SmallVector<ReassociationIndices> outputReassocIndices;
  SmallVector<int64_t> outputShape;

  auto findDimPosFromIndex = [](int64_t dimPos, int64_t index, SmallVector<int64_t> dimOrder) {
    for (int i = index, e = dimOrder.size(); i < e; i++) {
      if (dimOrder[i] == dimPos)
        return i;
    }
    return static_cast<int>(dimOrder.size());
  };

  auto updateReassociationAndShapeVecs = [&](int64_t dim, int64_t prevIndex,
          int64_t nextIndex, bool prevCollapsible, SmallVector<int64_t> dimOrder,
          SmallVector<int64_t> &shapeVec, SmallVector<ReassociationIndices> &reassociationMap) {
    auto dimSize = dimSizes[dim];
    if (prevIndex == nextIndex && prevCollapsible) {
      im2ColReassocIndices.back().push_back(dim);
      im2ColShape[im2ColShape.size() - 1] *= dimSize;
      reassociationMap.back().push_back(prevIndex);
      shapeVec[shapeVec.size() - 1] *= dimSize;
    } else {
      im2ColReassocIndices.push_back({dim});
      im2ColShape.push_back(dimSize);
      for (auto i = prevIndex; i < nextIndex + 1; i++) {
        reassociationMap.push_back({i});
        shapeVec.push_back(dimSizes[dimOrder[i]]);
      }
    }
  };

  int64_t filterDimPos = 0;
  int64_t outputDimPos = 0;
  bool prevWasFilterCollapsible = false;
  bool prevWasOutputCollapsible = false;
  for (auto dim : im2ColDimOrder) {
    if (isa<ConvolutionDimType::InputChannel,
            ConvolutionDimType::FilterLoop>(convDimMap[dim])) {
      prevWasOutputCollapsible = false;
      auto nextFilter = findDimPosFromIndex(dim, filterDimPos, filterDimOrder);
      if (nextFilter >= filterDimOrder.size())
        return rewriter.notifyMatchFailure(genericOp,
                                           "filter layout does not match im2col layout");
      updateReassociationAndShapeVecs(dim, filterDimPos, nextFilter,
              prevWasFilterCollapsible && collapseFilter, filterDimOrder, filterShape, filterReassocIndices);

      filterDimPos = nextFilter + 1;
      prevWasFilterCollapsible = true;
      continue;
    } else if (isa<ConvolutionDimType::OutputImage>(convDimMap[dim])) {
      prevWasFilterCollapsible = false;
      auto nextOutput = findDimPosFromIndex(dim, outputDimPos, outputDimOrder);
      if (nextOutput >= outputDimOrder.size())
        return rewriter.notifyMatchFailure(genericOp,
                                           "output layout does not match im2col layout");
      updateReassociationAndShapeVecs(dim, outputDimPos, nextOutput,
              prevWasOutputCollapsible && collapseOutput, outputDimOrder, outputShape, outputReassocIndices);
    
      outputDimPos = nextOutput + 1;
      prevWasOutputCollapsible = true;
      continue;
    } else if (isa<ConvolutionDimType::Batch>(convDimMap[dim])) {
      auto nextOutput = findDimPosFromIndex(dim, outputDimPos, outputDimOrder);
      if (nextOutput >= outputDimOrder.size())
        return rewriter.notifyMatchFailure(genericOp,
                                           "output layout does not match im2col layout");
      updateReassociationAndShapeVecs(dim, outputDimPos, nextOutput,
              prevWasOutputCollapsible && collapseOutput, outputDimOrder, outputShape, outputReassocIndices);

      outputDimPos = nextOutput + 1;
    }
    prevWasFilterCollapsible = false;
    prevWasOutputCollapsible = false;
  }
  for (int i = filterDimPos, e = filterDimOrder.size(); i < e; i++) {
    filterReassocIndices.push_back({i});
    filterShape.push_back(dimSizes[filterDimOrder[i]]);
  }
  for (int i = outputDimPos, e = outputDimOrder.size(); i < e; i++) {
    outputReassocIndices.push_back({i});
    outputShape.push_back(dimSizes[outputDimOrder[i]]);
  }

  auto reshapedFilterType =
      RankedTensorType::get(filterShape, filterType.getElementType());
  Value reshapedFilter = filter;
  if (collapseFilter) {
    reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, filterReassocIndices);
  }

  auto reshapedOutputType =
      RankedTensorType::get(outputShape, outputType.getElementType());
  Value reshapedOutput = output;
  if (collapseOutput) {
    reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, outputReassocIndices);
  }

  // Convert the input to a (BKN) tensor.
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, im2ColShape, inputType.getElementType());

  auto nloops = im2ColShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

  SmallVector<AffineMap, 4> img2colIndexingMaps = {
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/ValueRange{}, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Get the iterators named based on the matmul (batch?, m, k).
        DenseMap<int64_t, Value> indexMap;
        for (auto [index, group] : llvm::enumerate(im2ColReassocIndices)) {
          Value iterator = nestedBuilder.create<linalg::IndexOp>(loc, index);
          if (group.size() == 1) {
            indexMap[group[0]] = iterator;
            continue;
          }
          SmallVector<int64_t> groupShape;
          for (auto dim : group) groupShape.push_back(dimSizes[dim]);
          auto unrolledIndices = unrollIndex(nestedBuilder, nestedLoc, iterator, groupShape);
          for (auto [dim, unrolledIndex] : llvm::zip_equal(group, unrolledIndices))
            indexMap[dim] = unrolledIndex;
        }

        SmallVector<AffineExpr> convolvedExprs;
        for (auto expr : inputMap.getResults())
          if (!expr.isa<AffineDimExpr>()) convolvedExprs.push_back(expr);

        SmallVector<Value> convolvedIndices;
        for (auto [imageDim, filterDim, expr, stride] : llvm::zip_equal(
                    dimensions.outputImage, dimensions.filterLoop,
                    convolvedExprs, dimensions.strides)) {
          assert(expr.isFunctionOfDim(imageDim) && expr.isFunctionOfDim(filterDim)
                  && "convolved indices mismatch with indexing order in source map");
          convolvedIndices.push_back(getConvolvedIndex(nestedBuilder, nestedLoc,
                      indexMap[imageDim], indexMap[filterDim], stride));
        }

        SmallVector<Value> extractionIndices;
        unsigned convolvedIndexDim = 0;
        for (auto expr : inputMap.getResults()) {
          auto dimExpr = expr.dyn_cast<AffineDimExpr>();
          if (!dimExpr) {
            extractionIndices.push_back(convolvedIndices[convolvedIndexDim++]);
            continue;
          }
          extractionIndices.push_back(indexMap[dimExpr.getPosition()]);
        }

        Value inputVal = nestedBuilder.create<tensor::ExtractOp>(
                loc, input, extractionIndices);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
      });

  auto getGroupedDimOrder = [](SmallVector<int64_t> dimOrder,
                               SmallVector<ReassociationIndices> reassociationMap) {
    SmallVector<ReassociationIndices> groupedDimOrder;
    for (auto group : reassociationMap) {
      ReassociationIndices dims;
      for (auto i : group) dims.push_back(dimOrder[i]);
      groupedDimOrder.push_back(dims);
    }
    return groupedDimOrder;
  };
  SmallVector<ReassociationIndices> groupedFilterDimOrder =
      getGroupedDimOrder(filterDimOrder, filterReassocIndices);
  SmallVector<ReassociationIndices> groupedOutputDimOrder =
      getGroupedDimOrder(outputDimOrder, outputReassocIndices);

  LLVM_DEBUG({
    DBGS() << "im2col: Selected dimension groupings.\n";
    DBGS() << "im2ColDimMap: ";
    for (auto group : im2ColReassocIndices) {
      llvm::dbgs() << "{";
      llvm::interleaveComma(group, llvm::dbgs());
      llvm::dbgs() << "} ";
    }
    llvm::dbgs() << "\n";
    DBGS() << "filterDimMap: ";
    for (auto group : groupedFilterDimOrder) {
      llvm::dbgs() << "{";
      llvm::interleaveComma(group, llvm::dbgs());
      llvm::dbgs() << "} ";
    }
    llvm::dbgs() << "\n";
    DBGS() << "outputDimMap: ";
    for (auto group : groupedOutputDimOrder) {
      llvm::dbgs() << "{";
      llvm::interleaveComma(group, llvm::dbgs());
      llvm::dbgs() << "} ";
    }
    llvm::dbgs() << "\n";
  });

  DenseMap<int64_t, AffineExpr> leadingGroupDimToIterationDim;
  SmallVector<utils::IteratorType> iteratorTypes;
  int iterationDim = 0;

  auto advanceOperandIndex = [&](int &l, int r, SmallVector<ReassociationIndices> groupedDimOrder) {
    while (l < groupedDimOrder.size() &&
           leadingGroupDimToIterationDim.count(groupedDimOrder[l][0]))
      l++;
    for (; l < r; l++) {
      auto groupDim = groupedDimOrder[l][0];
      if (leadingGroupDimToIterationDim.count(groupDim))
        continue;
      leadingGroupDimToIterationDim[groupDim] = rewriter.getAffineDimExpr(iterationDim++);
      if (isa<ConvolutionDimType::FilterLoop,
              ConvolutionDimType::InputChannel>(convDimMap[groupDim]))
        iteratorTypes.push_back(reduction);
      else
        iteratorTypes.push_back(parallel);
    }
  };

  // If the inner most dim of the input is a reduced dimension, assume we should
  // keep the filter on the right hand side of the matrix multiplication.
  bool filterRHS = isa<ConvolutionDimType::FilterLoop,
                       ConvolutionDimType::InputChannel>(convDimMap[im2ColDimOrder.back()]);

  int inputIndex = 0;
  int filterIndex = 0;
  int outputIndex = 0;
  int inputEnd = im2ColShape.size();
  int filterEnd = filterShape.size();
  int outputEnd = outputShape.size();
  while (inputIndex < inputEnd &&
         filterIndex < filterEnd &&
         outputIndex < outputEnd) {
    ConvolutionDimType lhsDimType = filterRHS ? convDimMap[im2ColReassocIndices[inputIndex][0]]
                                              : convDimMap[groupedFilterDimOrder[filterIndex][0]];
    bool isLastLhsDim = filterRHS ? im2ColReassocIndices.size() - 1 == inputIndex
                                  : filterReassocIndices.size() - 1 == filterIndex;
    LLVM_DEBUG({
      DBGS() << "Current dim to iteration dim map:\n";
      for (auto [key, value] : leadingGroupDimToIterationDim) {
        DBGS() << key << " -> " << value << "\n";
      }
      DBGS() << "Input Index: " << inputIndex << "\n";
      DBGS() << "Filter Index: " << filterIndex << "\n";
      DBGS() << "Output Index: " << outputIndex << "\n";
      DBGS() << "Is last LHS Dim: " << (isLastLhsDim ? "true" : "false") << "\n";
      DBGS() << "LHS Dim Type: " << getShortDimTypeName(lhsDimType) << "\n";
    });
    if (isLastLhsDim && isa<ConvolutionDimType::FilterLoop,
                            ConvolutionDimType::InputChannel>(lhsDimType))
      break;

    if (groupedFilterDimOrder[filterIndex][0] == im2ColReassocIndices[inputIndex][0]) {
      advanceOperandIndex(filterIndex, filterIndex + 1, groupedFilterDimOrder);
    } else if (groupedOutputDimOrder[outputIndex][0] == groupedFilterDimOrder[filterIndex][0]) {
      advanceOperandIndex(outputIndex, outputIndex + 1, groupedOutputDimOrder);
    } else if (groupedOutputDimOrder[outputIndex][0] == im2ColReassocIndices[inputIndex][0]) {
      advanceOperandIndex(inputIndex, inputIndex + 1, im2ColReassocIndices);
    } else {
      // Greedily prefer iterating over dims of the output.
      advanceOperandIndex(outputIndex, outputIndex + 1, groupedOutputDimOrder);
    }

    // Catch up all indices based on the iterator dim map.
    advanceOperandIndex(outputIndex, outputIndex, groupedOutputDimOrder);
    advanceOperandIndex(filterIndex, filterIndex, groupedFilterDimOrder);
    advanceOperandIndex(inputIndex, inputIndex, im2ColReassocIndices);
  }

  // Fill in any missing iteration dimensions.
  advanceOperandIndex(outputIndex, outputEnd, groupedOutputDimOrder);
  advanceOperandIndex(filterIndex, filterEnd, groupedFilterDimOrder);
  advanceOperandIndex(inputIndex, inputEnd, im2ColReassocIndices);

  SmallVector<AffineExpr> inputExprs;
  SmallVector<AffineExpr> filterExprs;
  SmallVector<AffineExpr> outputExprs;

  for (auto group : im2ColReassocIndices)
    inputExprs.push_back(leadingGroupDimToIterationDim[group[0]]);
  for (auto group : groupedFilterDimOrder)
    filterExprs.push_back(leadingGroupDimToIterationDim[group[0]]);
  for (auto group : groupedOutputDimOrder)
    outputExprs.push_back(leadingGroupDimToIterationDim[group[0]]);

  auto newInputMap = AffineMap::get(iterationDim, 0, inputExprs, context);
  auto newFilterMap = AffineMap::get(iterationDim, 0, filterExprs, context);
  auto newOutputMap = AffineMap::get(iterationDim, 0, outputExprs, context);

  LLVM_DEBUG({
    DBGS() << "im2col: Indexing maps.\n";
    if (filterRHS) {
      DBGS() << "im2ColMap: " << newInputMap << "\n";
      DBGS() << "filterMap: " << newFilterMap << "\n";
    } else {
      DBGS() << "filterMap: " << newFilterMap << "\n";
      DBGS() << "im2ColMap: " << newInputMap << "\n";
    }
    DBGS() << "outputMap: " << newOutputMap << "\n";
  });

  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputType,
      /*inputs=*/filterRHS ? ValueRange{img2ColTensor.getResult(0), reshapedFilter}
                        : ValueRange{reshapedFilter, img2ColTensor.getResult(0)},
      /*outputs=*/ValueRange{reshapedOutput},
      filterRHS ? ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap}
                : ArrayRef<AffineMap>{newFilterMap, newInputMap, newOutputMap}, iteratorTypes);
  IRMapping mapper;
  genericOp.getRegion().cloneInto(&newGenericOp.getRegion(), mapper);

  if (!filterRHS) swapGenericBinaryInputArgs(newGenericOp);

  Value result = newGenericOp.getResults().front();

  Operation *transformResult = newGenericOp;
  Value reshapedResult = result;
  if (collapseOutput) {
    transformResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputType, result, outputReassocIndices);
    reshapedResult = transformResult->getResult(0);
  }

  rewriter.replaceOp(genericOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(), transformResult);
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<ShapedType>();
  auto filterType = convOp.getInputs()[1].getType().cast<ShapedType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<ShapedType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  MLIRContext *context = rewriter.getContext();
  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  ArrayRef<int64_t> filterShape = filterType.getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int64_t n = outputShape[0];
  int64_t oh = outputShape[1];
  int64_t ow = outputShape[2];
  int64_t oc = outputShape[3];
  int64_t fh = filterShape[0];
  int64_t fw = filterShape[1];
  int64_t ic = filterShape[2];

  Location loc = convOp.getLoc();

  // Reshape output and filter to the LHS and result of a (B)MNK matmul.
  SmallVector<ReassociationIndices> filterReassocIndices = {{0, 1, 2}, {3}};
  auto reshapedFilterType =
      RankedTensorType::get({fh * fw * ic, oc}, inputType.getElementType());
  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1, 2}, {3}};
  RankedTensorType reshapedOutputType =
      RankedTensorType::get({n, oh * ow, oc}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  SmallVector<int64_t> colTensorShape = {n, oh * ow, fh * fw * ic};
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  // Convert the input to a (BMK) column tensor.
  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType> img2colIterators(nloops, parallel);

  SmallVector<AffineMap> img2colIndexingMaps = {
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/ValueRange{}, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Get the iterators named based on the matmul (batch, m, k).
        Value bIndex = nestedBuilder.create<linalg::IndexOp>(loc, 0);
        Value mIndex = nestedBuilder.create<linalg::IndexOp>(loc, 1);
        Value kIndex = nestedBuilder.create<linalg::IndexOp>(loc, 2);

        // Recover the original iteration indices from the problem/input sizes.
        SmallVector<Value> mIndices = unrollIndex(
            nestedBuilder, nestedLoc, mIndex, ArrayRef<int64_t>{oh, ow});
        auto ohIndex = mIndices[0];
        auto owIndex = mIndices[1];

        SmallVector<Value> kIndices = unrollIndex(
            nestedBuilder, nestedLoc, kIndex, ArrayRef<int64_t>{fh, fw, ic});
        auto fhIndex = kIndices[0];
        auto fwIndex = kIndices[1];
        auto icIndex = kIndices[2];

        // Extract the input element corresponding to the expanded indices.
        Value hIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, ohIndex, fhIndex,
                              convOp.getStrides().getValues<int64_t>()[0]);
        Value wIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, owIndex, fwIndex,
                              convOp.getStrides().getValues<int64_t>()[1]);

        // im2col[n, oh*ow, fh*fw*ic] = input[n, sh*oh + fh, sw*ow + fw, ic]
        SmallVector<Value> extractionIndices{bIndex, hIndex, wIndex, icIndex};
        Value inputVal = nestedBuilder.create<tensor::ExtractOp>(
            loc, input, extractionIndices);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
      });

  // Because the filter does not share the same batch dimension,
  // the batch dimension is only used in indexing the input and output. Thus
  // we cannot use existing linalg named ops like linalg.batch_matmul.
  // i.e. (B x) M x K * K x N = (B x) M x N
  AffineExpr bDim, mDim, nDim, kDim;
  bindDims(context, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {bDim, mDim, kDim}, context);
  auto rhsMap = AffineMap::get(4, 0, {kDim, nDim}, context);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, context);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};

  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputType,
      /*inputs=*/ValueRange{img2ColTensor.getResult(0), reshapedFilter},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter,
                linalg::DepthwiseConv2DNhwcHwcOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<RankedTensorType>();
  auto filterType = convOp.getInputs()[1].getType().cast<RankedTensorType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<RankedTensorType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  Location loc = convOp.getLoc();

  auto transposeOperand = [&](Value operand, ArrayRef<int64_t> indices) {
    auto operandTensorType = operand.getType().cast<RankedTensorType>();
    auto nloops = indices.size();
    ArrayRef<int64_t> inputShape = operandTensorType.getShape();

    SmallVector<AffineExpr> exprs = llvm::to_vector<4>(
        llvm::map_range(indices, [&](int64_t index) -> AffineExpr {
          return rewriter.getAffineDimExpr(index);
        }));

    SmallVector<int64_t> targetShape = llvm::to_vector<4>(llvm::map_range(
        indices, [&](int64_t index) -> int64_t { return inputShape[index]; }));

    Value outputTensor = rewriter.create<tensor::EmptyOp>(
        loc, targetShape, operandTensorType.getElementType());

    SmallVector<utils::IteratorType> loopAttributeTypes(
        nloops, utils::IteratorType::parallel);

    SmallVector<AffineMap> indexingMaps = {
        inversePermutation(
            AffineMap::get(nloops, 0, exprs, rewriter.getContext())),
        AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

    auto transposedOp = rewriter.create<linalg::GenericOp>(
        loc, outputTensor.getType(),
        /*inputs=*/operand, /*outputs=*/outputTensor, indexingMaps,
        loopAttributeTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
        });

    return transposedOp.getResult(0);
  };

  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  // Transpose input, filter so channels are outermost
  Value inputT = transposeOperand(input, {0, 3, 1, 2});
  Value filterT = transposeOperand(filter, {2, 0, 1});
  ArrayRef<int64_t> filterTShape =
      filterT.getType().cast<RankedTensorType>().getShape();
  ArrayRef<int64_t> outputShape = outputType.getShape();

  int n = outputShape[0];
  int oh = outputShape[1];
  int ow = outputShape[2];
  int c = outputShape[3];
  int fh = filterTShape[1];
  int fw = filterTShape[2];

  SmallVector<int64_t> colTensorShape = {n, c, oh, ow, fh, fw};
  Value transposedOutputTensor = transposeOperand(output, {0, 3, 1, 2});

  AffineExpr nDim, cDim, ohDim, owDim, khDim, kwDim;
  bindDims(rewriter.getContext(), nDim, cDim, ohDim, owDim, khDim, kwDim);

  AffineExpr shSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[0]);
  AffineExpr swSym = rewriter.getAffineConstantExpr(
      convOp.getStrides().getValues<int64_t>()[1]);

  SmallVector<AffineExpr> inputExprs = {nDim, cDim, ohDim * shSym + khDim,
                                        owDim * swSym + kwDim};

  auto nloops = colTensorShape.size();

  SmallVector<utils::IteratorType> loopAttributeTypes(
      nloops, utils::IteratorType::parallel);

  SmallVector<AffineMap> indexingMaps = {
      AffineMap::get(nloops, 0, inputExprs, rewriter.getContext()),
      AffineMap::getMultiDimIdentityMap(nloops, rewriter.getContext())};

  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/inputT, /*outputs=*/colTensor, indexingMaps,
      loopAttributeTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, args[0]);
      });

  SmallVector<ReassociationIndices> img2ColTensorReassocIndices = {
      {0, 1}, {2, 3}, {4, 5}};
  SmallVector<ReassociationIndices> filterReassociationIndice = {{0}, {1, 2}};
  SmallVector<ReassociationIndices> outputReassociationIndice = {{0, 1},
                                                                 {2, 3}};

  auto reshapedImg2ColTensorType = RankedTensorType::get(
      {n * c, oh * ow, fh * fw}, inputType.getElementType());
  auto reshapedFilterTensorType =
      RankedTensorType::get({c, fh * fw}, filterType.getElementType());
  auto reshapedOutputTensorType =
      RankedTensorType::get({n * c, oh * ow}, outputType.getElementType());

  Value reshapedImg2ColTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedImg2ColTensorType, img2ColTensor.getResult(0),
      img2ColTensorReassocIndices);
  Value reshapedFilterTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterTensorType, filterT, filterReassociationIndice);
  Value reshapedoutputTensor = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputTensorType, transposedOutputTensor,
      outputReassociationIndice);

  auto batchMatVecResult = rewriter.create<linalg::BatchMatvecOp>(
      loc, TypeRange{reshapedoutputTensor.getType()},
      ValueRange{reshapedImg2ColTensor, reshapedFilterTensor},
      ValueRange{reshapedoutputTensor});

  SmallVector<ReassociationIndices> batchMatVecReassociationIndice = {{0, 1},
                                                                      {2, 3}};

  Value batchMatVecResultReshaped = rewriter.create<tensor::ExpandShapeOp>(
      loc, transposedOutputTensor.getType(), batchMatVecResult.getResult(0),
      batchMatVecReassociationIndice);

  Value transposedResult =
      transposeOperand(batchMatVecResultReshaped, {0, 2, 3, 1});

  rewriter.replaceOp(convOp, ArrayRef<Value>{transposedResult});
  return std::make_pair(img2ColTensor.getOperation(),
                        transposedResult.getDefiningOp());
}

FailureOr<std::pair<Operation *, Operation *>>
rewriteInIm2Col(RewriterBase &rewriter, linalg::Conv2DNchwFchwOp convOp) {
  auto inputType = convOp.getInputs()[0].getType().cast<ShapedType>();
  auto filterType = convOp.getInputs()[1].getType().cast<ShapedType>();
  auto outputType = convOp.getOutputs()[0].getType().cast<ShapedType>();

  if (!filterType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        convOp, "expected a static shape for the filter");

  if (!inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(convOp,
                                       "expected a static shape for the input");

  // TODO: Support dilation.
  if (!hasAllOneValues(convOp.getDilations()))
    return rewriter.notifyMatchFailure(convOp,
                                       "expected all ones for dilations");

  Value input = convOp.getInputs()[0];
  Value filter = convOp.getInputs()[1];
  Value output = convOp.getOutputs()[0];

  auto filterShape = filterType.getShape();
  auto outputShape = outputType.getShape();

  int64_t n = outputShape[0];
  int64_t oc = outputShape[1];
  int64_t oh = outputShape[2];
  int64_t ow = outputShape[3];
  int64_t ic = filterShape[1];
  int64_t fh = filterShape[2];
  int64_t fw = filterShape[3];

  auto loc = convOp.getLoc();
  MLIRContext *context = rewriter.getContext();

  SmallVector<ReassociationIndices> filterReassocIndices = {{0}, {1, 2, 3}};
  auto reshapedFilterType =
      RankedTensorType::get({oc, ic * fh * fw}, inputType.getElementType());
  Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedFilterType, filter, filterReassocIndices);

  SmallVector<ReassociationIndices> outputReassocIndices = {{0}, {1}, {2, 3}};
  auto reshapedOutputType =
      RankedTensorType::get({n, oc, oh * ow}, outputType.getElementType());
  Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
      loc, reshapedOutputType, output, outputReassocIndices);

  // Convert the input to a (BKN) tensor.
  SmallVector<int64_t, 4> colTensorShape = {n, ic * fh * fw, oh * ow};
  Value colTensor = rewriter.create<tensor::EmptyOp>(
      loc, colTensorShape, inputType.getElementType());

  auto nloops = colTensorShape.size();

  auto parallel = utils::IteratorType::parallel;
  auto reduction = utils::IteratorType::reduction;
  SmallVector<utils::IteratorType, 3> img2colIterators(nloops, parallel);

  SmallVector<AffineMap, 4> img2colIndexingMaps = {
      AffineMap::getMultiDimIdentityMap(nloops, context)};

  auto img2ColTensor = rewriter.create<linalg::GenericOp>(
      loc, colTensor.getType(),
      /*inputs=*/ValueRange{}, /*outputs=*/colTensor, img2colIndexingMaps,
      img2colIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        // Get the iterators named based on the matmul (batch, m, k).
        Value bIndex = nestedBuilder.create<linalg::IndexOp>(loc, 0);
        Value kIndex = nestedBuilder.create<linalg::IndexOp>(loc, 1);
        Value nIndex = nestedBuilder.create<linalg::IndexOp>(loc, 2);

        // Recover the original iteration indices from the problem/input sizes.
        SmallVector<Value> kIndices = unrollIndex(
            nestedBuilder, nestedLoc, kIndex, ArrayRef<int64_t>{ic, fh, fw});
        auto icIndex = kIndices[0];
        auto fhIndex = kIndices[1];
        auto fwIndex = kIndices[2];

        SmallVector<Value> nIndices = unrollIndex(
            nestedBuilder, nestedLoc, nIndex, ArrayRef<int64_t>{oh, ow});
        auto ohIndex = nIndices[0];
        auto owIndex = nIndices[1];

        // Extract the input element corresponding to the expanded indices.
        Value hIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, ohIndex, fhIndex,
                              convOp.getStrides().getValues<int64_t>()[0]);
        Value wIndex =
            getConvolvedIndex(nestedBuilder, nestedLoc, owIndex, fwIndex,
                              convOp.getStrides().getValues<int64_t>()[1]);

        // im2col[n, ic*fh*fw, oh*ow] = input[n, ic, sh*oh + fh, sw*ow + fw]
        SmallVector<Value> extractionIndices{bIndex, icIndex, hIndex, wIndex};
        Value inputVal = nestedBuilder.create<tensor::ExtractOp>(
            loc, input, extractionIndices);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, inputVal);
      });

  // Because the filter does not share the same batch dimension,
  // the batch dimension is only used in indexing the input and output. Thus
  // we cannot use existing linalg named ops like linalg.batch_matmul.
  // i.e. M x K * (B x) K x N = (B x) M x N
  AffineExpr bDim, mDim, nDim, kDim;
  bindDims(context, bDim, mDim, nDim, kDim);
  auto lhsMap = AffineMap::get(4, 0, {mDim, kDim}, context);
  auto rhsMap = AffineMap::get(4, 0, {bDim, kDim, nDim}, context);
  auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, context);
  SmallVector<utils::IteratorType> genericIterators = {parallel, parallel,
                                                       parallel, reduction};
  auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, reshapedOutputType,
      /*inputs=*/ValueRange{reshapedFilter, img2ColTensor.getResult(0)},
      /*outputs=*/ValueRange{reshapedOutput},
      ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value mul =
            createMul(loc, args[0], args[1], args[2].getType(), nestedBuilder);
        Value add = createAdd(loc, mul, args[2], nestedBuilder);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
      });
  Value result = genericOp.getResults().front();

  auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
      loc, outputType, result, outputReassocIndices);

  rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

  return std::make_pair(img2ColTensor.getOperation(),
                        reshapedResult.getOperation());
}

namespace {

class ConvertConv2DNhwcHwcf final
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};

class ConvertDepthwiseConv2DNhwcHwc final
    : public OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp> {
public:
  using OpRewritePattern<linalg::DepthwiseConv2DNhwcHwcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::DepthwiseConv2DNhwcHwcOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};

class ConvertConv2DNchwFchw final
    : public OpRewritePattern<linalg::Conv2DNchwFchwOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNchwFchwOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp)))
      return failure();
    return success();
  }
};

class ConvertConv2DGeneric final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp convOp,
                                PatternRewriter &rewriter) const override {
    if (failed(rewriteInIm2Col(rewriter, convOp, /*collapseFilter=*/true)))
      return failure();
    return success();
  }
};
} // end anonymous namespace

void populateConvertConv2DToImg2ColPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert<ConvertConv2DNhwcHwcf, ConvertDepthwiseConv2DNhwcHwc,
                  ConvertConv2DNchwFchw, ConvertConv2DGeneric>(context);
}
} // end namespace linalg
} // end namespace mlir
