add_circt_dialect_library(CIRCTCombTransforms
  LowerComb.cpp
  IntRangeOptimizations.cpp
  LowerToNand.cpp

  DEPENDS
  CIRCTCombTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
