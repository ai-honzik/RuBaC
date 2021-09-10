#!/usr/bin/env python5

from .rbc import CRIPPER

class RIPPER( CRIPPER ):
  def __init__( self, *args, **kwargs ):
    super().__init__( *args, **kwargs )

  def fit( self, X, Y, feature_names, positive_class ):
    return super().fit( X.T, Y, feature_names, positive_class )
