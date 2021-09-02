#!/usr/bin/env python3

from .rbc import CRIPPER

class RIPPER( CRIPPER ):
  def __init__( self, *args, **kwargs ):
    super( RIPPER, self ).__init__( *args, **kwargs )

  def fit( self, X, Y, feature_names, positive_class ):
    return super( RIPPER, self ).fit( X.T, Y, feature_names, positive_class )
