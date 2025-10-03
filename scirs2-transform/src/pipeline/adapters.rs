//! Adapter implementations for existing transformers to work with pipelines

use scirs2_core::ndarray::Array2;
use std::any::Any;

use crate::{
    encoding::{BinaryEncoder, OneHotEncoder, OrdinalEncoder, TargetEncoder},
    error::Result,
    features::{PolynomialFeatures, PowerTransformer},
    impute::{IterativeImputer, KNNImputer, SimpleImputer},
    normalize::Normalizer,
    pipeline::Transformer,
    reduction::{TruncatedSVD, LDA, PCA, TSNE},
    scaling::{MaxAbsScaler, QuantileTransformer},
    selection::VarianceThreshold,
};

// Macro to implement Transformer trait for types with fit and transform methods
macro_rules! impl_transformer {
    ($type:ty) => {
        impl Transformer for $type {
            fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
                self.fit(x)
            }

            fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
                self.transform(x)
            }

            fn clone_box(&self) -> Box<dyn Transformer> {
                Box::new(self.clone())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }
    };
}

// Implement Transformer for all existing transformers
impl_transformer!(PCA);
impl_transformer!(TruncatedSVD);
impl_transformer!(LDA);
impl_transformer!(TSNE);
impl_transformer!(Normalizer);
impl_transformer!(PolynomialFeatures);
impl_transformer!(PowerTransformer);
impl_transformer!(MaxAbsScaler);
impl_transformer!(QuantileTransformer);
impl_transformer!(SimpleImputer);
impl_transformer!(KNNImputer);
impl_transformer!(IterativeImputer);
impl_transformer!(OneHotEncoder);
impl_transformer!(OrdinalEncoder);
impl_transformer!(BinaryEncoder);
impl_transformer!(TargetEncoder);
impl_transformer!(VarianceThreshold);

/// Helper function to create a boxed transformer
#[allow(dead_code)]
pub fn boxed<T: Transformer + 'static>(transformer: T) -> Box<dyn Transformer> {
    Box::new(_transformer)
}
