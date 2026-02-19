
from dataclasses import dataclass
from typing import Sequence, Optional, List
import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,  RobustScaler, FunctionTransformer,PolynomialFeatures,KBinsDiscretizer
from sklearn.compose import ColumnTransformer
logger = logging.getLogger(__name__)

# Silence the Scikit-Learn 1.9 Future Warning for discretizers
warnings.filterwarnings(
    action="ignore", 
    category=FutureWarning, 
    module="sklearn.preprocessing._discretization"
)

# Silence the "Zero-Width Bins" warning (Expected behavior for financial data)
warnings.filterwarnings(
    action="ignore", 
    category=UserWarning, 
    module="sklearn.preprocessing._discretization"
)

def safe_log1p(x):
    """
    Safely applies log1p to avoid NaNs from negative financial numbers  refunds).
    """
    # Converts any negative number to 0.0 before applying the log
    x = np.nan_to_num(x, nan=0.0)
    return np.log1p(np.maximum(x, 0.0))


@dataclass
class BanditPreprocessorConfig:
    numeric_cols: Sequence[str]
    categorical_cols: Sequence[str]

class BanditPreprocessor:
    """
    preprocessing pipeline.
    uses RobustScaler to prevent over spenders form squashing features variance
    """
    def __init__(self, config: BanditPreprocessorConfig):
        self.config = config
        self._pipeline: Optional[Pipeline] = None

    def _build_pipeline(self) -> Pipeline:
        # 1. Numeric Pipeline
        #We use Log1p to handle the massive skew in financial data
        #then RobustScalar to center data on Median?IQR, ignoring outliers.
    
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            #normalize  distribution
            ("log_transform",FunctionTransformer(safe_log1p,validate=False)),
            #("scaler", RobustScaler()),
            #Expand the feature space to allow non-linear learning
            #("poly", PolynomialFeatures(degree=2, include_bias=False))
            ("binner", KBinsDiscretizer(n_bins=6, encode='onehot-dense', strategy='quantile'))
        ])

        # 2. Categorical Pipeline
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False is CRITICAL for Bandits (needs dense array)
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # 3. Combine
        transformers = [("num", num_pipe, list(self.config.numeric_cols))]
        
        # Only add categorical transformer if columns exist
        if self.config.categorical_cols:
            transformers.append(("cat", cat_pipe, list(self.config.categorical_cols)))

        ct = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False # Cleaner names (e.g. "age" instead of "num__age")
        )
        return Pipeline([("preprocessor", ct)])

    def fit(self, df: pd.DataFrame) -> "BanditPreprocessor":
        """Fits the pipeline to the training data."""
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms raw data into the context vector."""
        if self._pipeline is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        return self._pipeline.transform(df)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def get_feature_names(self) -> List[str]:
        """
        Returns the list of output feature names.
        Essential for debugging and feature importance analysis.
        """
        if self._pipeline is None:
            raise RuntimeError("Preprocessor not fitted.")
            
        try:
            # Access the 'preprocessor' step inside the pipeline
            ct = self._pipeline.named_steps["preprocessor"]
            return list(ct.get_feature_names_out())
        except Exception as e:
            logger.warning(f"Could not retrieve features names: {e}")
            return []