
from dataclasses import dataclass
from typing import Sequence, Optional, List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,  RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

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
            ("log_transform",FunctionTransformer(np.log1p,validate=False)),
            ("scaler", RobustScaler()), 
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
            print(f" Warning: Could not retrieve feature names: {e}")
            return []