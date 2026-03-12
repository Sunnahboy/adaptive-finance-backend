
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd


# 1. The Features (Strict Rules) but currency safe
class UserFeatures(BaseModel):
    """
    The mathematical input for the Bandit Model.
    Matches the columns from 'features.py'.
    """
    total_spend: float = Field(..., ge=0, description="Total lifetime spend")
    spending_volatility: float = Field(
        ..., 
        ge=0.0, 
        description="Standard deviation of daily spending (Risk Metric)"
    )
    return_rate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Percentage of items returned (0.0 to 1.0)"
    )
    transaction_count: int = Field(
        ..., 
        ge=0, 
        description="Total number of transactions"
    )
    avg_transaction_value: float = Field(
        ..., 
        ge=0.0, 
        description="Average transaction size"
    )

    # Pydantic V2 Config for Documentation
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_spend": 500.50,
                "spending_volatility": 12.5,
                "transaction_count": 5,
                "avg_transaction_value": 100.10,
                "return_rate": 0.05
            }
        }
    )

# 2. The Request (What Android sends)
class PredictionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique User Identifier")
    test_group: str = "adaptive"  
    amount: float                 
    category: str                 
    features: UserFeatures
    # Helper method to bridge API to AI
    def to_dataframe(self) -> pd.DataFrame:
        """
        convert the request features directly into a pandas dataframe
        compatible with the BanditPreprocessor
        
        """
        # Convert nested model to dict
        data = self.features.model_dump()
        #create a  dataframe with one row
        return pd.DataFrame([data])

# 3. The Response (What Android receives)
class PredictionResponse(BaseModel):
    prediction_id: str = Field(..., description="UUID for this specific prediction. Android MUST keep this for feedback.")
    user_id: str
    strategy: str       # "EXPLORE" or "EXPLOIT"
    action: str         # "strict_budget", "quiz", etc.
    notification: str   # The LLM message
    visual_theme: str   # "red", "blue"
    debug_info: dict    # Confidence scores
    

# 4 The Feedback ( what android sends back)
class FeedbackRequest(BaseModel):
    """The lightweight payload the android app sends back."""
    prediction_id: str = Field(..., description= "The UUID returned by the predict endpoint")
    reward: float = Field(..., description= "1.0 if user engaged, 0.0 if dismissed/ignored")


class LeaderboardUpdateRequest(BaseModel):
    user_id: str
    anonymous_name: str
    xp: int
    tier: str