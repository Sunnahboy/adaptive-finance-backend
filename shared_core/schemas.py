
from pydantic import BaseModel, Field, ConfigDict

# 1. The Features (Strict Rules)
class UserFeatures(BaseModel):
    """
    The mathematical input for the Bandit Model.
    Matches the columns from 'features.py'.
    """
    total_spend: float = Field(..., ge=0, description="Total lifetime spend")
    spending_volatility: float = Field(
        ..., 
        ge=0.0, 
        le=20000.0, # Reasonable cap to prevent outliers
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
    features: UserFeatures

# 3. The Response (What Android receives)
class PredictionResponse(BaseModel):
    user_id: str
    strategy: str       # "EXPLORE" or "EXPLOIT"
    action: str         # "strict_budget", "quiz", etc.
    notification: str   # The LLM message
    visual_theme: str   # "red", "blue"
    debug_info: dict    # Confidence scores