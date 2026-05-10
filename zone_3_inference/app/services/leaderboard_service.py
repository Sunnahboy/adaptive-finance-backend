import os
import logging
from supabase import create_client, Client

# Set up a professional logger for this module
Logger = logging.getLogger(__name__)

class LeaderboardService:
    """
    A dedicated service class to handle all Gamification and Leaderboard 
    database operations via Supabase.
    """
    
    def __init__(self) -> None:
        """Initializes the Supabase client using environment variables."""
        url: str | None = os.getenv("SUPABASE_URL")
        key: str | None = os.getenv("SUPABASE_KEY")

        if not url or not key:
            Logger.error("CRITICAL: Supabase credentials missing in .env file.")
            raise ValueError("Supabase credentials missing. Check your environment variables.")
            
        self.supabase: Client = create_client(url, key)
    
    def update_user_xp(self, user_id: str, anonymous_name: str, xp: int, tier: str) -> dict:
        """
        Upserts the user's latest XP and tier to the leaderboard.
        If the user doesn't exist, they are created. If they do, their stats are updated.
        """
        try:
            data = {
                "user_id": user_id,
                "anonymous_name": anonymous_name,
                "xp": xp,
                "tier": tier
            }
            
            response = self.supabase.table("leaderboard").upsert(data).execute()
            Logger.info(f"Successfully synced XP for user: {anonymous_name}")
            return response.data
            
        except Exception as e:
            Logger.error(f"Failed to update leaderboard for {user_id}: {e}")
            raise e  # We raise the error so the FastAPI router knows to return a 500 status!

    def get_top_50(self) -> list[dict]:
        """
        Fetches the top 50 users ranked by XP descending.
        Includes user_id and cheers to support interactive frontend UI features.
        """
        try:
            response = self.supabase.table("leaderboard")\
                .select("user_id, anonymous_name, xp, tier, cheers")\
                .order("xp", desc=True)\
                .limit(50)\
                .execute()
                
            return response.data
            
        except Exception as e:
            Logger.error(f"Failed to fetch Top 50 Leaderboard: {e}")
            raise e
    

    def register_cheer(self, target_user_id: str) -> dict:
        """
        Increments the public cheer count for a specific user when they 
        receive a double-tap interaction from another community member.
        """
        try:
            # 1. Fetch the current cheer count
            user_data = self.supabase.table("leaderboard").select("cheers").eq("user_id", target_user_id).execute()
            
            if user_data.data:
                # Safely get current cheers, defaulting to 0 if none exist
                current_cheers = user_data.data[0].get("cheers", 0) or 0
                new_cheers = current_cheers + 1
                
                # 2. Update the database with the new total
                self.supabase.table("leaderboard").update({"cheers": new_cheers}).eq("user_id", target_user_id).execute()
                Logger.info(f"🎉 Cheer added! User {target_user_id} now has {new_cheers} cheers.")
                
                return {"status": "success", "message": "Cheer added", "new_total": new_cheers}
            else:
                Logger.warning(f"Attempted to cheer unknown user: {target_user_id}")
                return {"status": "error", "message": "Target user not found on leaderboard"}
                
        except Exception as e:
            Logger.error(f"Failed to process cheer for {target_user_id}: {e}")
            raise e
        

        
    def get_historical_winners(self) -> list[dict]:
        """Fetches the last 5 weeks of top-3 winners for the Hall of Fame."""
        try:
            response = self.supabase.table("leaderboard_history")\
                .select("*")\
                .order("won_at", desc=True)\
                .limit(15)\
                .execute()
            return response.data
        except Exception as e:
            Logger.error(f"Failed to fetch Hall of Fame history: {e}")
            raise e