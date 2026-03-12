import os
import logging
from supabase import create_client, Client

Logger = logging.getLogger(__name__)

class LeaderboardService:
    def __init__(self):
        #Initialize supabase 
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")

        if not url or not key:
            Logger.error("Supabase credentials missing in .env")
            raise ValueError("Supabase credentials missing")
        self.supabase: Client = create_client(url,key)
    

    def update_user_xp(self, user_id: str, anonymous_name: str, xp: int, tier: str):
        """Upserts the users latest Xp and tier to the leaderboard"""
        try:
            data = {
                "user_id": user_id,
                "anonymous_name":anonymous_name,
                "xp":xp,
                "tier":tier
            }
            #Update if exists ,Insert if it's a new user
            response = self.supabase.table("leaderboard").upsert(data).execute()
            return response.data
        except Exception as e:
            Logger.error(f"Failed to update leaderboard for {user_id}: {e}")

    def get_top_50(self):
        """Fetches the top 50 users ranked by xp."""
        try:
            #select everything, order by highest xp , limit to 50
            response = self.supabase.table("leaderboard")\
                .select("anonymous_name, xp,tier")\
                .order("xp",desc=True)\
                .limit(50)\
                .execute()
            return response.data
        except Exception as e:
            Logger.error(f"Failed to fetch Leaderboard: {e}")
            raise e