"""
Included In Endpoint

Retrieves the branches and tags in which a change is included.
"""

from typing import Dict, Any
from src.collectors.base.base_api_client import BaseAPIClient


class IncludedInEndpoint(BaseAPIClient):
    """Included In Endpoint"""
    
    def get_endpoint_path(self, change_id: str, **kwargs) -> str:
        """Returns the endpoint path"""
        return f"changes/{change_id}/in"
    
    def fetch(self, change_id: str) -> Dict[str, Any]:
        """
        Retrieves the branches and tags in which a change is included.
        
        Args:
            change_id: Change ID
            
        Returns:
            IncludedInInfo entity
        """
        endpoint = self.get_endpoint_path(change_id=change_id)
        return self.make_request(endpoint)
