
import json
import logging
import sys
from fastapi import Security
from pathlib import Path
from typing import Dict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException 


logger = logging.getLogger(__name__)


auth_scheme = HTTPBearer()
project_root = Path(__file__).resolve().parent.parent.parent
tenant_file = project_root / 'tenants.json'

try:
    with open(tenant_file, 'r') as f:
        TENANT_CONFIGS = json.load(f)
        print(f"Tenant configurations loaded successfully.Len = {len(TENANT_CONFIGS)}")
except FileNotFoundError:
    logger.critical("FATAL ERROR: tenants.json config file not found.")
    print("___server is shutting down____")
    sys.exit(1)
    TENANT_CONFIGS = {}
except json.JSONDecodeError:
    logger.critical("FATAL ERROR: tenants.json is not valid JSON.")
    TENANT_CONFIGS = {}

async def get_shop_info(
    token: HTTPAuthorizationCredentials = Security(auth_scheme)
) -> Dict:
    """
    Validates the API key (Bearer token) by checking the tenants.json file.
    """
    api_key = token.credentials
    shop_info = TENANT_CONFIGS.get(api_key)
    
    if not shop_info:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    return shop_info