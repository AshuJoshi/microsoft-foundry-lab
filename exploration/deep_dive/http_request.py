"""
Simplified HTTP request function for Azure AI Agent Framework.
Adapted from AWS Strands Agent Framework to work as a function tool.

This module provides HTTP request functionality with authentication support.
Compatible with the Azure AI Agent's function tool pattern.
"""

import base64
import json
import os
from typing import Annotated, Optional

import requests
from pydantic import Field


def http_request(
    method: Annotated[str, Field(description="HTTP method (GET, POST, PUT, DELETE, etc.)")],
    url: Annotated[str, Field(description="The URL to send the request to")],
    auth_type: Annotated[
        Optional[str],
        Field(description="Authentication type: Bearer, token, basic, api_key, etc."),
    ] = None,
    auth_token: Annotated[
        Optional[str],
        Field(description="Authentication token (if not using auth_env_var)"),
    ] = None,
    auth_env_var: Annotated[
        Optional[str],
        Field(description="Name of environment variable containing the auth token"),
    ] = None,
    headers: Annotated[
        Optional[dict],
        Field(description="HTTP headers as key-value pairs"),
    ] = None,
    body: Annotated[
        Optional[str],
        Field(description="Request body (for POST, PUT, etc.)"),
    ] = None,
    verify_ssl: Annotated[
        bool,
        Field(description="Whether to verify SSL certificates"),
    ] = True,
    basic_auth_username: Annotated[
        Optional[str],
        Field(description="Username for basic authentication"),
    ] = None,
    basic_auth_password: Annotated[
        Optional[str],
        Field(description="Password for basic authentication"),
    ] = None,
) -> str:
    """
    Make HTTP requests to any API with authentication support.
    
    Supports multiple authentication methods:
    - Bearer token: auth_type="Bearer", auth_token="your-token"
    - GitHub token: auth_type="token", auth_env_var="GITHUB_TOKEN"
    - Basic auth: auth_type="basic", basic_auth_username="user", basic_auth_password="pass"
    - API key: auth_type="api_key", auth_token="your-key"
    
    Examples:
        # GET request with Bearer token
        http_request("GET", "https://api.example.com/data", auth_type="Bearer", auth_token="abc123")
        
        # POST request with JSON body
        http_request("POST", "https://api.example.com/users", body='{"name": "John"}', 
                    headers={"Content-Type": "application/json"})
        
        # GitHub API request
        http_request("GET", "https://api.github.com/user", auth_type="token", auth_env_var="GITHUB_TOKEN")
    """
    try:
        # Process headers
        request_headers = headers.copy() if headers else {}
        
        # Handle authentication
        token = auth_token
        if not token and auth_env_var:
            token = os.getenv(auth_env_var)
            if not token:
                return f"Error: Environment variable '{auth_env_var}' not found or empty"
        
        if token and auth_type:
            if auth_type == "Bearer":
                request_headers["Authorization"] = f"Bearer {token}"
            elif auth_type == "token":
                # GitHub API uses 'token' prefix
                request_headers["Authorization"] = f"token {token}"
                if "Accept" not in request_headers and "github" in url.lower():
                    request_headers["Accept"] = "application/vnd.github.v3+json"
            elif auth_type == "api_key":
                request_headers["X-API-Key"] = token
            elif auth_type == "custom":
                request_headers["Authorization"] = token
        
        # Handle basic authentication
        auth = None
        if auth_type == "basic" and basic_auth_username and basic_auth_password:
            credentials = base64.b64encode(
                f"{basic_auth_username}:{basic_auth_password}".encode()
            ).decode()
            request_headers["Authorization"] = f"Basic {credentials}"
        
        # Make the request
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=request_headers,
            data=body,
            verify=verify_ssl,
            auth=auth,
            timeout=30,
        )
        
        # Format response
        result = []
        result.append(f"Status: {response.status_code} {response.reason}")
        
        # Add URL info
        result.append(f"URL: {response.url}")
        
        # Add important headers
        content_type = response.headers.get("Content-Type", "unknown")
        result.append(f"Content-Type: {content_type}")
        
        # Add response body
        try:
            # Try to parse as JSON for pretty printing
            if "application/json" in content_type:
                json_data = response.json()
                result.append(f"Body: {json.dumps(json_data, indent=2)}")
            else:
                # Return text content
                text = response.text
                # Truncate very long responses
                if len(text) > 5000:
                    result.append(f"Body (truncated): {text[:5000]}... (total length: {len(text)})")
                else:
                    result.append(f"Body: {text}")
        except Exception:
            result.append(f"Body: {response.text[:1000]}")
        
        return "\n".join(result)
        
    except requests.exceptions.RequestException as e:
        return f"HTTP Request Error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
