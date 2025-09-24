import logging
import json
import os
import boto3
from botocore.exceptions import ClientError

from app.agents.tools.agent_tool import AgentTool
from app.repositories.models.custom_bot import BotModel
from app.routes.schemas.conversation import type_model_name
from app.utils import get_bedrock_runtime_client
from duckduckgo_search import DDGS
from firecrawl.firecrawl import FirecrawlApp
from pydantic import BaseModel, Field, root_validator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InternetSearchInput(BaseModel):
    query: str = Field(description="The query to search for on the internet.")
    locale: str = Field(
        default="en-us",
        description="The country code and language code for the search. Must be `{language}-{country}` for example `jp-jp` (Japanese - Japan), `zh-cn` (Chinese - China), `en-ca` (English - Canada), `fr-ca` (French - Canada), `en-nz` (English - New Zealand), etc. If empty the default is `en-us`.",
    )
    time_limit: str = Field(
        description="Retrieve only the most recent results, for example `1w` only returns the results from the last week. Units are 'd' (day), 'w' (week), 'm' (month), 'y' (year). Use empty string to retrieve all results."
    )

    @root_validator(pre=True)
    def validate_locale(cls, values):
        locale = values.get("locale")
        # Basic validation for locale format
        if not locale or locale.count("-") != 1:
            # Get the default value from the field definition
            default_locale = cls.__fields__["locale"].default
            values["locale"] = default_locale
        return values


def _get_internet_search_config():
    """Get internet search configuration from Secrets Manager.

    Returns:
        dict: Configuration with searchEngine, apiKey, maxResults
        None: If secret not found or error occurred
    """
    try:
        secret_name = os.environ.get("INTERNET_SEARCH_SECRET_NAME")
        if not secret_name:
            logger.warning("INTERNET_SEARCH_SECRET_NAME environment variable not set")
            return None

        secrets_client = boto3.client("secretsmanager")  # type: ignore
        response = secrets_client.get_secret_value(SecretId=secret_name)
        config = json.loads(response["SecretString"])

        logger.info(
            f"Retrieved internet search config: engine={config.get('searchEngine')}"
        )
        return config

    except ClientError as e:
        logger.warning(f"Failed to get internet search config: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in internet search secret: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting internet search config: {e}")
        return None


def _summarize_content(content: str, title: str, url: str, query: str) -> str:
    """
    Summarize content using Claude 3 Haiku to prevent context window bloat.
    Returns a concise summary (500-800 tokens max) preserving key information.
    """
    try:
        client = get_bedrock_runtime_client()

        # Truncate content if it's too long to avoid token limits
        max_input_length = 8000  # Conservative limit for input
        if len(content) > max_input_length:
            content = content[:max_input_length] + "..."

        prompt = f"""Please provide a concise summary of the following web content in 500-800 tokens maximum. Focus on information that directly answers or relates to the user's query: "{query}"

Title: {title}
URL: {url}
Content: {content}

Summary:"""

        response = client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}],
                }
            ),
        )

        response_body = json.loads(response["body"].read())
        summary = response_body["content"][0]["text"].strip()

        logger.info(
            f"Summarized content from {len(content)} chars to {len(summary)} chars"
        )
        return summary

    except Exception as e:
        logger.error(f"Error summarizing content: {e}")
        # Fallback: return truncated content if summarization fails
        fallback_content = content[:1000] + "..." if len(content) > 1000 else content
        logger.info(f"Using fallback content: {len(fallback_content)} chars")
        return fallback_content


def _search_with_duckduckgo(query: str, time_limit: str, locale: str) -> list:
    # Incoming locale expected as language-country (e.g. 'en-nz'). DDGS prefers country-language, so swap.
    language, country = locale.split("-", 1)
    REGION = f"{country}-{language}".lower()
    SAFE_SEARCH = "moderate"
    MAX_RESULTS = 20
    logger.info(
        f"Executing DuckDuckGo search with query: {query}, region: {REGION}, time_limit: {time_limit}"
    )
    with DDGS() as ddgs:
        results = list(
            ddgs.text(
                keywords=query,
                region=REGION,
                safesearch=SAFE_SEARCH,
                timelimit=time_limit,
                max_results=MAX_RESULTS,
            )
        )
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results")

        # Summarize each result to prevent context bloat
        summarized_results = []
        for result in results:
            title = result["title"]
            url = result["href"]
            content = result["body"]

            # Summarize the content
            summary = _summarize_content(content, title, url, query)

            summarized_results.append(
                {
                    "content": summary,
                    "source_name": title,
                    "source_link": url,
                }
            )

        return summarized_results


def _search_with_firecrawl(
    query: str, api_key: str, locale: str, max_results: int = 10
) -> list:
    logger.info(
        f"Searching with Firecrawl. Query: {query}, Max Results: {max_results}, Locale: {locale}"
    )

    try:
        app = FirecrawlApp(api_key=api_key)

        # Incoming locale is language-country (e.g. 'en-us').
        language, country = locale.split("-", 1)
        results = app.search(
            query,
            {
                "limit": max_results,
                "lang": language,
                "location": country,
                "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
            },
        )

        if not results:
            logger.warning("No results found")
            return []
        logger.info(f"results of firecrawl: {results}")

        # Format and summarize search results
        search_results = []
        for data in results.get("data", []):
            if isinstance(data, dict):
                title = data.get("title", "")
                url = data.get("metadata", {}).get("sourceURL", "")
                content = data.get("markdown", {})

                # Summarize the content
                summary = _summarize_content(content, title, url, query)

                search_results.append(
                    {
                        "content": summary,
                        "source_name": title,
                        "source_link": url,
                    }
                )

        logger.info(f"Found {len(search_results)} results from Firecrawl")
        return search_results

    except Exception as e:
        logger.error(f"Error searching with Firecrawl: {e}")
        raise e


def _internet_search(
    tool_input: InternetSearchInput, bot: BotModel | None, model: type_model_name | None
) -> list:
    query = tool_input.query
    time_limit = tool_input.time_limit
    locale = tool_input.locale

    logger.info(
        f"Internet search request - Query: {query}, Time Limit: {time_limit}, Locale: {locale}"
    )

    # Get centralized internet search configuration
    internet_search_config = _get_internet_search_config()

    if (
        internet_search_config
        and internet_search_config.get("searchEngine") == "firecrawl"
    ):
        api_key = internet_search_config.get("apiKey")
        max_results = internet_search_config.get("maxResults", 10)

        if api_key:
            try:
                logger.info("Using centralized Firecrawl configuration")
                return _search_with_firecrawl(
                    query=query,
                    api_key=api_key,
                    locale=locale,
                    max_results=max_results,
                )
            except Exception as e:
                logger.error(
                    f"Error with centralized Firecrawl search, falling back to DuckDuckGo: {e}"
                )
        else:
            logger.warning(
                "Firecrawl configured but API key is empty, falling back to DuckDuckGo"
            )

    # Default to DuckDuckGo (either configured as default or fallback)
    logger.info("Using DuckDuckGo search")
    return _search_with_duckduckgo(query, time_limit, locale)


internet_search_tool = AgentTool(
    name="internet_search",
    description="Search the internet for information.",
    args_schema=InternetSearchInput,
    function=_internet_search,
)
