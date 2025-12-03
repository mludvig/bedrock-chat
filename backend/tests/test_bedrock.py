import json
import os
import sys
from pathlib import Path

import boto3
from ulid import ULID

os.environ["REGION"] = "us-west-2"
os.environ["BEDROCK_REGION"] = "us-west-2"
os.environ["ENABLE_BEDROCK_GLOBAL_INFERENCE"] = "true"
os.environ["ENABLE_BEDROCK_CROSS_REGION_INFERENCE"] = "true"

sys.path.append(".")

import unittest
from pprint import pprint
from unittest.mock import patch, MagicMock

# Load real AWS inference profile data for mocking
MOCK_DATA_FILE = Path(__file__).parent / "test_inference_profiles_mock_data.json"
with open(MOCK_DATA_FILE, 'r') as f:
    MOCK_INFERENCE_PROFILES = json.load(f)

from app.bedrock import (
    call_converse_api,
    compose_args_for_converse_api,
    get_model_id,
    find_inference_profile,
    get_model_id,
    list_inference_profiles,
)
from app.repositories.models.conversation import SimpleMessageModel, TextContentModel
from app.repositories.models.custom_bot_guardrails import BedrockGuardrailsModel
from app.routes.schemas.conversation import type_model_name

# MODEL: type_model_name = "claude-v3-haiku"
MODEL: type_model_name = "claude-v3.7-sonnet"


class TestGetModelId(unittest.TestCase):
    def setUp(self):
        """Clear cache before each test"""
        from app.bedrock import _profile_cache
        _profile_cache.clear()

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_cross_region_supported_model(self, mock_get_client):
        # us-east-1 has regional but not global profile for claude-v3.5-sonnet
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["us-east-1"]
        }
        mock_get_client.return_value = mock_client
        
        model = "claude-v3.5-sonnet"
        # Prefix with "us." to enable cross-region
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    def test_get_model_id_without_cross_region(self):
        # When both flags are disabled, should return base model ID without calling API
        model = "claude-v3.5-sonnet"
        # No prefix to disable cross-region
        expected_model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=False,
                enable_cross_region=False,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_unsupported_region_for_cross_region(self, mock_get_client):
        # Mock ap-northeast-1 (APAC region) profile data
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["ap-northeast-1"]
        }
        mock_get_client.return_value = mock_client
        
        model = "claude-v3.5-sonnet"
        # Should find apac.* prefix in ap-northeast-1 region
        expected_model_id = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="ap-northeast-1",
            ),
            expected_model_id,
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_global_inference_priority(self, mock_get_client):
        """Global inference is selected for supported model and region"""
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["us-east-1"]
        }
        mock_get_client.return_value = mock_client
        
        model = "claude-v4-sonnet"
        expected_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_without_global_with_cross_region_inference_priority(self, mock_get_client):
        # Regional inference is selected when global is disabled
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["us-east-1"]
        }
        mock_get_client.return_value = mock_client
        
        # Use claude-v3.5-sonnet which has US regional profile
        model = "claude-v3.5-sonnet"
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=False,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_regional_fallback(self, mock_get_client):
        """Falls back to regional cross-region for non-global supported models"""
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["us-east-1"]
        }
        mock_get_client.return_value = mock_client
        
        model = "claude-v3.5-sonnet"  # Non-global supported
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="us-east-1",
            ),
            expected_model_id,
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_unsupported_region_fallback(self, mock_get_client):
        """Falls back to regional for global-supported model in unsupported region"""
        mock_client = MagicMock()
        mock_client.list_inference_profiles.return_value = {
            "inferenceProfileSummaries": MOCK_INFERENCE_PROFILES["ap-south-1"]
        }
        mock_get_client.return_value = mock_client
        
        model = "claude-v4-sonnet"
        # ap-south-1 has apac.* regional profile but no global profile
        expected_model_id = "apac.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(
            get_model_id(
                model,
                enable_global=True,
                enable_cross_region=True,
                bedrock_region="ap-south-1",
            ),
            expected_model_id,
        )


class TestBedrockInferenceProfiles(unittest.TestCase):
    def setUp(self):
        """Clear cache before each test"""
        from app.bedrock import _profile_cache

        _profile_cache.clear()

    @patch("app.bedrock.get_bedrock_client")
    def test_list_inference_profiles_caching(self, mock_get_client):
        """Test that inference profiles are cached"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        # First call - should hit API
        profiles1 = list_inference_profiles("us-east-1")
        self.assertEqual(len(profiles1), 1)
        self.assertEqual(mock_client.list_inference_profiles.call_count, 1)

        # Second call - should use cache
        profiles2 = list_inference_profiles("us-east-1")
        self.assertEqual(len(profiles2), 1)
        self.assertEqual(
            mock_client.list_inference_profiles.call_count, 1
        )  # Still 1, not 2

    @patch("app.bedrock.get_bedrock_client")
    def test_find_global_inference_profile(self, mock_get_client):
        """Test finding global inference profile"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "global.anthropic.claude-sonnet-4-20250514-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        profile = find_inference_profile("claude-v4-sonnet", "us-east-1", "global")
        self.assertEqual(
            profile, "global.anthropic.claude-sonnet-4-20250514-v1:0"
        )

    @patch("app.bedrock.get_bedrock_client")
    def test_find_regional_inference_profile(self, mock_get_client):
        """Test finding regional inference profile"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        profile = find_inference_profile(
            "claude-v3.5-sonnet", "us-east-1", "regional"
        )
        self.assertEqual(profile, "us.anthropic.claude-3-5-sonnet-20240620-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_global_priority(self, mock_get_client):
        """Test that global inference profile is prioritized when enabled"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "global.anthropic.claude-sonnet-4-20250514-v1:0",
                    "status": "ACTIVE",
                },
                {
                    "inferenceProfileId": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "status": "ACTIVE",
                },
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        model_id = get_model_id(
            "claude-v4-sonnet",
            "us-east-1",
            enable_global=True,
            enable_cross_region=True,
        )
        self.assertEqual(model_id, "global.anthropic.claude-sonnet-4-20250514-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_with_regional_fallback(self, mock_get_client):
        """Test that regional profile is used when global is disabled"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        model_id = get_model_id(
            "claude-v3.5-sonnet",
            "us-east-1",
            enable_global=False,
            enable_cross_region=True,
        )
        self.assertEqual(model_id, "us.anthropic.claude-3-5-sonnet-20240620-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_get_model_id_base_fallback(self, mock_get_client):
        """Test fallback to base model ID when no profiles available"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_response = {"inferenceProfileSummaries": []}
        mock_client.list_inference_profiles.return_value = mock_response

        model_id = get_model_id(
            "claude-v3.5-sonnet",
            "us-east-1",
            enable_global=False,
            enable_cross_region=False,
        )
        self.assertEqual(model_id, "anthropic.claude-3-5-sonnet-20240620-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_api_error_handling(self, mock_get_client):
        """Test that API errors are handled gracefully"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.list_inference_profiles.side_effect = Exception("API Error")

        # Should not raise, should return base model ID
        model_id = get_model_id(
            "claude-v3.5-sonnet",
            "us-east-1",
            enable_global=False,
            enable_cross_region=False,
        )
        self.assertEqual(model_id, "anthropic.claude-3-5-sonnet-20240620-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_both_inference_types_disabled(self, mock_get_client):
        """Test that base model ID is used when both global and regional are disabled"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Even if profiles exist, they shouldn't be queried
        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "global.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        model_id = get_model_id(
            "claude-v3.5-sonnet",
            "us-east-1",
            enable_global=False,
            enable_cross_region=False,
        )

        # Should return base model ID directly without trying inference profiles
        self.assertEqual(model_id, "anthropic.claude-3-5-sonnet-20240620-v1:0")

    @patch("app.bedrock.get_bedrock_client")
    def test_model_without_inference_profile_support(self, mock_get_client):
        """Test models that don't have inference profiles (e.g., Mistral, older models)"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # API returns profiles but not for this model
        mock_response = {
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileId": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "status": "ACTIVE",
                }
            ]
        }
        mock_client.list_inference_profiles.return_value = mock_response

        # Test with Mistral model that doesn't have inference profiles
        model_id = get_model_id(
            "mistral-7b-instruct",
            "us-east-1",
            enable_global=True,
            enable_cross_region=True,
        )

        # Should fallback to base model ID since no matching profile exists
        self.assertEqual(model_id, "mistral.mistral-7b-instruct-v0:2")

    @patch("app.bedrock.get_bedrock_client")
    def test_model_in_region_without_inference_profile(self, mock_get_client):
        """Test when a model is in a region but no inference profile exists"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Empty response - no inference profiles available in this region
        mock_response = {"inferenceProfileSummaries": []}
        mock_client.list_inference_profiles.return_value = mock_response

        model_id = get_model_id(
            "claude-v3.5-sonnet",
            "ap-south-1",  # Region that might not have all profiles
            enable_global=True,
            enable_cross_region=True,
        )

        # Should fallback to base model ID
        self.assertEqual(model_id, "anthropic.claude-3-5-sonnet-20240620-v1:0")


class TestCallConverseApi(unittest.TestCase):
    def test_call_converse_api_with_global_inference(self):
        """Actual LLM call using global inference profile"""
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello! Please respond with just 'Global inference works!'",
                )
            ],
        )

        # Use global inference supported model
        arg = compose_args_for_converse_api(
            [message],
            "claude-v4-sonnet",  # Global inference supported
            stream=False,
        )

        # Verify modelId is global profile
        expected_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
        self.assertEqual(arg["modelId"], expected_model_id)

        # Actual API call
        response = call_converse_api(arg)

        # Verify basic response structure
        self.assertIn("output", response)
        self.assertIn("message", response["output"])
        self.assertIn("content", response["output"]["message"])

        print(f"Global inference response: {response['output']['message']['content']}")

    def test_call_converse_api_with_regional_fallback(self):
        """Actual LLM call with regional cross-region fallback"""
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello! Please respond with just 'Regional inference works!'",
                )
            ],
        )

        # Use non-global supported model
        arg = compose_args_for_converse_api(
            [message],
            "claude-v3.5-sonnet",  # Non-global supported, regional supported
            stream=False,
        )

        # Verify modelId is regional profile
        expected_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
        self.assertEqual(arg["modelId"], expected_model_id)

        # Actual API call
        response = call_converse_api(arg)

        # Verify basic response structure
        self.assertIn("output", response)
        self.assertIn("message", response["output"])
        self.assertIn("content", response["output"]["message"])

        print(
            f"Regional inference response: {response['output']['message']['content']}"
        )

    def test_call_converse_api(self):
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello, World!",
                )
            ],
        )
        arg = compose_args_for_converse_api(
            [message],
            MODEL,
            stream=False,
        )

        response = call_converse_api(arg)
        pprint(response)


class TestCallConverseApiWithGuardrails(unittest.TestCase):
    def setUp(self):
        # Note that the region must be the same as the one used in the bedrock client
        # https://github.com/aws/aws-sdk-js-v3/issues/6482
        self.bedrock_client = boto3.client("bedrock", region_name="us-east-1")
        self.guardrail_name = f"test-guardrail-{ULID()}"

        # Create dummy guardrail
        res = self.bedrock_client.create_guardrail(
            name=self.guardrail_name,
            description="Test guardrail for unit tests",
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "SEXUAL", "inputStrength": "LOW", "outputStrength": "LOW"},
                ]
            },
            blockedInputMessaging="blocked",
            blockedOutputsMessaging="blocked",
        )

        res_ver = self.bedrock_client.create_guardrail_version(
            guardrailIdentifier=res["guardrailArn"],
        )

        self.guardrail = BedrockGuardrailsModel(
            is_guardrail_enabled=True,
            hate_threshold=0,
            insults_threshold=0,
            sexual_threshold=1,
            violence_threshold=0,
            misconduct_threshold=0,
            grounding_threshold=0,
            relevance_threshold=0,
            guardrail_arn=res["guardrailArn"],
            guardrail_version=res_ver["version"],
            # guardrail_version="DRAFT",
        )
        self.guardrail_arn = res["guardrailArn"]

    def tearDown(self):
        print("Cleaning up...")
        # Delete dummy guardrail
        try:
            self.bedrock_client.delete_guardrail(guardrailIdentifier=self.guardrail_arn)

        except Exception as e:
            print(f"Error deleting guardrail: {e}")

    def test_call_converse_api_with_guardrails(self):
        message = SimpleMessageModel(
            role="user",
            content=[
                TextContentModel(
                    content_type="text",
                    body="Hello, World!",
                )
            ],
        )
        arg = compose_args_for_converse_api(
            [message],
            MODEL,
            guardrail=self.guardrail,
            stream=False,
        )

        pprint(arg)

        response = call_converse_api(arg)
        pprint(response)


if __name__ == "__main__":
    unittest.main()
