import pytest
import aiohttp
from aioresponses import aioresponses
from dolma_classifiers.label.api import BaseApiRequest, Message  # Replace with your actual module name


@pytest.fixture
def mock_api():
    with aioresponses() as m:
        yield m


@pytest.mark.asyncio
async def test_successful_api_request(mock_api):
    # Arrange
    endpoint = "https://api.example.com/v1/chat"
    expected_response = {
        "response": "Hello, world!",
        "status": "success"
    }

    mock_api.post(endpoint, status=200, payload=expected_response)

    request = BaseApiRequest(
        endpoint=endpoint,
        messages=[Message(role="user", content="Hello!")],
        headers={"Authorization": "Bearer test-token"}
    )

    # Act
    response = await request.make()

    # Assert
    assert response == expected_response


@pytest.mark.asyncio
async def test_api_request_with_error(mock_api):
    # Arrange
    endpoint = "https://api.example.com/v1/chat"
    error_response = {
        "error": "Invalid token",
        "status": "error"
    }

    mock_api.post(endpoint, status=401, payload=error_response)

    request = BaseApiRequest(
        endpoint=endpoint,
        messages=[Message(role="user", content="Hello!")],
        headers={"Authorization": "Bearer invalid-token"}
    )

    # Act & Assert
    with pytest.raises(aiohttp.ClientResponseError) as exc_info:
        await request.make()
    assert exc_info.value.status == 401


@pytest.mark.asyncio
async def test_api_request_payload(mock_api):
    # Arrange
    endpoint = "https://api.example.com/v1/chat"
    messages = [Message(role="user", content="Hello!")]
    parameters = {"temperature": 0.7}

    expected_payload = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    }

    def match_payload(url, **kwargs):
        assert kwargs['json'] == expected_payload
        return True

    mock_api.post(endpoint, status=200, callback=match_payload)

    request = BaseApiRequest(
        endpoint=endpoint,
        messages=messages,
        parameters=parameters
    )

    # Act
    await request.make()  # If no assertion error is raised, the payload matched


@pytest.mark.asyncio
async def test_network_error(mock_api):
    # Arrange
    endpoint = "https://api.example.com/v1/chat"
    mock_api.post(endpoint, exception=aiohttp.ClientConnectionError())

    request = BaseApiRequest(
        endpoint=endpoint,
        messages=[Message(role="user", content="Hello!")]
    )

    # Act & Assert
    with pytest.raises(aiohttp.ClientConnectionError):
        await request.make()
