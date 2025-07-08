import os
from dotenv import load_dotenv
from openai import OpenAI
import time
import requests

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key:
    print(f'API key found: {api_key[:10]}...{api_key[-4:]}')
    print('Length:', len(api_key))
else:
    print('No API key found')
    exit(1)

# Test 1: Basic HTTP connectivity
print('\n=== Test 1: Basic HTTP connectivity ===')
try:
    response = requests.get('https://httpbin.org/status/200', timeout=10)
    print(f'Basic HTTPS works: {response.status_code}')
except Exception as e:
    print(f'Basic HTTPS failed: {e}')

# Test 2: OpenAI API endpoint connectivity
print('\n=== Test 2: OpenAI API endpoint ===')
try:
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    response = requests.get('https://api.openai.com/v1/models', headers=headers, timeout=30)
    print(f'OpenAI API endpoint status: {response.status_code}')
    if response.status_code == 200:
        print('✅ Direct API access works!')
    else:
        print(f'❌ API returned: {response.text[:200]}')
except Exception as e:
    print(f'❌ Direct API access failed: {e}')

# Test 3: OpenAI Python client with different settings
print('\n=== Test 3: OpenAI Python client ===')
try:
    print('Testing OpenAI client...')
    client = OpenAI(
        api_key=api_key,
        timeout=60.0,  # Longer timeout
        max_retries=3
    )
    
    # Test with a simple completion
    response = client.chat.completions.create(
        model='gpt-4o-mini',  # Updated to use new default model
        messages=[
            {'role': 'user', 'content': 'Say hello in one word'}
        ],
        max_tokens=10
    )
    
    print('✅ OpenAI connection successful!')
    print(f'Response: {response.choices[0].message.content}')
    
except Exception as e:
    print(f'❌ OpenAI connection failed: {e}')
    print(f'Error type: {type(e).__name__}')
    

    # Additional debugging
    if hasattr(e, 'response'):
        print(f'Response status: {e.response.status_code if e.response else "No response"}')
    
# Test 4: Alternative model
print('\n=== Test 4: Alternative model ===')
try:
    client = OpenAI(api_key=api_key, timeout=60.0)
    response = client.chat.completions.create(
        model='gpt-4o-mini',  # Try a different model
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tokens=5
    )
    print('✅ Alternative model works!')
    print(f'Response: {response.choices[0].message.content}')
except Exception as e:
    print(f'❌ Alternative model failed: {e}')