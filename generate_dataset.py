#!/usr/bin/env python3
"""
Generate realistic-looking secret strings using cryptographically secure randomness.

This script generates various types of strings that resemble real secrets like
UUIDs, SSH keys, API keys, etc., using the `secrets` module for true randomness.
"""

import secrets
import string
import json
import sys
import random
from typing import List

# Initialize random with cryptographically secure seed
random.seed(secrets.randbits(256))


def generate_uuid() -> str:
    """Generate a realistic UUID v4.

    Example: f5e63e02-d9ca-49dc-b6bc-999eb635030c
    """
    hex_chars = string.hexdigits.lower()[:16]

    # UUID format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    # where y is one of 8, 9, a, or b
    parts = [
        ''.join(secrets.choice(hex_chars) for _ in range(8)),
        ''.join(secrets.choice(hex_chars) for _ in range(4)),
        '4' + ''.join(secrets.choice(hex_chars) for _ in range(3)),
        secrets.choice('89ab') + ''.join(secrets.choice(hex_chars) for _ in range(3)),
        ''.join(secrets.choice(hex_chars) for _ in range(12))
    ]
    return '-'.join(parts)


def generate_ssh_private_key() -> str:
    """Generate a realistic-looking private key format (sanitized).

    Emulates: SSH RSA private keys
    Example:
    _____START DUMMY NUMBER LETTERS_____
    CWsDPGco10x4wcX01ZCm4wY9fNM+h4SpQU0PcetOoF9przaIGlh+rLzLh+NgbrFx
    ...
    _____FINISH DUMMY NUMBER LETTERS_____
    """
    # Base64 characters
    b64_chars = string.ascii_letters + string.digits + '+/='

    # Generate random base64-like content with sanitized headers
    lines = ['=====START DUMMY NUMBER LETTERS=====']

    # Typical key has about 25-27 lines of 64 characters
    num_lines = secrets.randbelow(3) + 25
    for i in range(num_lines):
        if i == num_lines - 1:
            # Last line is typically shorter
            line_len = secrets.randbelow(32) + 16
        else:
            line_len = 64
        lines.append(''.join(secrets.choice(b64_chars) for _ in range(line_len)))

    lines.append('=====FINISH DUMMY NUMBER LETTERS=====')
    return '\n'.join(lines)


def generate_ssh_public_key() -> str:
    """Generate a realistic-looking public key format (sanitized).

    Emulates: SSH RSA public keys
    Example: llq-ynz AAAAB3NzaC1yc2EAAAADAQABAAABgQC7... person&location
    """
    b64_chars = string.ascii_letters + string.digits + '+/='

    # Public keys are typically 300-400 characters
    key_len = secrets.randbelow(100) + 300
    key_data = ''.join(secrets.choice(b64_chars) for _ in range(key_len))

    # Format: llq-ynz <key_data> comment (sanitized version)
    return f"llq-ynz {key_data} person&location"


def generate_medium_b64() -> str:
    """Generate a medium-length base64 string (single line).

    Around half the length of a typical SSH private key (~800 characters).
    Uses the same base64 character set but without headers/footers.
    """
    b64_chars = string.ascii_letters + string.digits + '+/='

    # Generate around 800 characters (+/- some randomness)
    length = secrets.randbelow(100) + 750
    return ''.join(secrets.choice(b64_chars) for _ in range(length))


def generate_long_b64() -> str:
    """Generate a long base64 string (single line).

    Around twice the length of a typical SSH private key (~3200 characters).
    Uses the same base64 character set but without headers/footers.
    """
    b64_chars = string.ascii_letters + string.digits + '+/='

    # Generate around 3200 characters (+/- some randomness)
    length = secrets.randbelow(200) + 3100
    return ''.join(secrets.choice(b64_chars) for _ in range(length))


def generate_aws_access_key() -> str:
    """Generate a realistic-looking access key ID (sanitized).

    Emulates: AWS access key IDs
    Example: OCTOE8VZVJFWZWKML3Q0
    """
    # Access keys start with XYZW and are 20 characters total (sanitized)
    chars = string.ascii_uppercase + string.digits
    return 'OCTO' + ''.join(secrets.choice(chars) for _ in range(16))


def generate_aws_secret_key() -> str:
    """Generate a realistic-looking secret access key (sanitized).

    Emulates: AWS secret access keys
    Example: iYKY9AhyrpcEHTia8/7XzGS+/oV3pHCzULMywP7+
    """
    # Secret keys are 40 characters of base64-like strings
    chars = string.ascii_letters + string.digits + '+/'
    return ''.join(secrets.choice(chars) for _ in range(40))


def generate_github_token() -> str:
    """Generate a realistic-looking personal access token (sanitized).

    Emulates: GitHub personal access tokens
    Example: ywn_Dq47zTfEjoie3qCM2nJBQJEEij454er4mN4c
    """
    # Tokens start with abc_ and are followed by 36 alphanumeric chars (sanitized)
    chars = string.ascii_letters + string.digits
    return 'dum_' + ''.join(secrets.choice(chars) for _ in range(36))


def generate_api_key() -> str:
    """Generate a generic API key.

    Example: 791108a2bb989e93de1e5945d7dff023
    """
    # Generic 32-character hex string
    hex_chars = string.hexdigits.lower()[:16]
    return ''.join(secrets.choice(hex_chars) for _ in range(32))


def generate_bearer_token() -> str:
    """Generate a realistic-looking bearer token.

    Example: rd8zDapamsexIbMUFGzmNhWCgqsooNnjmJz_Y4
    """
    # Random base64-like string, typically 32-64 characters
    chars = string.ascii_letters + string.digits + '-_'
    length = secrets.randbelow(33) + 32
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_jwt() -> str:
    """Generate a realistic-looking JWT token.

    Example: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9+eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ+SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c
    """
    b64_chars = string.ascii_letters + string.digits + '-_'

    # JWT has three parts separated by dots
    header_len = secrets.randbelow(20) + 20
    payload_len = secrets.randbelow(50) + 50
    signature_len = secrets.randbelow(30) + 30

    header = ''.join(secrets.choice(b64_chars) for _ in range(header_len))
    payload = ''.join(secrets.choice(b64_chars) for _ in range(payload_len))
    signature = ''.join(secrets.choice(b64_chars) for _ in range(signature_len))

    return f"{header}+{payload}+{signature}"


def generate_password() -> str:
    """Generate a realistic strong passphrase (sanitized).

    Emulates: Strong passwords
    Example: s**ou5)pTTZmh>6Evm},8p?
    """
    # Mix of various character types (same complexity, different name)
    chars = string.ascii_letters + string.digits + '!@#$%^&*()_+-=[]{}|;:,.<>?'
    length = secrets.randbelow(17) + 16  # 16-32 characters
    return ''.join(secrets.choice(chars) for _ in range(length))


def generate_username() -> str:
    """Generate a realistic username.

    Example: hxgapx796, mtq514, ukxld_lwf, gluc.wpkvyl
    """
    # Common username patterns
    patterns = [
        lambda: ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(8) + 5)),  # simple lowercase
        lambda: ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(5) + 3)) + str(secrets.randbelow(1000)),  # name + number
        lambda: ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(4) + 3)) + '_' + ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(4) + 3)),  # name_name
        lambda: ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(5) + 3)) + '.' + ''.join(secrets.choice(string.ascii_lowercase) for _ in range(secrets.randbelow(5) + 3)),  # first.last
    ]
    return secrets.choice(patterns)()


def generate_ip_address() -> str:
    """Generate a realistic IP address.

    Example: 96.65.25.217, 192.168.1.100, 10.0.0.5
    """
    # Generate random IP address (mix of public and private ranges)
    octets = [
        secrets.randbelow(256),
        secrets.randbelow(256),
        secrets.randbelow(256),
        secrets.randbelow(256)
    ]
    return '.'.join(str(octet) for octet in octets)


def generate_user_credentials() -> str:
    """Generate user info with username, passphrase, and network address (sanitized).

    Emulates: User credential files
    Example:
    identifier: hxgapx796
    random_chars: s**ou5)pTTZmh>6Evm},8p?
    sim_network_addr: 96.65.25.217
    """
    username = generate_username()
    password = generate_password()
    ip_addr = generate_ip_address()

    return f"identifier: {username}\nrandom_chars: {password}\nsim_network_addr: {ip_addr}"


def generate_email() -> str:
    """Generate realistic email addresses.

    Example: gluc.wpkvyl@test.org, fzgap_vse+tag@app.net, manager@localhost
    """
    patterns = [
        lambda: f"{generate_username()}@{secrets.choice(['outlook.com', 'gmail.com', 'unicef.org', 'bbc.co.uk'])}",
        lambda: f"{generate_username()}.{generate_username()}@{secrets.choice(['yahoo.com', 'university.edu', 'whales.org'])}",
        lambda: f"{generate_username()}+tag@{secrets.choice(['fanta.io', 'apply.net', 'platform.dev'])}",
        lambda: f"manager@{secrets.choice(['localhost', 'service', 'host'])}",
    ]
    return secrets.choice(patterns)()


def generate_url() -> str:
    """Generate realistic URLs.

    Example: https://github.com/config/settings, http://localhost:8080/user/6110
    """
    protocols = ['http://', 'https://']
    domains = ['plumage.com', 'api.duffel.com', 'localhost:8080', 'github.com', 'grizzlies.org']
    paths = [
        '/api/v1/users',
        '/repos/owner/repo',
        '/config/settings',
        '/admin/dashboard',
        f'/user/{secrets.randbelow(9999)}',
        '/data.json',
        '',
    ]

    protocol = secrets.choice(protocols)
    domain = secrets.choice(domains)
    path = secrets.choice(paths)

    return f"{protocol}{domain}{path}"


def generate_json_object() -> str:
    """Generate small JSON objects (sanitized).

    Emulates: Configuration JSON, API responses
    Example: {"dummy_user":"manager","dummy_phrase":"CW6^{N$JJzU<Xgs4"}, {"key":"value"}
    """
    patterns = [
        lambda: '{"key":"value"}',
        lambda: f'{{"dummy_user":"manager","dummy_phrase":"{generate_password()[:16]}"}}',
        lambda: f'{{"id":"{"".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))}","name":"person","active":true}}',
        lambda: f'{{"identifier":"{generate_api_key()}","expires":{secrets.randbelow(9999999999)}}}',
        lambda: f'{{"status":"{secrets.choice(["success", "error", "pending"])}"}}',
    ]
    return secrets.choice(patterns)()


def generate_file_path() -> str:
    """Generate file system paths (sanitized).

    Emulates: Unix/Windows file paths
    Example: /opt/local/bin/script, C:\\Programs\\Module, ../../../etc/config
    """
    unix_paths = [
        '/opt/local/bin/script',
        '/etc/settings',
        '/var/log/activity.log',
        f'/temp/{"".join(secrets.choice(string.ascii_lowercase) for _ in range(8))}.txt',
        '../../../etc/config',
        './files/output.json',
        f'/home/someperson/{secrets.randbelow(999)}.dat',
    ]

    windows_paths = [
        'C:\\Programs\\Module',
        'C:\\Apps\\Program\\config.ini',
        f'D:\\Files\\document_{secrets.randbelow(999)}.txt',
        '..\\..\\config\\settings.xml',
    ]

    return secrets.choice(unix_paths + windows_paths)


def generate_tech_term() -> str:
    """Generate technical terms and jargon (sanitized).

    Emulates: Technical vocabulary, API terms
    Example: WebSocket connection established, QRS token verification, microservice architecture
    """
    terms = [
        'verification', 'validation', 'configuration', 'implementation',
        'documentation', 'development', 'deployment', 'transformation',
        'QRS token verification', 'XYZ validation flow',
        'RESTful API interface', 'CORS request handler',
        'WebSocket connection established', 'microservice architecture',
        'containerized deployment', 'stateless function',
        'machine learning model', 'artificial intelligence',
        'blockchain technology', 'distributed system',
    ]
    return secrets.choice(terms)


def generate_shell_command() -> str:
    """Generate shell/CLI commands (sanitized).

    Emulates: Command line instructions
    Example: clean -rf /temp/cache/*, git commit -m "Update files", chmod +x process.sh
    """
    commands = [
        'ls -la /temp',
        'git commit -m "Update files"',
        'npm install --save-dev',
        'docker run -it ubuntu bash',
        'chmod +x someprocess.sh',
        'cat /etc/config',
        'ps aux | grep somescript',
        'curl -X GET https://api.example.com/api/v1/items',
        f'connect fakeperson@{generate_ip_address()}',
        'service restart dummywebapp',
        'git checkout -b feat/dummy',
        'ls /temp/fakefiles/old',
        'git reset --soft HEAD',
    ]
    return secrets.choice(commands)


def generate_error_message() -> str:
    """Generate error messages and status codes (sanitized).

    Emulates: HTTP status codes, system errors
    Example: 401 Unauthorized, File not found, Connection timeout
    """
    errors = [
        'Error: Connection refused',
        '404 Not Found',
        '500 Internal Server Error',
        '429 Too Many Requests',
        'Access restricted',
        'Resource unavailable',
        'Timeout occurred',
        'Invalid parameters',
        'Resource not available',
        'Connection timeout',
        'Validation failed',
        'File not found',
    ]
    return secrets.choice(errors)


def generate_common_word() -> str:
    """Generate common words that might appear in data (sanitized).

    Emulates: Configuration values, variable names
    Example: manager, localhost, identifier, session, config, module
    """
    words = [
        'manager', 'person', 'test', 'hello', 'phrase', 'code',
        'localhost', 'api', 'config', 'content', 'module', 'service',
        'client', 'session', 'identifier', 'reference', 'value', 'name',
        'email', 'phone', 'address', 'country', 'city', 'company',
        'true', 'false', 'null', 'undefined', 'none', 'empty',
    ]
    return secrets.choice(words)


# Cache for words loaded from words.txt
_cached_words = None

def _load_words() -> List[str]:
    """Load words from words.txt file, with caching."""
    global _cached_words
    if _cached_words is None:
        try:
            with open('words.txt', 'r') as f:
                _cached_words = [word.strip() for word in f.readlines() if word.strip()]
        except FileNotFoundError:
            # Fallback to a small set of common words if file not found
            _cached_words = [
                'the', 'and', 'you', 'that', 'was', 'for', 'are', 'with', 'his', 'they',
                'this', 'have', 'from', 'not', 'had', 'but', 'what', 'can', 'out', 'other'
            ]
    return _cached_words


def generate_long_english_paragraph() -> str:
    """Generate a long paragraph with random English words.

    Creates 10-12 sentences with common English words from words.txt to form
    realistic-looking paragraph text. Each sentence has 12-15 words.
    """
    words = _load_words()

    # Generate 10-12 sentences
    num_sentences = secrets.randbelow(3) + 10
    sentences = []

    for _ in range(num_sentences):
        # Each sentence has 12-15 words
        sentence_length = secrets.randbelow(3) + 12
        sentence_words = []

        for i in range(sentence_length):
            word = secrets.choice(words)
            # Capitalize first word of sentence
            if i == 0:
                word = word.capitalize()
            sentence_words.append(word)

        # Join words and add period
        sentence = ' '.join(sentence_words) + '.'
        sentences.append(sentence)

    return ' '.join(sentences)


def generate_edge_case() -> str:
    """Generate edge cases and special strings.

    Example: 'a', '', ' ', 'MixedCaseTest', 'snake_case_function', '😀🌍🚀'
    """
    cases = [
        'a',  # single character
        '  leading spaces',
        'trailing spaces  ',
        'multiple  spaces  between',
        'tab\tseparated\tvalues',
        'line\nbreak\ntest',
        '!@#$%^&*()_+-=[]{}|;\':",./<>?',  # all special chars
        '😀🌍🚀',  # emoji
        'MixedCaseTest',
        'camelCaseVariable',
        'snake_case_function',
        'kebab-case-identifier',
        'SCREAMING_SNAKE_CASE',
        '0',  # single digit
        '42', # answer to everything
        '1234567890',
        'aaa', # repeated chars
        'abcabcabc', # repeated pattern
    ]
    return secrets.choice(cases)


def generate_agent_message() -> str:
    """Generate structured AI agent communication messages (sanitized).

    Emulates: Various AI agent communication protocols
    Examples:
    - signal_type:instruction,signal:2027-07-06,operation:organize,target:instance,priority:standard
    - {"worker_id":"worker_5520","action":"process","resource":"settings"}
    - PING --async --fast 43K5B2NX
    - lat:45.036,lng:32.887,elev:5194,data:3YJKIXKVRIT5
    """
    # Different message formats that AI agents might use
    formats = [
        # Key-value format
        lambda: _generate_kv_message(),
        # JSON-like format
        lambda: _generate_json_message(),
        # Command protocol format
        lambda: _generate_command_message(),
        # Coordinate/location format
        lambda: _generate_coordinate_message(),
        # Configuration format
        lambda: _generate_config_message(),
        # URL parameter format
        lambda: _generate_url_param_message(),
        # XML-like tag format
        lambda: _generate_xml_message(),
        # CSV-like format
        lambda: _generate_csv_message(),
    ]

    return secrets.choice(formats)()


def _generate_kv_message() -> str:
    """Generate key-value pair message (sanitized).

    Emulates: Structured agent communication
    Example: signal_type:instruction,signal:2027-07-06,operation:organize,target:instance,priority:standard
    """
    actions = ['process', 'analyze', 'transform', 'convert', 'examine', 'organize', 'format', 'validate']
    targets = ['service', 'storage', 'connection', 'module', 'interface', 'channel', 'component', 'instance']
    priorities = ['low', 'medium', 'high', 'urgent', 'standard']

    # Generate random date
    year = secrets.randbelow(10) + 2025
    month = secrets.randbelow(12) + 1
    day = secrets.randbelow(28) + 1
    date = f"{year}-{month:02d}-{day:02d}"

    return f"signal_type:instruction,signal:{date},operation:{secrets.choice(actions)},target:{secrets.choice(targets)},priority:{secrets.choice(priorities)}"


def _generate_json_message() -> str:
    """Generate JSON-like structured message (sanitized).

    Emulates: API payloads, structured data
    Example: {"worker_id":"worker_5520","timestamp":2130793260,"action":"process","resource":"settings","payload":"yT7BpDO9QIw5STEx","status":"completed"}
    """
    import json

    actions = ['process', 'retrieve', 'modify', 'remove', 'synchronize', 'archive', 'restore', 'transfer']
    resources = ['profile_data', 'settings', 'configuration', 'records', 'storage', 'session', 'identifier', 'reference']

    message = {
        'worker_id': f"worker_{secrets.randbelow(9999):04d}",
        'timestamp': secrets.randbelow(1999999999) + 1000000000,
        'action': secrets.choice(actions),
        'resource': secrets.choice(resources),
        'payload': ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16)),
        'status': secrets.choice(['pending', 'active', 'completed', 'failed'])
    }

    return json.dumps(message, separators=(',', ':'))


def _generate_command_message() -> str:
    """Generate command protocol message (sanitized).

    Emulates: CLI protocols, system commands
    Example: PING --async --fast 43K5B2NX, PROC --batch PAYLOAD123
    """
    commands = ['LOAD', 'PROC', 'DONE', 'SYNC', 'VERIFY', 'PING', 'DATA', 'CTRL']
    params = ['--batch', '--quiet', '--verbose', '--async', '--check', '--fast', '--debug']

    cmd = secrets.choice(commands)
    param_count = secrets.randbelow(3) + 1
    selected_params = [secrets.choice(params) for _ in range(param_count)]
    payload = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))

    return f"{cmd} {' '.join(selected_params)} {payload}"


def _generate_coordinate_message() -> str:
    """Generate coordinate/location-based message.

    Emulates: GPS data, sensor readings
    Example: lat:45.036,lng:32.887,elev:5194,data:3YJKIXKVRIT5,time:706859480
    """
    # Random coordinates and encoded data
    lat = round((secrets.randbelow(180000) - 90000) / 1000.0, 6)
    lng = round((secrets.randbelow(360000) - 180000) / 1000.0, 6)
    elevation = secrets.randbelow(9999)

    encoded_data = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(12))

    return f"lat:{lat},lng:{lng},elev:{elevation},data:{encoded_data},time:{secrets.randbelow(999999999)}"


def _generate_config_message() -> str:
    """Generate configuration-style message (sanitized).

    Emulates: INI files, configuration blocks
    Example:
    [connection]
    host=192.168.1.1
    port=8080
    enabled=true
    """
    sections = ['connection', 'settings', 'storage', 'validation', 'memory', 'recording']
    section = secrets.choice(sections)

    configs = [
        f"host={generate_ip_address()}",
        f"port={secrets.randbelow(65535)}",
        f"timeout={secrets.randbelow(300)}",
        f"retries={secrets.randbelow(10)}",
        f"key={''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16))}",
        f"enabled={'true' if secrets.choice([True, False]) else 'false'}"
    ]

    # Ensure we don't sample more than available
    num_to_select = min(secrets.randbelow(len(configs)) + 2, len(configs))
    selected_configs = random.sample(configs, num_to_select)

    return f"[{section}]\n" + "\n".join(selected_configs)


def _generate_url_param_message() -> str:
    """Generate URL parameter-style message (sanitized).

    Emulates: Query strings, API parameters
    Example: ?identifier=ABC123&session=XYZ789&operation=read&resource=profile&ts=1234567890
    """
    params = {
        'identifier': ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32)),
        'session': ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(16)),
        'operation': secrets.choice(['read', 'write', 'remove', 'update', 'create']),
        'resource': secrets.choice(['profile', 'module', 'settings', 'content', 'record']),
        'ts': str(secrets.randbelow(1999999999) + 1000000000),
        'sig': ''.join(secrets.choice(string.ascii_letters + string.digits + '+/') for _ in range(20))
    }

    # Select random subset of parameters (ensure we don't sample more than available)
    num_to_select = min(secrets.randbelow(len(params)) + 3, len(params))
    selected_params = dict(random.sample(list(params.items()), num_to_select))

    param_string = '&'.join(f"{k}={v}" for k, v in selected_params.items())
    return f"?{param_string}"


def _generate_xml_message() -> str:
    """Generate XML-like tag message (sanitized).

    Emulates: XML structures, markup data
    Example: <instruction type="process" timestamp="2936610960" id="id_4134">eujEwdrcawgsFXtb7U5j</instruction>
    """
    tags = ['instruction', 'content', 'validation', 'request', 'response', 'message']
    tag = secrets.choice(tags)

    attributes = []
    attr_names = ['id', 'type', 'priority', 'encrypted', 'timestamp']
    # Ensure we don't sample more than available
    num_attrs = min(secrets.randbelow(len(attr_names)) + 1, len(attr_names))
    for attr in random.sample(attr_names, num_attrs):
        if attr == 'id':
            val = f"id_{secrets.randbelow(9999)}"
        elif attr == 'timestamp':
            val = str(secrets.randbelow(1999999999) + 1000000000)
        elif attr == 'encrypted':
            val = 'true' if secrets.choice([True, False]) else 'false'
        else:
            val = ''.join(secrets.choice(string.ascii_lowercase) for _ in range(6))
        attributes.append(f'{attr}="{val}"')

    content = ''.join(secrets.choice(string.ascii_letters + string.digits + ' ') for _ in range(20)).strip()

    attr_str = ' ' + ' '.join(attributes) if attributes else ''
    return f"<{tag}{attr_str}>{content}</{tag}>"


def _generate_csv_message() -> str:
    """Generate CSV-like structured message (sanitized).

    Emulates: CSV data, tabular formats
    Example:
    timestamp,worker,action,target,status,content
    2871521603,worker_299,check,interface,success,C7rCKuHv2Zj8
    """
    headers = ['timestamp', 'worker', 'action', 'target', 'status', 'content']

    row = [
        str(secrets.randbelow(1999999999) + 1000000000),  # timestamp
        f"worker_{secrets.randbelow(999):03d}",  # worker
        secrets.choice(['check', 'link', 'process', 'send', 'receive']),  # action
        secrets.choice(['interface', 'service', 'client', 'bridge', 'relay']),  # target
        secrets.choice(['success', 'pending', 'error', 'timeout']),  # status
        ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))  # data
    ]

    return ','.join(headers) + '\n' + ','.join(row)


def generate_hex_string(length: int = 64) -> str:
    """Generate a hex string of specified length.

    Example: 57aa17b18b934b60f5a0a2bff2fd3422 (32-char), f3fe5063d8bf94be093440c1d8e2e850 (64-char)
    """
    hex_chars = string.hexdigits.lower()[:16]
    return ''.join(secrets.choice(hex_chars) for _ in range(length))


def generate_base64_blob(length: int = 128) -> str:
    """Generate a base64-like blob.

    Example: SGVsbG8gV29ybGQ=, QWxsIHlvdXIgYmFzZSBhcmUgYmVsb25nIHRvIHVz
    """
    b64_chars = string.ascii_letters + string.digits + '+/='
    return ''.join(secrets.choice(b64_chars) for _ in range(length))


def generate_dataset(output_file: str = None, num_each: int = 5) -> List[dict]:
    """
    Generate a dataset of complex strings (sanitized to avoid content filtering).

    Creates realistic but non-suspicious strings that emulate:
    - Cryptographic keys and tokens
    - Structured data formats (JSON, XML, CSV, INI)
    - System identifiers and file paths
    - Agent communication protocols
    - Technical terminology and commands

    Args:
        output_file: Optional file path to write JSONL output
        num_each: Number of each type of string to generate

    Returns:
        List of dictionaries with 'text' and 'type' fields
    """
    generators = [
        ("uuid", "UUID", generate_uuid),
        ("aws_access_key", "Access Key ID (Sanitized)", generate_aws_access_key),
        ("aws_secret_key", "Secret Access Key (Sanitized)", generate_aws_secret_key),
        ("github_token", "Personal Access Token (Sanitized)", generate_github_token),
        ("api_key", "API Key", generate_api_key),
        ("bearer_token", "Bearer Token", generate_bearer_token),
        ("jwt", "JWT", generate_jwt),
        ("password", "Passphrase (Sanitized)", generate_password),
        ("user_credentials", "User Info (Sanitized)", generate_user_credentials),
        ("agent_message_kv", "Agent Message (Key-Value, Sanitized)", _generate_kv_message),
        ("agent_message_json", "Agent Message (JSON, Sanitized)", _generate_json_message),
        ("agent_message_command", "Agent Message (Command, Sanitized)", _generate_command_message),
        ("agent_message_coordinate", "Agent Message (Coordinate)", _generate_coordinate_message),
        ("agent_message_config", "Agent Message (Config, Sanitized)", _generate_config_message),
        ("agent_message_url_param", "Agent Message (URL Param, Sanitized)", _generate_url_param_message),
        ("agent_message_xml", "Agent Message (XML, Sanitized)", _generate_xml_message),
        ("agent_message_csv", "Agent Message (CSV, Sanitized)", _generate_csv_message),
        ("email", "Email Address", generate_email),
        ("url", "URL", generate_url),
        ("json_object", "JSON Object (Sanitized)", generate_json_object),
        ("file_path", "File Path (Sanitized)", generate_file_path),
        ("tech_term", "Technical Term (Sanitized)", generate_tech_term),
        ("shell_command", "Shell Command (Sanitized)", generate_shell_command),
        ("error_message", "Error Message (Sanitized)", generate_error_message),
        ("common_word", "Common Word (Sanitized)", generate_common_word),
        ("edge_case", "Edge Case", generate_edge_case),
        ("hex_32", "Hex String (32)", lambda: generate_hex_string(32)),
        ("hex_64", "Hex String (64)", lambda: generate_hex_string(64)),
        ("base64_blob", "Base64 Blob", generate_base64_blob),
        ("ssh_public_key", "Public Key Format (Sanitized)", generate_ssh_public_key),
        ("medium_b64", "Medium Base64 String", generate_medium_b64),
        ("long_b64", "Long Base64 String", generate_long_b64),
        ("long_english_paragraph", "Long English Paragraph", generate_long_english_paragraph),
    ]

    dataset = []

    for type_id, name, generator in generators:
        for _ in range(num_each):
            text = generator()
            dataset.append({"text": text, "type": type_id})

    # Add a few private key formats (they're multi-line, sanitized)
    for _ in range(num_each):
        text = generate_ssh_private_key()
        dataset.append({"text": text, "type": "private_key_format"})

    # Write to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        print(f"Generated {len(dataset)} samples and wrote to {output_file}", file=sys.stderr)

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate realistic secret strings using true randomness'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (JSONL format). If not specified, prints to stdout.'
    )
    parser.add_argument(
        '-n', '--num-each',
        type=int,
        default=5,
        help='Number of each type of secret to generate (default: 5)'
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output (only for stdout)'
    )

    args = parser.parse_args()

    dataset = generate_dataset(output_file=args.output, num_each=args.num_each)

    # If no output file, print to stdout
    if not args.output:
        for item in dataset:
            if args.pretty:
                print(json.dumps(item, indent=2))
            else:
                print(json.dumps(item))


if __name__ == '__main__':
    main()

