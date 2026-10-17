"""
Shared test setup.

The worker reads its configuration at import time and calls load_dotenv(), which would
pull the repo's local .env (real credentials and live topic names) into the test process.
dotenv is therefore always replaced with a no-op, live-call env vars are blanked, and the
cloud SDKs are stubbed when they are not installed so the pure parts can be tested on any
interpreter.
"""
import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, 'tests', 'fixtures')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_dotenv = types.ModuleType('dotenv')
_dotenv.load_dotenv = lambda *args, **kwargs: False
sys.modules['dotenv'] = _dotenv

for _var in ('PROGRESS_TOPIC_ID', 'RESULTS_TOPIC_ID', 'PROJECT_ID', 'SUBSCRIPTION_ID', 'BUCKET_NAME',
             'YTDLP_PROXY', 'REPLICATE_API_TOKEN', 'MAX_VIDEO_DURATION_SECONDS'):
    os.environ.pop(_var, None)
os.environ['YOUTUBE_COOKIES_FILE'] = os.path.join(REPO_ROOT, 'tests', 'no-such-cookies.txt')


def _stub_if_missing(name, attrs):
    try:
        importlib.import_module(name)
    except ImportError:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        parent, _, child = name.rpartition('.')
        if parent:
            setattr(sys.modules[parent], child, module)


_stub_if_missing('replicate', {'run': lambda *a, **k: None})

class _DownloadError(Exception):
    pass


_stub_if_missing('yt_dlp', {'YoutubeDL': object, 'utils': types.SimpleNamespace(DownloadError=_DownloadError)})
_stub_if_missing('google', {})
_stub_if_missing('google.cloud', {})
_stub_if_missing('google.cloud.pubsub_v1', {'PublisherClient': object, 'SubscriberClient': object,
                                            'types': types.SimpleNamespace(FlowControl=dict)})
_stub_if_missing('google.cloud.storage', {'Client': object})


def load_script(filename, module_name):
    """Import a repo script whose file name is not a valid module name."""
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(REPO_ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def processor():
    return load_script('cloud-vm-processor.py', 'cloud_vm_processor')
