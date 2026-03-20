"""
Django settings for config project.
Supports both development (DEBUG=True) and production via environment variables.
"""

import os
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    'DJANGO_SECRET_KEY',
    'django-insecure-dev-only-change-in-production-1m25dft8f^hiki3n78x^6a1!'
)

DEBUG = os.environ.get('DJANGO_DEBUG', 'True') == 'True'

# Accept comma OR space separated hosts, plus always allow localhost in dev
_raw_hosts = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost 127.0.0.1')
ALLOWED_HOSTS = [h.strip() for h in _raw_hosts.replace(',', ' ').split() if h.strip()]

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'django.contrib.humanize',
    'analytics',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
        'OPTIONS': {
            'init_command': 'PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;',
        },
    }
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ---------------------------------------------------------------------------
# Password validation
# ---------------------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ---------------------------------------------------------------------------
# Internationalisation
# ---------------------------------------------------------------------------
LANGUAGE_CODE = 'en-us'
TIME_ZONE     = 'UTC'
USE_I18N      = True
USE_TZ        = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
STATIC_URL   = '/static/'
STATIC_ROOT  = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# ---------------------------------------------------------------------------
# Auth / Login
# ---------------------------------------------------------------------------
LOGIN_URL             = 'login'
LOGIN_REDIRECT_URL    = 'analytics:dashboard'
LOGOUT_REDIRECT_URL   = 'login'

# ---------------------------------------------------------------------------
# Session security
# ---------------------------------------------------------------------------
SESSION_COOKIE_HTTPONLY = True   # Session cookie: JS must never read this
SESSION_COOKIE_SAMESITE = 'Lax'
# CSRF cookie must be readable by JavaScript for AJAX requests (fetch/XHR).
# HttpOnly=True would block JS from reading it — Django's own docs say False here.
CSRF_COOKIE_HTTPONLY    = False

# ---------------------------------------------------------------------------
# Warning filters — suppress known harmless sklearn/joblib warnings
# ---------------------------------------------------------------------------
warnings.filterwarnings('ignore', message='.*sklearn.*parallel.*delayed.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*joblib.*', category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
#
# django.server  → every HTTP request line, e.g. "GET /dashboard/ 200 12ms"
# django.request → 4xx/5xx errors with tracebacks
# analytics      → INFO+ from your own app code (views, ml pipeline)
# ---------------------------------------------------------------------------
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'request': {
            # Clean format matching Django's default dev-server output
            'format': '[{asctime}] {message}',
            'datefmt': '%d/%b/%Y %H:%M:%S',
            'style': '{',
        },
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'console_request': {
            'class': 'logging.StreamHandler',
            'formatter': 'request',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'WARNING',
    },
    'loggers': {
        # Prints every  GET /url/ STATUS  line — this was missing before
        'django.server': {
            'handlers': ['console_request'],
            'level': 'INFO',
            'propagate': False,
        },
        # 4xx / 5xx with full tracebacks
        'django.request': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False,
        },
        # Internal Django messages (migrations, system checks)
        'django': {
            'handlers': ['console'],
            'level': os.environ.get('DJANGO_LOG_LEVEL', 'WARNING'),
            'propagate': False,
        },
        # Your app code — use logging.getLogger('analytics') anywhere
        'analytics': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}