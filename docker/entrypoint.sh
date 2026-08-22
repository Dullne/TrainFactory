#!/bin/sh
set -eu

if [ "${MYSQL_URL_FILE+x}" = x ]; then
    if [ -z "$MYSQL_URL_FILE" ]; then
        echo "secret source environment is invalid" >&2
        exit 1
    fi
    if [ "${MYSQL_URL+x}" = x ]; then
        if [ -z "$MYSQL_URL" ]; then
            unset MYSQL_URL
        else
            echo "secret source environment is invalid" >&2
            exit 1
        fi
    fi
fi

if [ "${JWT_SECRET_KEY_FILE+x}" = x ]; then
    if [ -z "$JWT_SECRET_KEY_FILE" ]; then
        echo "secret source environment is invalid" >&2
        exit 1
    fi
    if [ "${JWT_SECRET_KEY+x}" = x ]; then
        if [ -z "$JWT_SECRET_KEY" ]; then
            unset JWT_SECRET_KEY
        else
            echo "secret source environment is invalid" >&2
            exit 1
        fi
    fi
fi

if [ "${DEFAULT_ADMIN_PASSWORD_FILE+x}" = x ]; then
    if [ -z "$DEFAULT_ADMIN_PASSWORD_FILE" ]; then
        echo "secret source environment is invalid" >&2
        exit 1
    fi
    if [ "${DEFAULT_ADMIN_PASSWORD+x}" = x ]; then
        if [ -z "$DEFAULT_ADMIN_PASSWORD" ]; then
            unset DEFAULT_ADMIN_PASSWORD
        else
            echo "secret source environment is invalid" >&2
            exit 1
        fi
    fi
fi

if [ "${MYSQL_URL+x}" != x ] && [ "${MYSQL_URL_FILE+x}" != x ]; then
    if [ "${MYSQL_APP_USER+x}" = x ] || \
       [ "${MYSQL_APP_PASSWORD+x}" = x ] || \
       [ "${MYSQL_HOST+x}" = x ] || \
       [ "${MYSQL_DATABASE+x}" = x ]; then
        : "${MYSQL_APP_USER:?MYSQL_APP_USER is required}"
        : "${MYSQL_APP_PASSWORD:?MYSQL_APP_PASSWORD is required}"
        : "${MYSQL_HOST:?MYSQL_HOST is required}"
        : "${MYSQL_DATABASE:?MYSQL_DATABASE is required}"
        MYSQL_URL="$(python -c '
import os
from urllib.parse import quote

user = quote(os.environ["MYSQL_APP_USER"], safe="")
password = quote(os.environ["MYSQL_APP_PASSWORD"], safe="")
host = quote(os.environ["MYSQL_HOST"], safe="[].:")
database = quote(os.environ["MYSQL_DATABASE"], safe="")
print(f"mysql+pymysql://{user}:{password}@{host}:3306/{database}")
')"
        export MYSQL_URL
    fi
fi

gpu_preflight_mode="${GPU_PREFLIGHT_MODE:-required}"
if [ -n "${NVIDIA_DISABLE_REQUIRE:-}" ] && [ "$gpu_preflight_mode" != "required" ]; then
    echo "GPU compatibility policy is invalid" >&2
    exit 1
fi

python -m train_factory.runtime.gpu_preflight --mode "$gpu_preflight_mode"
exec "$@"
