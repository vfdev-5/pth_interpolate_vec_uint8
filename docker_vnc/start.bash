#!/bin/bash

exec /bin/tini -- supervisord -n -c /etc/supervisor/supervisord.conf
