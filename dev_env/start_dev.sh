#!/bin/bash

# Create .env file if it doesn't exist
cat > dev_env/.env << EOL
# Django settings
DEBUG=True
DJANGO_SECRET_KEY=django-insecure-change-this-in-production
DJANGO_SETTINGS_MODULE=config.settings.base

# Database settings
DATABASE_URL=sqlite:///dev_env/db/db.sqlite3

# Development settings
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1
EOL

# Load environment variables
set -a
source dev_env/.env
set +a

# Create database directory if it doesn't exist
mkdir -p dev_env/db

# Initialize and update git submodules
git submodule update --init --recursive

# Install dependencies if needed
poetry install --with test

# Run migrations
cd src
poetry run python manage.py migrate

# Start development server with auto-reload
poetry run python manage.py runserver 0.0.0.0:8000 