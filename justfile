# Load .env file if it exists
set dotenv-load
set dotenv-path := "dev_env/.env"

# List available commands
default:
    @just --list

# Create initial .env file
create-env:
    #!/bin/bash
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

# Initialize and update git submodules
init-submodules:
    git submodule update --init --recursive

# Install development dependencies
install: init-submodules
    poetry install --with test

# Run database migrations
migrate:
    cd src && poetry run python manage.py migrate

# Create a new migration
makemigrations app="":
    cd src && poetry run python manage.py makemigrations {{app}}

# Start development server
run: migrate
    cd src && poetry run python manage.py runserver 0.0.0.0:8000

# Create a superuser
createsuperuser:
    cd src && poetry run python manage.py createsuperuser

# Run tests
test:
    cd src && poetry run python manage.py test

# Run tests with coverage
coverage:
    cd src && poetry run coverage run manage.py test
    cd src && poetry run coverage report
    cd src && poetry run coverage html

# Clean Python cache files
clean:
    find . -type d -name "__pycache__" -exec rm -r {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type f -name ".coverage" -delete
    find . -type d -name "*.egg-info" -exec rm -r {} +
    find . -type d -name "*.egg" -exec rm -r {} +
    find . -type d -name ".pytest_cache" -exec rm -r {} +
    find . -type d -name ".coverage*" -exec rm -r {} +
    find . -type d -name "htmlcov" -exec rm -r {} +

# Run pre-commit hooks
lint:
    pre-commit run --all-files

# Shell
shell:
    cd src && poetry run python manage.py shell

# Reset database
reset-db: clean
    rm -f dev_env/db/db.sqlite3
    mkdir -p dev_env/db
    just migrate

# Full development setup
setup: create-env init-submodules install migrate 