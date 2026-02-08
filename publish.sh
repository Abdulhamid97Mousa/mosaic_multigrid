#!/bin/bash
# Quick publish script for mosaic_multigrid to PyPI
# Author: Abdulhamid Mousa <mousa.abdulhamid97@gmail.com>

set -e  # Exit on error

echo "🚀 Publishing mosaic_multigrid to PyPI"
echo "======================================="
echo ""

# Get current version from pyproject.toml
VERSION=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
echo "📦 Current version: $VERSION"
echo ""

# Confirm
read -p "Publish version $VERSION to PyPI? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted"
    exit 1
fi

echo ""
echo "Step 1: Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info mosaic_multigrid.egg-info
echo "✓ Clean"

echo ""
echo "Step 2: Running tests..."
pytest tests/ -q || { echo "❌ Tests failed!"; exit 1; }
echo "✓ All tests passed"

echo ""
echo "Step 3: Building distribution..."
python -m build || { echo "❌ Build failed!"; exit 1; }
echo "✓ Build complete"

echo ""
echo "Step 4: Checking package..."
twine check dist/* || { echo "❌ Package check failed!"; exit 1; }
echo "✓ Package OK"

echo ""
echo "Step 5: Uploading to PyPI..."
echo "You will be prompted for credentials:"
echo "  Username: __token__"
echo "  Password: [your API token]"
echo ""

twine upload dist/* || { echo "❌ Upload failed!"; exit 1; }

echo ""
echo "🎉 SUCCESS! Package published to PyPI"
echo "📍 https://pypi.org/project/mosaic-multigrid/$VERSION/"
echo ""
echo "Users can now install with:"
echo "  pip install mosaic-multigrid"
echo ""
