#!/bin/bash

echo "🚀 Setting up environment for Vietnamese Legal NLP..."

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Pre-download heavy models
echo "📥 Downloading Vietnamese NLP models (Stanza)..."
python -c "import stanza; stanza.download('vi')"

echo "✅ Setup complete! Run 'source venv/bin/activate' to activate the environment."