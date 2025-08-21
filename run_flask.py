#!/usr/bin/env python3
"""
Flask Invoice PII Masker Application Runner
Start the Flask application with proper configuration
"""

import os
import sys
from app import app

if __name__ == '__main__':
    print("🛡️ Starting Invoice PII Masker Flask Application...")
    print("📍 Server will be available at: http://localhost:5000")
    print("📁 Make sure Tesseract OCR is installed on your system")
    print("=" * 60)
    
    # Create temp directory if it doesn't exist
    os.makedirs('temp_uploads', exist_ok=True)
    
    try:
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5000,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
