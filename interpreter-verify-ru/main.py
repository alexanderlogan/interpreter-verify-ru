"""
Interpreter-Verify-RU
Medical Translation and Terminology Verification Tool

Entry point for the application.
"""
import sys
from config import config


def check_prerequisites():
    """Verify all required components are available before starting."""
    errors = []

    # Check Python version
    if sys.version_info < (3, 10):
        errors.append(
            f"Python 3.10+ required. You have {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )

    # Check Ollama is running
    try:
        import requests
        response = requests.get(
            f"{config.ollama.base_url}/api/tags",
            timeout=5
        )
        if response.status_code != 200:
            errors.append("Ollama is not responding. Run: ollama serve")
    except Exception:
        errors.append(
            "Cannot connect to Ollama. Make sure it is running.\n"
            "  Start it with: ollama serve"
        )

    # Check pharma database exists
    if not config.pharma.database_path.exists():
        errors.append(
            f"Pharma database not found at {config.pharma.database_path}\n"
            f"  Make sure pharma_map.json is in the src/pharma/ directory."
        )

    if errors:
        print("=" * 60)
        print("STARTUP CHECK FAILED")
        print("=" * 60)
        for i, error in enumerate(errors, 1):
            print(f"\n  {i}. {error}")
        print("\n" + "=" * 60)
        print("Fix the issues above and try again.")
        sys.exit(1)


def main():
    """Application entry point."""
    print(f"\n  {config.app_name} v{config.version}")
    print(f"  Medical Translation + Terminology Verification")
    print(f"  All processing is LOCAL. No data leaves this machine.\n")

    check_prerequisites()

    print("  [OK] Prerequisites verified.")
    print("  [OK] Ollama is running.")
    print("  [OK] Pharma database loaded.")
    print()

    # Phase 1: Audio capture will be implemented here
    # Phase 2: Whisper transcription will be added here
    # Phase 3: Ollama translation will be added here
    # Phase 4: Pharma lookup will be added here
    # Phase 5: UI overlay will be added here

    print("  Application is ready.")
    print("  (Core features will be added phase by phase.)")
    print("  Press Ctrl+C to exit.\n")

    try:
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Shutting down. Goodbye.")


if __name__ == "__main__":
    main()
