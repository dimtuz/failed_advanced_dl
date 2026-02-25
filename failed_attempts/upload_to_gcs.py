#!/usr/bin/env python3
"""Upload uncertainty report and artifacts to Google Cloud Storage."""
import os
import json
from pathlib import Path

def main():
    bucket_name = os.environ.get("GCS_BUCKET")
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not bucket_name:
        print("GCS_BUCKET not set; skipping upload.")
        return
    if not creds_json:
        print("GOOGLE_APPLICATION_CREDENTIALS_JSON not set; skipping upload.")
        return

    # Write service account key to temp file for GCP client
    creds_path = Path("/tmp/gcp-sa-key.json")
    creds_path.write_text(creds_json)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    report_dir = Path(__file__).resolve().parent.parent / "reports"
    if not report_dir.exists():
        print(f"Reports dir {report_dir} not found; nothing to upload.")
        return

    from datetime import datetime
    prefix = datetime.utcnow().strftime("uncertainty_reports/%Y-%m-%d")
    for f in report_dir.glob("*"):
        if f.is_file():
            blob = bucket.blob(f"{prefix}/{f.name}")
            blob.upload_from_filename(str(f))
            print(f"Uploaded {f.name} -> gs://{bucket_name}/{prefix}/{f.name}")
    creds_path.unlink(missing_ok=True)
    print("Upload complete.")

if __name__ == "__main__":
    main()
