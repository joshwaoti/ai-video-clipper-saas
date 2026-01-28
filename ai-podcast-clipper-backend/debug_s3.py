
import modal
import boto3
import os

# Create a minimal app to debug S3
image = modal.Image.debian_slim().pip_install("boto3")
app = modal.App("debug-s3", image=image)

@app.function(secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")])
def list_bucket_files(bucket_name, prefix):
    print(f"Checking bucket: {bucket_name} with prefix: {prefix}")
    s3 = boto3.client("s3")
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' in response:
            print(f"Generously listing files in {bucket_name}/{prefix}:")
            for obj in response['Contents']:
                print(f" - {obj['Key']} (Size: {obj['Size']} bytes)")
        else:
            print(f"No files found in {bucket_name} with prefix '{prefix}'.")
            
        # Also list root just in case
        if prefix != "":
            print("\nListing root files (top 5) just in case path is wrong:")
            response_root = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=5)
            if 'Contents' in response_root:
                 for obj in response_root['Contents']:
                    print(f" - {obj['Key']}")
                    
    except Exception as e:
        print(f"Error accessing S3: {e}")

@app.local_entrypoint()
def main():
    print("Running S3 Debugger...")
    # Trigger the remote function
    list_bucket_files.remote("josh-video-clipper", "test2/")
