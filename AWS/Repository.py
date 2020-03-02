from typing import List, Dict

import boto3
from botocore.exceptions import ClientError
import os


# Repository class
class Repository:
    s3_client = None
    s3_bucket = None
    local_dir = ''
    bucket_name = ''

    # Initialize the Repository Object
    def __init__(self,
                 region='eu-central-1',
                 bucket_name='rec-sys-2020-polimi',
                 local_dir='aws_local'):

        self.bucket_name = bucket_name
        self.local_dir = os.path.join(local_dir, bucket_name)
        self._init_bucket()
        self._init_local_dir()

        try:
            self.s3_client = boto3.client('s3', region_name=region)
            location = {'LocationConstraint': region}
            self.s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration=location
            )
        except ClientError as e:
            pass

    def list_remote(self) -> List[str]:
        return [object.key for object in self.s3_bucket.objects.all()]

    def list_local(self) -> List[str]:
        return os.listdir(self.local_dir)

    def resource_diff(self) -> Dict:
        local_resources = self.list_local()
        remote_resources = self.list_remote()

        to_be_uploaded_resources = [local for local in local_resources if local not in remote_resources]
        to_be_downloaded_resources = [remote for remote in remote_resources if remote not in local_resources]

        return {'to_be_uploaded_resources': to_be_uploaded_resources,
                'to_be_downloaded_resources': to_be_downloaded_resources}

    def upload(self, resource: str):
        file_path = os.path.join(self.local_dir, resource)
        with open(file_path, "rb") as f:
            self.s3_client.upload_fileobj(f, self.bucket_name, resource)

    def download(self, resource: str):
        file_path = os.path.join(self.local_dir, resource)
        with open(file_path, "wb") as f:
            self.s3_client.download_fileobj(self.bucket_name, resource, f)

    def download_multiple(self, resource_pattern):
        diff = self.resource_diff()

        to_be_downloaded = diff['to_be_downloaded_resources']

        to_be_downloaded = [resource
                            for resource in to_be_downloaded
                            if resource_pattern in resource]

        [self.download(resource) for resource in to_be_downloaded]

    def sync_all(self):
        diff = self.resource_diff()

        to_be_downloaded = diff['to_be_downloaded_resources']
        [print(f"---> Downloading from {self.bucket_name} ---> {resource}")for resource in to_be_downloaded]
        [self.download(resource) for resource in to_be_downloaded]

        to_be_uploaded = diff['to_be_uploaded_resources']
        [print(f"<--- Uploading to {self.bucket_name} <--- {resource}")for resource in to_be_uploaded]
        [self.upload(resource) for resource in to_be_uploaded]

    def _init_bucket(self):
        s3 = boto3.resource('s3')
        self.s3_bucket = s3.Bucket(self.bucket_name)

    def _init_local_dir(self):
        if not os.path.isdir(self.local_dir):
            os.makedirs(self.local_dir)
