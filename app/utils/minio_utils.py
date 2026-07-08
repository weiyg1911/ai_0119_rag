from minio import Minio

from app.conf.minio_config import minio_config
from app.core.logger import logger, node_log

_minio_client = None

def _create_minio_client():
    client = Minio(
        minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.secure,
    )
    return client

def _create_minio_bucket():
    client = get_minio_client()
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": "*",
                    },
                    "Resource": "arn:aws:s3:::{}".format(minio_config.bucket_name),
                }
            ]
        }
        client.set_bucket_policy(minio_config.bucket_name, policy)
    else:
        logger.info("Minio bucket already exists")


def get_minio_client():
    global _minio_client
    if _minio_client is None:
        client = _create_minio_client()
        _minio_client = client
        return _minio_client
    else:
        return _minio_client