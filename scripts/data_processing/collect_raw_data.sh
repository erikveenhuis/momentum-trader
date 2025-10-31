#! /bin/bash
rclone sync s3polygon:flatfiles/global_crypto/minute_aggs_v1 ../data/raw/ --progress
