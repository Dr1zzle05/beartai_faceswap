import os
import json
import time
import random
import requests
import datetime
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from qcloud_cos import CosConfig, CosS3Client
from dotenv import load_dotenv

# 加载本地.env文件（仅本地开发有用，Railway 上用环境变量）
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 从环境变量中获取配置
region = os.getenv('REGION')
secret_id = os.getenv('SECRET_ID')
secret_key = os.getenv('SECRET_KEY')
bucket = os.getenv('BUCKET')
image_output_path = os.getenv('IMAGE_OUTPUT_PATH', './output')
product_serial = os.getenv('PRODUCT_SERIAL')
valid_tokens = os.getenv('VALID_TOKENS', 'token1,token2').split(',')

app = FastAPI(title="BeArt AI Face Swap API", description="AI换脸服务API", version="0.2")

class FaceSwapService:
    API_HEADERS = {
        "accept": "*/*",
        "accept-language": "zh-CN,zh;q=0.9",
        "origin": "https://beart.ai",
        "priority": "u=1, i",
        "product-code": "067003",
        "referer": "https://beart.ai/",
        "sec-ch-ua": '"Google Chrome";v="129", "Not=A?Brand";v="8", "Chromium";v="129"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
    }

    @classmethod
    def _validate_image(cls, image_data: bytes) -> bool:
        header = image_data[:12]
        return any([
            header.startswith(b'\xFF\xD8\xFF'),
            header.startswith(b'\x89PNG\r\n\x1a\n'),
            header.startswith(b'GIF87a'), header.startswith(b'GIF89a'),
            header.startswith(b'RIFF') and header[8:12] == b'WEBP',
            header.startswith(b'BM')
        ])

    @classmethod
    def _get_mime_type(cls, image_data: bytes) -> str:
        header = image_data[:12]
        if header.startswith(b'\xFF\xD8\xFF'): return 'image/jpeg'
        if header.startswith(b'\x89PNG\r\n\x1a\n'): return 'image/png'
        if header.startswith(b'GIF87a') or header.startswith(b'GIF89a'): return 'image/gif'
        if header.startswith(b'RIFF') and header[8:12] == b'WEBP': return 'image/webp'
        if header.startswith(b'BM'): return 'image/bmp'
        return 'image/jpeg'

    @classmethod
    async def create_face_swap_job(cls, source_image: bytes, target_image: bytes) -> str:
        url = "https://api.beart.ai/api/beart/face-swap/create-job"
        headers = cls.API_HEADERS.copy()
        headers["product-serial"] = product_serial

        files = {
            "target_image": (f"target.jpg", source_image, cls._get_mime_type(source_image)),
            "swap_image": (f"source.jpg", target_image, cls._get_mime_type(target_image)),
        }
        resp = requests.post(url, headers=headers, files=files)
        if resp.status_code == 200 and resp.json().get("code") == 100000:
            return resp.json()["result"]["job_id"]
        raise Exception(f"BeArt 任务创建失败: {resp.text}")

    @classmethod
    async def get_face_swap_result(cls, job_id: str, retries: int = 30, interval: int = 2) -> str:
        url = f"https://api.beart.ai/api/beart/face-swap/get-job/{job_id}"
        headers = cls.API_HEADERS.copy()
        for _ in range(retries):
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == 100000:
                    return data["result"]["output"][0]
                if data.get("code") == 300001:
                    time.sleep(interval)
                    continue
            raise Exception("换脸结果获取失败")
        raise Exception("超过最大重试次数")


def verify_auth_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token not in valid_tokens:
        raise HTTPException(status_code=403, detail="Invalid or Expired Token")
    return token


def generate_filename(ext='jpg'):
    return f"faceswap_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}.{ext}"


def download_image(url, output_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    if not os.path.exists(output_path): os.makedirs(output_path)
    filename = generate_filename()
    path = os.path.join(output_path, filename)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    return filename, path


def upload_to_cos(region, secret_id, secret_key, bucket, file_name, base_path):
    client = CosS3Client(CosConfig(Region=region, SecretId=secret_id, SecretKey=secret_key))
    local_path = os.path.join(base_path, file_name)
    resp = client.upload_file(Bucket=bucket, LocalFilePath=local_path, Key=file_name)
    if 'ETag' in resp:
        return f"https://{bucket}.cos.{region}.myqcloud.com/{file_name}"
    raise Exception("上传失败")


@app.post("/beartAI/face-swap")
async def face_swap(source_image: UploadFile = File(...), target_image: UploadFile = File(...), auth_token: str = Depends(verify_auth_token)):
    try:
        source_data = await source_image.read()
        target_data = await target_image.read()
        if not FaceSwapService._validate_image(source_data) or not FaceSwapService._validate_image(target_data):
            raise HTTPException(status_code=400, detail="图片格式不支持")

        job_id = await FaceSwapService.create_face_swap_job(source_data, target_data)
        result_url = await FaceSwapService.get_face_swap_result(job_id)
        filename, path = download_image(result_url, image_output_path)
        cos_url = upload_to_cos(region, secret_id, secret_key, bucket, filename, image_output_path)
        os.remove(path)
        return {"success": True, "image_url": cos_url, "original_url": result_url}

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8089)
